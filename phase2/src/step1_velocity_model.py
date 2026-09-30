"""
step1_velocity_model.py  –  Phase 2 Step 1
==========================================
Supervised velocity estimation:  IMU window → vehicle speed (m/s)

Architecture: 1D-CNN + GRU (lightweight, edge-deployable)
  Input:  (batch, window=20, 6-features)
  Output: (batch,) – speed in m/s  [non-negative]

Design rationale vs edge deployment:
  - Target: 10 Hz real-time on mid-range smartphone CPU
  - Budget: <10 ms per inference @ 10 Hz → budget is 100 ms; we target <5 ms
  - Architecture chosen: tiny CNN (2 layers, 32/64 ch) + GRU(32)
  - Parameter count: ~17 K  (well under 500 K limit)
  - ONNX-exportable, no dynamic shapes

Window-size empirical sweep:
  - Runs W ∈ {5, 10, 15, 20, 30} on train split
  - Reports val RMSE per W; picks the winner
  - Justification logged to outputs/reports/window_sweep.json

Training:
  - Loss: Huber (robust to velocity outliers from GPS noise in labels)
  - Optimizer: AdamW, LR=1e-3, cosine annealing
  - Batch: 256, Epochs: 60 (early stop on val RMSE, patience=10)
  - Normalisation: per-feature z-score from train set (saved for inference)
"""

import sys
import json
import time
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PHASE2_OUT  = Path(__file__).parent.parent / 'outputs'
MODELS_DIR  = PHASE2_OUT / 'models'
REPORTS_DIR = PHASE2_OUT / 'reports'
PLOTS_DIR   = PHASE2_OUT / 'plots'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────── pure-numpy model (no-torch fallback) ─────────────────
class NumpyGRUCell:
    """Minimal GRU cell in numpy for inference fallback."""
    def __init__(self, W_z, W_r, W_h, b_z, b_r, b_h):
        self.W_z, self.W_r, self.W_h = W_z, W_r, W_h
        self.b_z, self.b_r, self.b_h = b_z, b_r, b_h
    def step(self, x, h):
        z = self._sigmoid(x @ self.W_z[:x.shape[-1]].T +
                          h @ self.W_z[x.shape[-1]:].T + self.b_z)
        r = self._sigmoid(x @ self.W_r[:x.shape[-1]].T +
                          h @ self.W_r[x.shape[-1]:].T + self.b_r)
        n = np.tanh(x @ self.W_h[:x.shape[-1]].T +
                    (r*h) @ self.W_h[x.shape[-1]:].T + self.b_h)
        return (1-z)*h + z*n
    @staticmethod
    def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-15,15)))


def _relu(x): return np.maximum(0, x)


# ─────────────────────── PyTorch model definition ─────────────────────────────
def build_torch_model(window: int, n_features: int = 6,
                       cnn_ch=(32, 64), gru_h: int = 32):
    """Returns nn.Module. Imported lazily so numpy fallback works without torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class VelocityNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, cnn_ch[0], 3, padding=1)
            self.bn1   = nn.BatchNorm1d(cnn_ch[0])
            self.conv2 = nn.Conv1d(cnn_ch[0], cnn_ch[1], 3, padding=1)
            self.bn2   = nn.BatchNorm1d(cnn_ch[1])
            self.drop  = nn.Dropout(0.1)
            self.gru   = nn.GRU(cnn_ch[1], gru_h, batch_first=True)
            self.head  = nn.Sequential(
                nn.Linear(gru_h, 16), nn.ReLU(),
                nn.Linear(16, 1),     nn.Softplus()   # smooth non-negative
            )

        def forward(self, x):
            # x: (B, T, F)
            x = x.permute(0, 2, 1)                    # → (B, F, T)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.drop(x)
            x = x.permute(0, 2, 1)                    # → (B, T, C)
            _, h = self.gru(x)
            return self.head(h.squeeze(0)).squeeze(-1) # (B,)

    model = VelocityNet()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f'VelocityNet: {n_params:,} parameters  (window={window})')
    return model


# ──────────────────────── normalisation ───────────────────────────────────────
class Normalizer:
    def __init__(self):
        self.mean = None; self.std = None

    def fit(self, X: np.ndarray):
        # X: (N, T, F)
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(0)
        self.std  = flat.std(0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def save(self, path: Path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> 'Normalizer':
        n = cls()
        d = np.load(path)
        n.mean = d['mean']; n.std = d['std']
        return n


# ──────────────────────── window-size sweep ───────────────────────────────────
def window_sweep(datasets_fn, candidate_windows=(5, 10, 15, 20, 30)) -> int:
    """
    Train a micro-model at each window size, report val RMSE, return best W.
    datasets_fn: callable(window) → {'train_X', 'train_y', 'val_X', 'val_y'}
    """
    try:
        import torch, torch.nn as nn
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        log.warning('PyTorch not available – skipping window sweep; defaulting W=20')
        return 20

    results = {}
    for W in candidate_windows:
        ds = datasets_fn(W)
        trX = ds['train_X']; trY = ds['train_y']
        vaX = ds['val_X'];   vaY = ds['val_y']

        if len(trX) < 50 or len(vaX) < 10:
            log.warning(f'  W={W}: insufficient data – skip')
            continue

        norm = Normalizer(); norm.fit(trX)
        trX_n = torch.tensor(norm.transform(trX))
        trY_t = torch.tensor(trY)
        vaX_n = torch.tensor(norm.transform(vaX))
        vaY_t = torch.tensor(vaY)

        net = build_torch_model(W, cnn_ch=(16, 32), gru_h=16)   # tiny for sweep
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_rmse = float('inf')
        for ep in range(30):     # fast sweep epochs
            net.train()
            idx = torch.randperm(len(trX_n))
            for b_start in range(0, len(trX_n), 256):
                bi = idx[b_start:b_start+256]
                pred = net(trX_n[bi])
                loss = loss_fn(pred, trY_t[bi])
                opt.zero_grad(); loss.backward(); opt.step()
            # val
            net.eval()
            with torch.no_grad():
                vpred = net(vaX_n).numpy()
            rmse = float(np.sqrt(np.mean((vpred - vaY)**2)))
            best_rmse = min(best_rmse, rmse)

        results[W] = round(best_rmse, 4)
        log.info(f'  W={W:2d}: val_RMSE={best_rmse:.4f} m/s')

    if not results:
        return 20

    sweep_path = REPORTS_DIR / 'window_sweep.json'
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)

    best_W = min(results, key=results.get)
    log.info(f'Window sweep best: W={best_W} (RMSE={results[best_W]:.4f} m/s)')

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(results.keys()), list(results.values()), 'o-', color='steelblue', lw=2)
    ax.set_xlabel('Window size (samples @ 10 Hz)')
    ax.set_ylabel('Validation RMSE (m/s)')
    ax.set_title('Window Size Sweep – VelocityNet')
    ax.axvline(best_W, color='red', ls='--', label=f'Best W={best_W}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'window_sweep.png', dpi=120)
    plt.close()
    return best_W


# ──────────────────────── main training ───────────────────────────────────────
def train_velocity_model(dataset: dict, epochs: int = 60,
                          batch_size: int = 256,
                          lr: float = 1e-3) -> dict:
    """
    Train VelocityNet on dataset['train_X'/'train_y'],
    validate on dataset['val_X'/'val_y'].
    Returns dict with model, normalizer, history, metrics.
    """
    try:
        import torch, torch.nn as nn
        from torch.optim.lr_scheduler import CosineAnnealingLR
        HAS_TORCH = True
    except ImportError:
        log.error('PyTorch required for training. Install: pip3 install torch')
        return {}

    window   = dataset['window']
    trX = dataset['train_X']; trY = dataset['train_y']
    vaX = dataset['val_X'];   vaY = dataset['val_y']

    log.info(f'Training VelocityNet: train={trX.shape} val={vaX.shape} '
             f'window={window} epochs={epochs}')

    # normalise
    norm = Normalizer(); norm.fit(trX)
    trX_n = torch.tensor(norm.transform(trX), dtype=torch.float32)
    trY_t = torch.tensor(trY, dtype=torch.float32)
    vaX_n = torch.tensor(norm.transform(vaX), dtype=torch.float32)
    vaY_np = vaY.copy()

    net     = build_torch_model(window)
    opt     = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched   = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=1.5)

    train_losses, val_rmses = [], []
    best_rmse  = float('inf')
    best_state = None
    patience   = 10; no_improve = 0

    t0 = time.time()
    for ep in range(1, epochs+1):
        net.train()
        idx   = torch.randperm(len(trX_n))
        ep_loss = 0.0; n_batches = 0
        for b_start in range(0, len(trX_n), batch_size):
            bi   = idx[b_start:b_start+batch_size]
            pred = net(trX_n[bi])
            loss = loss_fn(pred, trY_t[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # validation
        net.eval()
        with torch.no_grad():
            vpred = net(vaX_n).numpy()
        rmse = float(np.sqrt(np.mean((vpred - vaY_np)**2)))
        val_rmses.append(rmse)

        if rmse < best_rmse:
            best_rmse  = rmse; no_improve = 0
            import copy; best_state = copy.deepcopy(net.state_dict())
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1:
            log.info(f'  Epoch {ep:3d}/{epochs}: loss={avg_loss:.4f}  '
                     f'val_RMSE={rmse:.4f} m/s  best={best_rmse:.4f}')
        if no_improve >= patience:
            log.info(f'  Early stop at epoch {ep} (patience={patience})')
            break

    elapsed = time.time() - t0
    log.info(f'Training done in {elapsed:.1f}s. Best val RMSE: {best_rmse:.4f} m/s')

    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        import copy
        best_state = copy.deepcopy(net.state_dict())

    # test RMSE
    teX_n = torch.tensor(norm.transform(dataset['test_X']), dtype=torch.float32)
    teY   = dataset['test_y']
    net.eval()
    with torch.no_grad():
        test_pred = net(teX_n).numpy()
    test_rmse = float(np.sqrt(np.mean((test_pred - teY)**2)))
    test_mae  = float(np.mean(np.abs(test_pred - teY)))
    log.info(f'Test RMSE: {test_rmse:.4f} m/s  MAE: {test_mae:.4f} m/s')

    # save model
    model_path = MODELS_DIR / 'velocity_net.pt'
    torch.save({'state_dict': best_state,
                'window': window, 'n_features': 6,
                'train_rmse': float(train_losses[-1]),
                'val_rmse': best_rmse,
                'test_rmse': test_rmse}, model_path)
    norm.save(MODELS_DIR / 'velocity_normalizer.npz')

    # plot loss curves
    _plot_training_curves(train_losses, val_rmses, best_rmse)

    # speed error scatter
    _plot_speed_scatter(teY, test_pred, test_rmse)

    metrics = {
        'best_val_rmse_ms': round(best_rmse, 4),
        'test_rmse_ms':     round(test_rmse, 4),
        'test_mae_ms':      round(test_mae, 4),
        'epochs_run':       len(train_losses),
        'n_params':         sum(p.numel() for p in net.parameters()),
        'window':           window,
        'training_time_s':  round(elapsed, 1),
    }
    with open(REPORTS_DIR / 'velocity_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'model': net, 'normalizer': norm,
            'metrics': metrics, 'history': {'train_loss': train_losses,
                                             'val_rmse': val_rmses}}


def _plot_training_curves(train_losses, val_rmses, best_rmse):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, 'b-', lw=1.5, label='Train Huber Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(val_rmses, 'r-', lw=1.5, label='Val RMSE (m/s)')
    ax2.axhline(best_rmse, color='green', ls='--',
                label=f'Best={best_rmse:.4f} m/s')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('RMSE (m/s)')
    ax2.set_title('Validation RMSE'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('VelocityNet Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'velocity_training_curves.png', dpi=120)
    plt.close()
    log.info('Training curve plot saved')


def _plot_speed_scatter(y_true, y_pred, rmse):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true * 3.6, y_pred * 3.6, s=3, alpha=0.3, c='steelblue')
    lim = max(y_true.max(), y_pred.max()) * 3.6 + 5
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect')
    ax.set_xlabel('True Speed (km/h)'); ax.set_ylabel('Predicted Speed (km/h)')
    ax.set_title(f'VelocityNet Test Predictions  RMSE={rmse:.4f} m/s')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'velocity_scatter.png', dpi=120)
    plt.close()


# ──────────────────────── inference helper ────────────────────────────────────
class VelocityPredictor:
    """Wraps trained model + normalizer for streaming inference at 10 Hz."""

    def __init__(self, model, normalizer, window: int):
        import torch
        self._net  = model
        self._net.eval()
        self._norm = normalizer
        self._win  = window
        self._buf  = []   # rolling buffer of raw feature vectors

    def push(self, imu_vec: np.ndarray) -> float | None:
        """
        Feed one 10 Hz IMU sample (6-vector).
        Returns predicted speed (m/s) once buffer is full, else None.
        """
        self._buf.append(imu_vec.copy())
        if len(self._buf) > self._win:
            self._buf.pop(0)
        if len(self._buf) < self._win:
            return None
        import torch
        X = np.array(self._buf, dtype=np.float32)[np.newaxis]   # (1, W, 6)
        X_n = torch.tensor(self._norm.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return float(self._net(X_n).item())

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, W, 6). Returns (N,) speed in m/s."""
        import torch
        X_n = torch.tensor(self._norm.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return self._net(X_n).numpy()
