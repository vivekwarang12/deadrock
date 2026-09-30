"""
dataset_builder.py  –  Phase 2 Step 1 prerequisite
====================================================
Builds windowed IMU→speed tensors from Phase 1 pipeline outputs.
Also runs empirical window-size sweep (W=5,10,15,20,30) to justify choice.

Input features (vehicle-frame aligned from Phase 1 Step 4):
  Accel_veh_X_ms2, Accel_veh_Y_ms2, Accel_veh_Z_ms2,
  Gyro_veh_yaw_rads, Gyro_veh_pitch_rads, Gyro_veh_roll_rads

Label:
  Velocity_kmh (from V-file CAN bus) → converted to m/s
"""

import sys
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'phase1' / 'src'))

log = logging.getLogger(__name__)

PHASE1_SRC  = Path(__file__).parent.parent.parent / 'phase1' / 'src'
PHASE2_OUT  = Path(__file__).parent.parent / 'outputs'
DT          = 0.1

# ── feature columns (must exist in aligned S-df from Phase 1 Step 4) ──────────
IMU_FEATURES = [
    'Accel_veh_X_ms2', 'Accel_veh_Y_ms2', 'Accel_veh_Z_ms2',
    'Gyro_veh_yaw_rads', 'Gyro_veh_pitch_rads', 'Gyro_veh_roll_rads',
]
N_FEATURES   = len(IMU_FEATURES)
LABEL_COL_V  = 'Velocity_kmh'


# ─────────────────────────── window extraction ────────────────────────────────
def extract_windows(s_aligned: pd.DataFrame,
                    v_df: pd.DataFrame,
                    window: int,
                    stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over paired (S-aligned, V) DataFrames.
    Returns:
        X : (N_windows, window, N_FEATURES)  float32
        y : (N_windows,)                     float32  speed in m/s
    """
    n = min(len(s_aligned), len(v_df))
    # Fill NaNs robustly with ffill, bfill, and 0.0 fallback
    feat_df = s_aligned[IMU_FEATURES].iloc[:n].ffill().bfill().fillna(0.0)
    feat_arr = feat_df.values.astype(np.float32)
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    speed_s = v_df[LABEL_COL_V].iloc[:n].ffill().bfill().fillna(0.0) / 3.6
    speed_arr = speed_s.values.astype(np.float32)
    speed_arr = np.nan_to_num(speed_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X, y = [], []
    for start in range(0, n - window, stride):
        end = start + window
        X.append(feat_arr[start:end])
        y.append(float(np.mean(speed_arr[start:end])))

    if not X:
        return np.empty((0, window, N_FEATURES), dtype=np.float32), \
               np.empty(0, dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def extract_windows_yaw(s_aligned: pd.DataFrame,
                         v_df: pd.DataFrame,
                         window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as extract_windows but also returns yaw_rate label (deg/s)."""
    n = min(len(s_aligned), len(v_df))
    feat_df = s_aligned[IMU_FEATURES].iloc[:n].ffill().bfill().fillna(0.0)
    feat_arr = feat_df.values.astype(np.float32)
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

    speed_s = v_df[LABEL_COL_V].iloc[:n].ffill().bfill().fillna(0.0) / 3.6
    speed_arr = speed_s.values.astype(np.float32)
    speed_arr = np.nan_to_num(speed_arr, nan=0.0, posinf=0.0, neginf=0.0)

    yr_s = v_df['YawRate_degs'].iloc[:n].ffill().bfill().fillna(0.0)
    yr_arr = yr_s.values.astype(np.float32)
    yr_arr = np.nan_to_num(yr_arr, nan=0.0, posinf=0.0, neginf=0.0)

    X, y_spd, y_yr = [], [], []
    for start in range(0, n - window):
        end = start + window
        X.append(feat_arr[start:end])
        y_spd.append(float(np.mean(speed_arr[start:end])))
        y_yr.append(float(np.mean(yr_arr[start:end])))
    return (np.array(X, dtype=np.float32),
            np.array(y_spd, dtype=np.float32),
            np.array(y_yr, dtype=np.float32))


# ──────────────────────── scenario label builder ──────────────────────────────
SCENARIO_TO_CLASS = {
    'stationary': 'idle',
    'hard_brake':  'hard_brake',
    'turn':        'turn',
    'motorway':    'normal',
    'urban':       'normal',
    'pothole':     'pothole_bump',
    'bump':        'pothole_bump',
}
CLASS_TO_IDX = {'normal': 0, 'hard_brake': 1, 'turn': 2,
                 'idle': 3, 'pothole_bump': 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def infer_window_class(v_win: pd.DataFrame, scenario: str) -> int:
    """
    Heuristically assign a motion class to a V-file window.
    Uses CAN-bus signals as proxy labels (weak supervision).
    """
    spd   = v_win['Velocity_kmh'].mean() if 'Velocity_kmh' in v_win else 0
    yr    = abs(v_win['YawRate_degs'].mean()) if 'YawRate_degs' in v_win else 0
    la    = abs(v_win['LatAccel_g'].mean()) if 'LatAccel_g' in v_win else 0
    longa = v_win['LongAccel_g'].mean() if 'LongAccel_g' in v_win else 0

    if spd < 1.0:
        return CLASS_TO_IDX['idle']
    if longa < -0.3:           # deceleration > 0.3g
        return CLASS_TO_IDX['hard_brake']
    if yr > 15.0 or la > 0.25: # tight turn
        return CLASS_TO_IDX['turn']
    # pothole: high-frequency vertical accel spike – approximated by scenario tag
    if scenario in ('pothole', 'bump'):
        return CLASS_TO_IDX['pothole_bump']
    return CLASS_TO_IDX['normal']


def extract_classifier_windows(s_aligned: pd.DataFrame,
                                v_df: pd.DataFrame,
                                scenario: str,
                                window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (features, class_label) pairs for the artifact classifier.
    Features are hand-crafted statistical descriptors per window.
    """
    n = min(len(s_aligned), len(v_df))
    feat_arr = s_aligned[IMU_FEATURES].values[:n].astype(np.float32)

    X_feats, y_cls = [], []
    for start in range(0, n - window, window):   # non-overlapping for class balance
        end = start + window
        win_imu = feat_arr[start:end]
        win_v   = v_df.iloc[start:end]

        # Statistical features per IMU channel
        feats = []
        for ch_idx in range(N_FEATURES):
            ch = win_imu[:, ch_idx]
            ch = ch[~np.isnan(ch)]
            if len(ch) == 0:
                feats.extend([0.0]*7)
                continue
            feats.append(float(np.mean(ch)))
            feats.append(float(np.std(ch)))
            feats.append(float(np.min(ch)))
            feats.append(float(np.max(ch)))
            feats.append(float(np.max(ch) - np.min(ch)))    # peak-to-peak
            # zero-crossing rate
            zc = float(np.sum(np.diff(np.sign(ch - np.mean(ch))) != 0)) / max(len(ch)-1, 1)
            feats.append(zc)
            # RMS
            feats.append(float(np.sqrt(np.mean(ch**2))))

        cls = infer_window_class(win_v, scenario)
        X_feats.append(feats)
        y_cls.append(cls)

    if not X_feats:
        return np.empty((0, N_FEATURES * 7)), np.empty(0, dtype=int)
    return np.array(X_feats, dtype=np.float32), np.array(y_cls, dtype=np.int32)


# ────────────────────────── full dataset assembly ────────────────────────────
def build_all_splits(loaded_dfs: dict, manifest: list[dict],
                     alignment_results: dict,
                     window: int = 20) -> dict:
    """
    Build train/val/test arrays for velocity model and classifier.
    Returns dict with keys: train_X, train_y, val_X, val_y, test_X, test_y,
                            clf_train_X, clf_train_y, clf_test_X, clf_test_y
    """
    splits_vel   = {'train': [], 'val': [], 'test': []}
    splits_label = {'train': [], 'val': [], 'test': []}
    splits_clf_X = {'train': [], 'test': []}
    splits_clf_y = {'train': [], 'test': []}

    sync_pairs = [m for m in manifest
                  if m.get('parse_status') == 'OK'
                  and m['type'] == 'S'
                  and m['sync_status'] == 'Y'
                  and m['split'] in ('train', 'val', 'test')]

    for m in sync_pairs:
        run_id   = m['run_id']
        split    = m['split']
        scenario = m.get('scenario', 'urban')
        v_path   = m.get('paired_file')
        v_df     = loaded_dfs.get(v_path) if v_path else None

        aln = alignment_results.get(run_id)
        if aln and 'aligned_s_df' in aln:
            s_df = aln['aligned_s_df']
        else:
            s_raw = loaded_dfs.get(m['path'])
            if s_raw is None:
                continue
            s_df = s_raw.copy()
            for c_new, c_old in [('Accel_veh_X_ms2','Accel_X_ms2'),
                                   ('Accel_veh_Y_ms2','Accel_Y_ms2'),
                                   ('Accel_veh_Z_ms2','Accel_Z_ms2'),
                                   ('Gyro_veh_yaw_rads','Gyro_Yaw_rads'),
                                   ('Gyro_veh_pitch_rads','Gyro_Pitch_rads'),
                                   ('Gyro_veh_roll_rads','Gyro_Roll_rads')]:
                s_df[c_new] = s_df[c_old]

        if v_df is None:
            log.warning(f'[{run_id}] No V-file – skip dataset build')
            continue

        # velocity windows
        X, y = extract_windows(s_df, v_df, window, stride=5)
        if len(X) == 0:
            continue
        splits_vel[split].append(X)
        splits_label[split].append(y)

        # classifier windows
        if split in ('train', 'test'):
            Xc, yc = extract_classifier_windows(s_df, v_df, scenario, window)
            splits_clf_X[split].append(Xc)
            splits_clf_y[split].append(yc)

        log.info(f'  [{run_id}] {split}: vel_windows={len(X)}  '
                 f'clf_windows={len(splits_clf_X.get(split,[[]])[-1]) if split in splits_clf_X else "—"}')

    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.empty((0,))

    result = {}
    all_tr_X = cat(splits_vel['train'])
    all_tr_y = cat(splits_label['train'])
    all_va_X = cat(splits_vel['val'])
    all_va_y = cat(splits_label['val'])

    if len(all_va_X) == 0 and len(all_tr_X) > 0:
        np.random.seed(42)
        n_val = max(int(len(all_tr_X) * 0.15), 1)
        val_idx = np.random.choice(len(all_tr_X), size=n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(len(all_tr_X)), val_idx)
        result['train_X'] = all_tr_X[train_idx]
        result['train_y'] = all_tr_y[train_idx]
        result['val_X']   = all_tr_X[val_idx]
        result['val_y']   = all_tr_y[val_idx]
    else:
        result['train_X'] = all_tr_X
        result['train_y'] = all_tr_y
        result['val_X']   = all_va_X
        result['val_y']   = all_va_y

    teX = cat(splits_vel['test'])
    teY = cat(splits_label['test'])
    result['test_X'] = teX.reshape(-1, window, N_FEATURES) if len(teX) > 0 else np.empty((0, window, N_FEATURES), np.float32)
    result['test_y'] = teY

    result['clf_train_X'] = cat(splits_clf_X['train'])
    result['clf_train_y'] = cat(splits_clf_y['train']).astype(np.int32)
    result['clf_test_X']  = cat(splits_clf_X['test'])
    result['clf_test_y']  = cat(splits_clf_y['test']).astype(np.int32)
    result['window']      = window

    log.info(f'Dataset built (W={window}): '
             f'train={result["train_X"].shape}  '
             f'val={result["val_X"].shape}  '
             f'test={result["test_X"].shape}  '
             f'clf_train={result["clf_train_X"].shape}')
    return result
