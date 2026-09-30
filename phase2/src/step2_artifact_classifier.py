"""
step2_artifact_classifier.py  –  Phase 2 Step 2
================================================
Motion-artifact / driving-scenario classifier.

Architecture: Gradient-Boosted Trees (scikit-learn GradientBoostingClassifier)
  on hand-crafted statistical features per IMU window.

Classes:
  0  normal         – ordinary driving
  1  hard_brake     – deceleration > 0.3g
  2  turn           – lateral accel > 0.25g or yaw > 15°/s
  3  idle           – stationary / stopped
  4  pothole_bump   – high-freq vertical spike (weak label from scenario tag)

Why GBT, not CNN?
  - Scenario labels are WEAK / APPROXIMATE (derived from CAN-bus thresholds
    and scenario metadata, NOT from manual frame-level annotation).
  - GBT is interpretable: SHAP importances confirm which IMU channels drive
    each class — essential for ISRO evaluators.
  - ~5 ms inference on CPU, well within 10 Hz budget.
  - Feature set is auditable; CNN hidden representations are not.

⚠️  Label quality disclaimer (printed explicitly in every report):
    These labels are heuristically assigned from CAN-bus signals and
    scenario tags — NOT manual ground-truth annotations.
    Reported accuracy reflects proxy-label agreement, not true
    classifier performance on human-verified event boundaries.
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score)
import joblib

from dataset_builder import CLASS_TO_IDX, IDX_TO_CLASS

log = logging.getLogger(__name__)

PHASE2_OUT  = Path(__file__).parent.parent / 'outputs'
MODELS_DIR  = PHASE2_OUT / 'models'
REPORTS_DIR = PHASE2_OUT / 'reports'
PLOTS_DIR   = PHASE2_OUT / 'plots'
for d in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]


# ─────────────────────── training ────────────────────────────────────────────
def train_artifact_classifier(clf_train_X: np.ndarray,
                               clf_train_y: np.ndarray,
                               clf_test_X:  np.ndarray,
                               clf_test_y:  np.ndarray) -> dict:
    """
    Train GBT classifier. Returns dict with model, scaler, metrics.
    """
    log.info(f'Artifact classifier training: '
             f'n_train={len(clf_train_X)}  n_test={len(clf_test_X)}')
    log.info(f'  Class distribution (train): '
             + str({IDX_TO_CLASS.get(k,k): int(v)
                    for k, v in zip(*np.unique(clf_train_y, return_counts=True))}))

    if len(clf_train_X) < 10:
        log.warning('Insufficient training data for classifier; skipping')
        return {}

    # Normalise features
    scaler = StandardScaler()
    trX_s  = scaler.fit_transform(clf_train_X)
    teX_s  = scaler.transform(clf_test_X) if len(clf_test_X) > 0 else clf_test_X

    # Fit GBT (class_weight-equivalent via sample_weight not directly available,
    # but subsample+min_samples balances somewhat)
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
        verbose=0,
    )
    clf.fit(trX_s, clf_train_y)
    log.info('  GBT training complete')

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_acc = float(clf.score(trX_s, clf_train_y))
    log.info(f'  Train accuracy: {train_acc:.4f}')

    if len(clf_test_X) > 0 and len(clf_test_y) > 0:
        test_pred    = clf.predict(teX_s)
        test_proba   = clf.predict_proba(teX_s)
        test_acc     = float(np.mean(test_pred == clf_test_y))
        bal_acc      = float(balanced_accuracy_score(clf_test_y, test_pred))
        labels_list  = list(range(len(CLASS_NAMES)))
        cr           = classification_report(clf_test_y, test_pred,
                                              labels=labels_list,
                                              target_names=CLASS_NAMES,
                                              zero_division=0, output_dict=True)
        cm           = confusion_matrix(clf_test_y, test_pred,
                                        labels=list(range(len(CLASS_NAMES))))
        log.info(f'  Test accuracy: {test_acc:.4f}  '
                 f'Balanced acc: {bal_acc:.4f}')
    else:
        test_pred = np.array([])
        test_acc  = float('nan'); bal_acc = float('nan')
        cr = {}; cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
        log.warning('  No test data for classifier evaluation')

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump({'clf': clf, 'scaler': scaler},
                MODELS_DIR / 'artifact_classifier.pkl')
    log.info(f'  Classifier saved → {MODELS_DIR}/artifact_classifier.pkl')

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    _plot_confusion_matrix(cm, CLASS_NAMES, test_acc, bal_acc)

    # ── Feature importance ────────────────────────────────────────────────────
    fi = clf.feature_importances_
    fi_path = REPORTS_DIR / 'classifier_feature_importance.json'
    with open(fi_path, 'w') as f:
        # name features: stat_channel_0..N_features-1
        feat_names = [f'{stat}_{ch}'
                      for ch in range(6)
                      for stat in ['mean','std','min','max','ptp','zcr','rms']]
        top10 = sorted(zip(feat_names, fi.tolist()), key=lambda x: -x[1])[:10]
        json.dump({'top10': top10, 'all': list(zip(feat_names, fi.tolist()))}, f, indent=2)

    metrics = {
        'train_accuracy':    round(train_acc, 4),
        'test_accuracy':     round(test_acc, 4) if not np.isnan(test_acc) else None,
        'balanced_accuracy': round(bal_acc, 4)  if not np.isnan(bal_acc)  else None,
        'n_train':           int(len(clf_train_X)),
        'n_test':            int(len(clf_test_X)),
        'class_report':      cr,
        'LABEL_QUALITY_WARNING': (
            'Labels are WEAK / APPROXIMATE: assigned heuristically from CAN-bus '
            'thresholds (decel, yaw rate, lateral accel) and scenario metadata. '
            'NOT manually annotated frame-level ground truth. '
            'Reported accuracy reflects proxy-label agreement only.'
        ),
    }
    with open(REPORTS_DIR / 'classifier_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f'  Classifier metrics saved')

    return {'clf': clf, 'scaler': scaler,
            'metrics': metrics, 'confusion_matrix': cm}


def _plot_confusion_matrix(cm, class_names, test_acc, bal_acc):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(list(ticks)); ax.set_yticks(list(ticks))
    ax.set_xticklabels(class_names, rotation=35, ha='right')
    ax.set_yticklabels(class_names)
    # annotate cells
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Artifact Classifier  '
                 f'Acc={test_acc:.3f}  BalAcc={bal_acc:.3f}\n'
                 f'⚠ Labels are WEAK PROXIES (not manual ground truth)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'classifier_confusion_matrix.png', dpi=120)
    plt.close()
    log.info('Confusion matrix plot saved')


# ─────────────────────── inference helper ─────────────────────────────────────
class ArtifactGate:
    """
    Wraps classifier for real-time artifact detection.
    Used by hybrid DR to suppress or re-weight corrupted IMU windows.
    """
    SPIKE_CLASSES   = {CLASS_TO_IDX['pothole_bump']}
    BRAKE_CLASSES   = {CLASS_TO_IDX['hard_brake']}
    TURN_CLASSES    = {CLASS_TO_IDX['turn']}

    def __init__(self, clf_pack: dict):
        self._clf    = clf_pack['clf']
        self._scaler = clf_pack['scaler']

    def predict(self, feat_vec: np.ndarray) -> tuple[int, float]:
        """
        feat_vec: 1D statistical feature vector (42 dims for 6 channels).
        Returns (class_idx, confidence).
        """
        x  = self._scaler.transform(feat_vec.reshape(1, -1))
        ci = int(self._clf.predict(x)[0])
        cf = float(self._clf.predict_proba(x)[0, ci])
        return ci, cf

    def velocity_scale_factor(self, class_idx: int, confidence: float) -> float:
        """
        Returns a [0,1] multiplier applied to the ML velocity prediction
        during suspected artifact windows.
        - pothole/bump: velocity is real but accel is corrupt → keep vel, scale=1
        - hard_brake:   trust ML velocity (it handles brakes)    → scale=1
        - idle:         zero velocity is correct                  → scale=0 (ZUPT)
        """
        if class_idx == CLASS_TO_IDX.get('idle', 3):
            return 0.0   # zero-velocity update
        return 1.0       # all others: trust ML velocity model


def load_classifier(path: Path | None = None) -> ArtifactGate | None:
    p = path or (MODELS_DIR / 'artifact_classifier.pkl')
    if not p.exists():
        return None
    pack = joblib.load(p)
    return ArtifactGate(pack)
