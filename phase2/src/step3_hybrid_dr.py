"""
step3_hybrid_dr.py  –  Phase 2 Step 3
======================================
Hybrid dead-reckoning: ML velocity + gyro heading + artifact gating.

Replaces Phase 1's double-integrated accel velocity with:
  velocity(t) = VelocityNet(IMU_window[t-W:t])   * artifact_gate(t)

Heading still comes from gyro integration (corrected by alignment R from Phase 1).
Position from velocity × cos/sin(heading) × dt — same as Phase 1 mechanization.

Key improvement over Phase 1:
  - VelocityNet predicts speed from IMU patterns (no double integration)
  - Artifact gate applies ZUPT when classifier predicts 'idle'
  - This eliminates the major source of Phase 1's velocity drift

Re-evaluates on IDENTICAL outage windows from Phase 1 for apples-to-apples
comparison. Reads Phase 1's evaluation results JSON to get exact window indices.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

PHASE1_OUT  = Path(__file__).parent.parent.parent / 'phase1' / 'outputs'
PHASE2_OUT  = Path(__file__).parent.parent / 'outputs'
PLOTS_DIR   = PHASE2_OUT / 'plots'
REPORTS_DIR = PHASE2_OUT / 'reports'

DT        = 0.1
R_EARTH   = 6_371_000.0
DEG2RAD   = np.pi / 180.0
IMU_FEATURES = [
    'Accel_veh_X_ms2', 'Accel_veh_Y_ms2', 'Accel_veh_Z_ms2',
    'Gyro_veh_yaw_rads', 'Gyro_veh_pitch_rads', 'Gyro_veh_roll_rads',
]


# ─────────────────────── geometry helpers ─────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    dlat = (lat2 - lat1) * DEG2RAD
    dlon = (lon2 - lon1) * DEG2RAD
    a    = (np.sin(dlat/2)**2 +
            np.cos(lat1*DEG2RAD) * np.cos(lat2*DEG2RAD) * np.sin(dlon/2)**2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cumulative_distance_m(lat, lon) -> np.ndarray:
    dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dist[i] = dist[i-1] + haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    return dist


def lat_lon_to_local_m(lat, lon, lat0, lon0):
    E = (lon - lon0) * DEG2RAD * R_EARTH * np.cos(lat0 * DEG2RAD)
    N = (lat - lat0) * DEG2RAD * R_EARTH
    return E, N


# ─────────────────────── build classifier features for one timestep ───────────
def _imu_window_features(imu_win: np.ndarray) -> np.ndarray:
    """42-dim statistical feature vector for one window."""
    feats = []
    for ch in range(imu_win.shape[1]):
        c = imu_win[:, ch]
        c = c[~np.isnan(c)]
        if len(c) == 0:
            feats.extend([0.0]*7); continue
        feats.append(float(np.mean(c)))
        feats.append(float(np.std(c)))
        feats.append(float(np.min(c)))
        feats.append(float(np.max(c)))
        feats.append(float(np.max(c) - np.min(c)))
        zc = float(np.sum(np.diff(np.sign(c - np.mean(c))) != 0)) / max(len(c)-1, 1)
        feats.append(zc)
        feats.append(float(np.sqrt(np.mean(c**2))))
    return np.array(feats, dtype=np.float32)


# ─────────────────────── core hybrid DR ───────────────────────────────────────
def hybrid_dead_reckoning(s_aligned: pd.DataFrame,
                           v_predictor,        # VelocityPredictor
                           artifact_gate,       # ArtifactGate | None
                           init_lat_deg: float,
                           init_lon_deg: float,
                           init_heading_deg: float = 0.0,
                           window: int = 20) -> dict:
    """
    Full-run hybrid DR using ML velocity + gyro heading.
    Returns same-format dict as Phase 1 strapdown_dead_reckoning.
    """
    import torch
    n = len(s_aligned)
    feat_arr = s_aligned[IMU_FEATURES].values.astype(np.float32)
    gyro_yaw = s_aligned['Gyro_veh_yaw_rads'].values.astype(np.float32)

    # forward-fill NaNs
    def ffill(arr):
        s = pd.Series(arr); return s.ffill().fillna(0.0).values
    feat_arr  = np.column_stack([ffill(feat_arr[:, i]) for i in range(6)])
    gyro_yaw  = ffill(gyro_yaw)

    lat_rad     = np.zeros(n)
    lon_rad     = np.zeros(n)
    speed       = np.zeros(n)
    heading_rad = np.zeros(n)

    lat_rad[0]     = init_lat_deg * DEG2RAD
    lon_rad[0]     = init_lon_deg * DEG2RAD
    heading_rad[0] = init_heading_deg * DEG2RAD

    for i in range(1, n):
        # 1. heading from gyro
        heading_rad[i] = heading_rad[i-1] + gyro_yaw[i-1] * DT

        # 2. velocity from ML (once buffer is full)
        v_ml = None
        if i >= window:
            X_win = feat_arr[i-window:i][np.newaxis]  # (1, W, 6)
            v_ml  = float(v_predictor.predict_batch(X_win)[0])
            v_ml  = max(0.0, v_ml)   # enforce non-negative

            # artifact gate: ZUPT for idle
            if artifact_gate is not None:
                feat_vec = _imu_window_features(feat_arr[i-window:i])
                cls_idx, conf = artifact_gate.predict(feat_vec)
                scale = artifact_gate.velocity_scale_factor(cls_idx, conf)
                v_ml *= scale

        if v_ml is None:
            v_ml = speed[i-1]   # hold last known speed during warm-up
        speed[i] = v_ml

        # 3. position update
        h_mid = (heading_rad[i-1] + heading_rad[i]) / 2.0
        d     = speed[i-1] * DT
        dlat  = d * np.cos(h_mid) / R_EARTH
        dlon  = d * np.sin(h_mid) / (R_EARTH * np.cos(lat_rad[i-1]))
        lat_rad[i] = lat_rad[i-1] + dlat
        lon_rad[i] = lon_rad[i-1] + dlon

    return {
        'lat_deg':     lat_rad * (180.0/np.pi),
        'lon_deg':     lon_rad * (180.0/np.pi),
        'speed_ms':    speed,
        'heading_deg': (heading_rad * (180.0/np.pi)) % 360.0,
        'time_s':      np.arange(n) * DT,
    }


# ─────────────────────── segment evaluator ────────────────────────────────────
def evaluate_segment_hybrid(s_aligned: pd.DataFrame,
                              v_df: pd.DataFrame,
                              v_predictor,
                              artifact_gate,
                              start_idx: int,
                              end_idx: int,
                              window: int = 20) -> dict:
    """Evaluate hybrid DR on one masked window (same interface as Phase 1)."""
    seg_s = s_aligned.iloc[start_idx:end_idx].reset_index(drop=True)
    seg_v = v_df.iloc[start_idx:end_idx].reset_index(drop=True)

    init_lat  = float(v_df['Latitude_deg'].iloc[start_idx])
    init_lon  = float(v_df['Longitude_deg'].iloc[start_idx])
    init_head = float(v_df['Heading_deg'].iloc[start_idx])

    dr = hybrid_dead_reckoning(seg_s, v_predictor, artifact_gate,
                                init_lat, init_lon, init_head, window)

    gt = {
        'lat_deg':   seg_v['Latitude_deg'].values.astype(float),
        'lon_deg':   seg_v['Longitude_deg'].values.astype(float),
        'speed_ms':  seg_v['Velocity_kmh'].values.astype(float) / 3.6,
    }
    return {'dr': dr, 'gt': gt, 'start_idx': start_idx, 'end_idx': end_idx}


# ─────────────────────── compute metrics (shared) ─────────────────────────────
def compute_window_metrics(dr: dict, gt: dict) -> dict:
    n = len(dr['lat_deg'])
    gt_lat = gt['lat_deg'][:n]; gt_lon = gt['lon_deg'][:n]
    dr_lat = dr['lat_deg'][:n]; dr_lon = dr['lon_deg'][:n]

    drift_m = haversine_m(dr_lat[-1], dr_lon[-1], gt_lat[-1], gt_lon[-1])
    cum     = cumulative_distance_m(gt_lat, gt_lon)
    true_dist = cum[-1]
    drift_pct = (drift_m / true_dist * 100.0) if true_dist > 0 else 0.0

    errs = np.array([haversine_m(dr_lat[i], dr_lon[i], gt_lat[i], gt_lon[i])
                     for i in range(n)])
    vel_rmse = float(np.sqrt(np.nanmean((dr['speed_ms'][:n] - gt['speed_ms'][:n])**2)))

    return {
        'endpoint_drift_m': round(drift_m, 2),
        'true_dist_m':      round(true_dist, 2),
        'drift_pct':        round(drift_pct, 2),
        'vel_rmse_ms':      round(vel_rmse, 4),
        'track_rmse_m':     round(float(np.sqrt(np.nanmean(errs**2))), 2),
        'max_track_err_m':  round(float(np.nanmax(errs)), 2),
        'window_dur_s':     round(n * DT, 1),
    }


# ─────────────────────── full re-evaluation harness ──────────────────────────
def run_hybrid_evaluation(loaded_dfs: dict, manifest: list[dict],
                           alignment_results: dict,
                           v_predictor,
                           artifact_gate,
                           window: int = 20,
                           phase1_results: list[dict] | None = None) -> list[dict]:
    """
    Re-runs evaluation on IDENTICAL windows as Phase 1 if phase1_results given.
    Otherwise discovers windows fresh (same window-finding logic as Phase 1).
    """
    from step6_evaluation import find_outage_windows, OUTAGE_DISTANCES_M

    test_pairs = [m for m in manifest
                  if m.get('parse_status') == 'OK'
                  and m['type'] == 'S'
                  and m['sync_status'] == 'Y'
                  and m['split'] in ('test',)]
    if not test_pairs:
        test_pairs = [m for m in manifest
                      if m.get('parse_status') == 'OK'
                      and m['type'] == 'S'
                      and m['sync_status'] == 'Y'
                      and m['split'] == 'train'][:4]

    # Build index of Phase 1 windows per run if available
    p1_idx = {}
    if phase1_results:
        for r in phase1_results:
            key = (r['run_id'], r.get('target_dist_m'))
            p1_idx.setdefault(r['run_id'], []).append(r)

    all_results = []
    plot_paths  = []

    for m in test_pairs:
        run_id = m['run_id']
        v_path = m.get('paired_file')
        v_df   = loaded_dfs.get(v_path) if v_path else None
        if v_df is None:
            continue

        aln = alignment_results.get(run_id)
        if aln and 'aligned_s_df' in aln:
            s_aligned = aln['aligned_s_df']
        else:
            s_raw = loaded_dfs.get(m['path'])
            if s_raw is None: continue
            s_aligned = s_raw.copy()
            for c_new, c_old in [('Accel_veh_X_ms2','Accel_X_ms2'),
                                   ('Accel_veh_Y_ms2','Accel_Y_ms2'),
                                   ('Accel_veh_Z_ms2','Accel_Z_ms2'),
                                   ('Gyro_veh_yaw_rads','Gyro_Yaw_rads'),
                                   ('Gyro_veh_pitch_rads','Gyro_Pitch_rads'),
                                   ('Gyro_veh_roll_rads','Gyro_Roll_rads')]:
                s_aligned[c_new] = s_aligned[c_old]

        n_v = min(len(s_aligned), len(v_df))
        s_aligned = s_aligned.iloc[:n_v].reset_index(drop=True)
        v_df_trim = v_df.iloc[:n_v].reset_index(drop=True)

        log.info(f'\n=== Hybrid DR: {run_id} ===')
        for target_m in OUTAGE_DISTANCES_M:
            # use Phase 1 window indices if available (ensures identical comparison)
            if run_id in p1_idx:
                p1_run = [r for r in p1_idx[run_id]
                          if r.get('target_dist_m') == target_m]
                windows = [(r['start_idx'], r['end_idx']) for r in p1_run
                           if r.get('start_idx') is not None]
            else:
                windows = []

            if not windows:
                windows = find_outage_windows(v_df_trim, target_m,
                                               n_windows=4, min_speed_kmh=5.0)

            for w_idx, (si, ei) in enumerate(windows):
                seg = evaluate_segment_hybrid(s_aligned, v_df_trim,
                                              v_predictor, artifact_gate,
                                              si, ei, window)
                metrics = compute_window_metrics(seg['dr'], seg['gt'])
                result  = {'run_id': run_id, 'window_idx': w_idx+1,
                            'target_dist_m': target_m,
                            'start_idx': si, 'end_idx': ei, **metrics}
                all_results.append(result)
                log.info(f'  [{run_id}] w{w_idx+1} {target_m}m: '
                         f'drift={metrics["endpoint_drift_m"]:.1f}m '
                         f'({metrics["drift_pct"]:.1f}%)  '
                         f'vel_rmse={metrics["vel_rmse_ms"]:.4f} m/s')

                if w_idx < 2:
                    pth = _plot_hybrid_trajectory(seg['dr'], seg['gt'],
                                                   metrics, run_id, w_idx, target_m)
                    plot_paths.append(pth)

    return all_results, plot_paths


def _plot_hybrid_trajectory(dr, gt, metrics, run_id, w_idx, target_m):
    n     = len(dr['lat_deg'])
    lat0, lon0 = gt['lat_deg'][0], gt['lon_deg'][0]
    gt_E, gt_N = lat_lon_to_local_m(gt['lat_deg'][:n], gt['lon_deg'][:n], lat0, lon0)
    dr_E, dr_N = lat_lon_to_local_m(dr['lat_deg'][:n], dr['lon_deg'][:n], lat0, lon0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Phase 2 Hybrid DR vs Ground Truth\n'
                 f'Run: {run_id}  Window #{w_idx+1}  Target: {target_m} m',
                 fontweight='bold')
    ax = axes[0]
    ax.plot(gt_E, gt_N, 'g-', lw=2, label='GPS Ground Truth', zorder=3)
    ax.plot(dr_E, dr_N, 'b--', lw=1.5, label='Hybrid DR (ML velocity)', zorder=3)
    ax.scatter([gt_E[0]], [gt_N[0]], c='blue', s=80, zorder=5, label='Start')
    ax.scatter([gt_E[-1]], [gt_N[-1]], c='green', s=80, marker='*', zorder=5)
    ax.scatter([dr_E[-1]], [dr_N[-1]], c='blue', s=80, marker='x', zorder=5)
    ax.annotate('', xy=(dr_E[-1], dr_N[-1]), xytext=(gt_E[-1], gt_N[-1]),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    ax.text(0.02, 0.02,
            f'Drift: {metrics["endpoint_drift_m"]:.1f}m ({metrics["drift_pct"]:.1f}%)\n'
            f'True dist: {metrics["true_dist_m"]:.1f}m',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
    ax.set_title('Trajectory (blue=hybrid, green=GT)')
    ax.legend(fontsize=8); ax.set_aspect('equal', 'datalim'); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    t = np.arange(n) * DT
    ax2.plot(t, gt['speed_ms'][:n]*3.6, 'g-', lw=2, label='GT Speed')
    ax2.plot(t, dr['speed_ms'][:n]*3.6, 'b--', lw=1.5, label='ML Speed')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Speed (km/h)')
    ax2.set_title(f'Speed   Vel RMSE={metrics["vel_rmse_ms"]:.3f} m/s')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = PLOTS_DIR / f'hybrid_dr_{run_id}_w{w_idx+1}_{int(target_m)}m.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    log.info(f'  Plot saved → {fname.name}')
    return fname
