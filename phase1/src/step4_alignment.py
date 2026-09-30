"""
step4_alignment.py  –  STEP 4: Phone-to-Vehicle Frame Alignment
================================================================
Implements:
  1. Gravity-vector pitch/roll estimation (classical, deterministic)
  2. Yaw alignment via correlation of phone-forward accel vs GPS heading
  3. Builds rotation quaternion / matrix: phone frame → vehicle frame
  4. Validates alignment on turning events (roundabouts, sharp turns)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

OUTPUT_DIR   = Path('/Users/vivekwarang/sih/phase1/outputs')
CALIB_DIR    = OUTPUT_DIR / 'calibration'
DT           = 0.1
G            = 9.81
DEG2RAD      = np.pi / 180.0
RAD2DEG      = 180.0 / np.pi


# ─────────────────────────── pitch/roll from gravity ─────────────────────────
def estimate_pitch_roll_from_gravity(gx: np.ndarray, gy: np.ndarray,
                                     gz: np.ndarray,
                                     n_samples: int = 500) -> tuple[float, float]:
    """
    Use the phone's Gravity X/Y/Z columns to estimate static pitch and roll.
    Gravity vector: [gx, gy, gz] in phone frame, magnitude ≈ 9.81 m/s².
    Phone Z-axis upward convention (Androsensor).
    
    Returns (pitch_deg, roll_deg) – phone tilt relative to level ground.
    """
    # Use first n_samples stationary-like samples (or all if fewer)
    n = min(n_samples, len(gx))
    gx_m = np.nanmean(gx[:n])
    gy_m = np.nanmean(gy[:n])
    gz_m = np.nanmean(gz[:n])

    # Normalise
    g_mag = np.sqrt(gx_m**2 + gy_m**2 + gz_m**2)
    if g_mag < 0.1:
        log.warning('Gravity magnitude near zero – cannot estimate tilt')
        return 0.0, 0.0
    gx_n, gy_n, gz_n = gx_m/g_mag, gy_m/g_mag, gz_m/g_mag

    # Pitch around Y-axis: arcsin(gx_n) → tilt of X axis from horizontal
    pitch_rad = np.arcsin(np.clip(gx_n, -1, 1))
    # Roll  around X-axis: arctan2(gy_n, gz_n)
    roll_rad  = np.arctan2(gy_n, gz_n)

    pitch_deg = pitch_rad * RAD2DEG
    roll_deg  = roll_rad  * RAD2DEG
    return float(pitch_deg), float(roll_deg)


# ────────────────────── yaw alignment from straight driving ──────────────────
def estimate_yaw_offset(s_df: pd.DataFrame, v_df: pd.DataFrame,
                        lat_accel_thresh_g: float = 0.05,
                        steering_thresh_deg: float = 5.0,
                        min_speed_kmh: float = 20.0,
                        window_s: float = 10.0) -> float:
    """
    Find segments of sustained straight-line driving by:
      - V-file: lateral accel < threshold AND steering < threshold
      - V-file: speed > min_speed_kmh
    On these segments, correlate phone accel_x direction with GPS heading change
    to estimate yaw rotation between phone-X-axis and vehicle-forward-axis.

    Returns yaw_offset_deg (phone forward axis rotated this many degrees
    from vehicle forward axis, measured in vehicle frame).
    """
    n = min(len(s_df), len(v_df))
    win = int(window_s / DT)

    speed   = v_df['Velocity_kmh'].values[:n]
    lat_acc = v_df['LatAccel_g'].values[:n]
    steer   = v_df['SteeringAngle_deg'].values[:n]

    # GPS heading (from V-file)
    heading = v_df['Heading_deg'].values[:n]

    # phone accelerometer X (forward in phone frame)
    ax = s_df['Accel_X_ms2'].values[:n]
    ay = s_df['Accel_Y_ms2'].values[:n]

    straight_mask = (
        (speed > min_speed_kmh) &
        (np.abs(lat_acc) < lat_accel_thresh_g) &
        (np.abs(steer) < steering_thresh_deg)
    )

    if straight_mask.sum() < win:
        log.warning('Insufficient straight driving segments for yaw alignment – '
                    'using 0°')
        return 0.0

    # On straight segments, vehicle forward acceleration ≈ ax_phone * cos(yaw_off)
    # Estimate by finding angle that best aligns phone lateral accel = 0 (centripetal)
    # Method: during near-zero yaw-rate driving, phone forward axis = vehicle forward
    # → yaw_offset = mean angular difference between phone accel vector and vehicle heading
    
    yaw_estimates = []
    for i in range(win, n - win):
        if not straight_mask[i]:
            continue
        # heading change over window (vehicle forward direction)
        dh = heading[i] - heading[i - win]
        dh = (dh + 180) % 360 - 180   # normalise to [-180, 180]
        if abs(dh) > 2.0:
            continue   # too much heading change – not straight enough

        # phone accel vector in XY plane (horizontal)
        ax_mean = np.nanmean(ax[i - win:i + win])
        ay_mean = np.nanmean(ay[i - win:i + win])
        if np.sqrt(ax_mean**2 + ay_mean**2) < 0.05:
            continue   # near zero accel – no direction info

        phone_forward_angle = np.arctan2(ay_mean, ax_mean) * RAD2DEG
        veh_heading = heading[i]
        yaw_off = veh_heading - phone_forward_angle
        yaw_off = (yaw_off + 180) % 360 - 180
        yaw_estimates.append(yaw_off)

    if not yaw_estimates:
        log.warning('No valid straight windows – using 0° yaw offset')
        return 0.0

    # Circular mean of yaw estimates
    angles_rad = np.array(yaw_estimates) * DEG2RAD
    yaw_off_deg = float(np.arctan2(np.mean(np.sin(angles_rad)),
                                   np.mean(np.cos(angles_rad))) * RAD2DEG)
    log.info(f'  Yaw offset estimated from {len(yaw_estimates)} windows: '
             f'{yaw_off_deg:.2f}°  (std={np.std(yaw_estimates):.2f}°)')
    return yaw_off_deg


# ────────────────────── build rotation matrix ─────────────────────────────────
def build_rotation_matrix(pitch_deg: float, roll_deg: float,
                           yaw_deg: float) -> np.ndarray:
    """
    Return 3×3 rotation matrix R such that:
        accel_vehicle = R @ accel_phone
    Convention: ZYX Euler (yaw → pitch → roll applied in that order).
    """
    r = Rotation.from_euler('ZYX',
                             [yaw_deg, pitch_deg, roll_deg],
                             degrees=True)
    return r.as_matrix()


# ────────────────────── apply alignment to S-dataframe ───────────────────────
def apply_alignment(s_df: pd.DataFrame,
                    R: np.ndarray,
                    calib: dict | None = None) -> pd.DataFrame:
    """
    Apply R (phone→vehicle rotation) and subtract calibrated biases.
    Returns a copy of s_df with new columns:
        Accel_veh_X_ms2, Accel_veh_Y_ms2, Accel_veh_Z_ms2   (vehicle frame, bias-corrected)
        Gyro_veh_roll_rads, Gyro_veh_pitch_rads, Gyro_veh_yaw_rads
    """
    df = s_df.copy()

    ax = df['Accel_X_ms2'].values.astype(float)
    ay = df['Accel_Y_ms2'].values.astype(float)
    az = df['Accel_Z_ms2'].values.astype(float)

    wr = df['Gyro_Roll_rads'].values.astype(float)
    wp = df['Gyro_Pitch_rads'].values.astype(float)
    wy = df['Gyro_Yaw_rads'].values.astype(float)

    # subtract calibrated biases
    if calib:
        ab = calib.get('accel_bias', {})
        gb = calib.get('gyro_bias', {})
        ax -= ab.get('x_ms2', 0.0)
        ay -= ab.get('y_ms2', 0.0)
        az -= ab.get('z_ms2', 0.0)
        wr -= gb.get('roll_rads', 0.0)
        wp -= gb.get('pitch_rads', 0.0)
        wy -= gb.get('yaw_rads', 0.0)

    accel_phone = np.column_stack([ax, ay, az])
    gyro_phone  = np.column_stack([wr, wp, wy])

    accel_veh = (R @ accel_phone.T).T
    gyro_veh  = (R @ gyro_phone.T).T

    df['Accel_veh_X_ms2']     = accel_veh[:, 0]
    df['Accel_veh_Y_ms2']     = accel_veh[:, 1]
    df['Accel_veh_Z_ms2']     = accel_veh[:, 2]
    df['Gyro_veh_roll_rads']  = gyro_veh[:, 0]
    df['Gyro_veh_pitch_rads'] = gyro_veh[:, 1]
    df['Gyro_veh_yaw_rads']   = gyro_veh[:, 2]
    return df


# ────────────────────── turning-event validation ──────────────────────────────
def validate_on_turning_events(s_aligned: pd.DataFrame,
                                v_df: pd.DataFrame,
                                run_id: str,
                                lat_g_thresh: float = 0.1) -> dict:
    """
    Validate alignment by checking that transformed lateral acceleration
    (Accel_veh_Y_ms2) correlates with known turning events from the V-file
    (lateral accel / yaw rate).

    Returns validation metrics dict.
    """
    n = min(len(s_aligned), len(v_df))

    lat_veh_gt = v_df['LatAccel_g'].values[:n] * 9.81  # g → m/s²
    lat_phone_raw   = s_aligned['Accel_Y_ms2'].values[:n]      # raw (un-rotated Y)
    lat_phone_veh   = s_aligned['Accel_veh_Y_ms2'].values[:n]  # rotated to veh frame

    # identify turning events: V-file lateral accel > threshold
    turn_mask = np.abs(lat_veh_gt) > (lat_g_thresh * 9.81)
    n_turns   = turn_mask.sum()

    if n_turns < 10:
        log.warning(f'  [{run_id}] Only {n_turns} turning samples – '
                    f'validation has low confidence')

    def pearson_r(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 10:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    corr_raw    = pearson_r(lat_veh_gt[turn_mask], lat_phone_raw[turn_mask])
    corr_aligned = pearson_r(lat_veh_gt[turn_mask], lat_phone_veh[turn_mask])

    metrics = {
        'run_id':              run_id,
        'n_turning_samples':   int(n_turns),
        'pearson_r_raw':       round(corr_raw,    4),
        'pearson_r_aligned':   round(corr_aligned, 4),
        'improvement':         round(abs(corr_aligned) - abs(corr_raw), 4),
        'validation_passed':   abs(corr_aligned) > 0.6,
    }
    log.info(f'  [{run_id}] Alignment validation: '
             f'r_raw={corr_raw:.3f} → r_aligned={corr_aligned:.3f}  '
             f'(improvement={metrics["improvement"]:+.3f})  '
             f'{"✓ PASS" if metrics["validation_passed"] else "✗ FAIL"}')
    return metrics


# ────────────────────── main entry point ─────────────────────────────────────
def run_alignment(loaded_dfs: dict, manifest: list[dict],
                  calib_aggregate: dict | None = None) -> dict:
    """
    Run alignment on all synchronized train + test pairs.
    Returns dict: run_id → {'R': array, 'pitch_deg', 'roll_deg', 'yaw_deg',
                             'aligned_s_df': DataFrame}
    """
    sync_pairs = [m for m in manifest
                  if m.get('parse_status') == 'OK'
                  and m['type'] == 'S'
                  and m['sync_status'] == 'Y'
                  and m['split'] in ('train', 'test', 'val')]

    results   = {}
    val_table = []

    for m in sync_pairs:
        run_id  = m['run_id']
        s_path  = m['path']
        v_path  = m.get('paired_file')

        s_df = loaded_dfs.get(s_path)
        v_df = loaded_dfs.get(v_path) if v_path else None

        if s_df is None:
            log.warning(f'  [{run_id}] S-df not in memory – skip alignment')
            continue

        log.info(f'[{run_id}] Estimating alignment…')

        # ── Step 1: pitch/roll from gravity ──────────────────────────────────
        gx = s_df['Gravity_X_ms2'].values
        gy = s_df['Gravity_Y_ms2'].values
        gz = s_df['Gravity_Z_ms2'].values
        pitch_deg, roll_deg = estimate_pitch_roll_from_gravity(gx, gy, gz)
        log.info(f'  [{run_id}] Pitch={pitch_deg:.2f}°  Roll={roll_deg:.2f}°')

        # ── Step 2: yaw from straight driving ─────────────────────────────────
        if v_df is not None:
            yaw_deg = estimate_yaw_offset(s_df, v_df)
        else:
            yaw_deg = 0.0
            log.warning(f'  [{run_id}] No V-file – yaw offset = 0°')

        # ── Step 3: build rotation matrix ─────────────────────────────────────
        R = build_rotation_matrix(pitch_deg, roll_deg, yaw_deg)

        # ── Step 4: apply ─────────────────────────────────────────────────────
        # use calibration biases if available
        calib = calib_aggregate or {}
        if calib and 'accel_bias_mean' in calib:
            calib_for_apply = {
                'accel_bias': calib['accel_bias_mean'],
                'gyro_bias':  calib['gyro_bias_mean'],
            }
        else:
            calib_for_apply = None

        s_aligned = apply_alignment(s_df, R, calib_for_apply)

        # ── Step 5: validate on turning events ────────────────────────────────
        if v_df is not None:
            val = validate_on_turning_events(s_aligned, v_df, run_id)
            val_table.append(val)
        else:
            val = None

        results[run_id] = {
            'pitch_deg':  pitch_deg,
            'roll_deg':   roll_deg,
            'yaw_deg':    yaw_deg,
            'R_matrix':   R.tolist(),
            'aligned_s_df': s_aligned,
            'validation': val,
        }

    # ── save alignment params ─────────────────────────────────────────────────
    params_to_save = {
        k: {kk: vv for kk, vv in v.items() if kk != 'aligned_s_df'}
        for k, v in results.items()
    }
    aln_path = CALIB_DIR / 'alignment_params.json'
    with open(aln_path, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    log.info(f'Alignment params saved → {aln_path}')

    # ── print validation summary ───────────────────────────────────────────────
    if val_table:
        passes = sum(1 for v in val_table if v['validation_passed'])
        log.info(f'\nAlignment validation: {passes}/{len(val_table)} runs passed '
                 f'(|r_aligned| > 0.6)')
        for v in val_table:
            status = '✓' if v['validation_passed'] else '✗'
            log.info(f'  {status} [{v["run_id"]}]  '
                     f'r_raw={v["pearson_r_raw"]:.3f} → '
                     f'r_aligned={v["pearson_r_aligned"]:.3f}  '
                     f'n_turns={v["n_turning_samples"]}')

    return results


if __name__ == '__main__':
    from step1_ingest import build_manifest
    from step3_calibration import run_calibration
    manifest, loaded_dfs = build_manifest()
    calib = run_calibration(loaded_dfs, manifest)
    run_alignment(loaded_dfs, manifest, calib)
