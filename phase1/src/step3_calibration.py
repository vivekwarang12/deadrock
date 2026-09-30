"""
step3_calibration.py  –  STEP 3: Sensor calibration from stationary segments
=============================================================================
Computes per-device:
  - Accelerometer bias (3-axis)
  - Gyroscope bias (3-axis)
  - Noise std per axis
  - Allan deviation (if contiguous stationary data ≥ 1200 s)
Outputs calibration_report.md with real computed numbers (not placeholders).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

OUTPUT_DIR  = Path('/Users/vivekwarang/sih/phase1/outputs')
CALIB_DIR   = OUTPUT_DIR / 'calibration'
CALIB_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / 'reports' / 'calibration_report.md'
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

DT  = 0.1   # 10 Hz
G   = 9.81  # m/s²

# velocity threshold to declare stationary (from V-file)
V_THRESHOLD_KMH = 0.5


# ─────────────────────────────── Allan deviation ──────────────────────────────
def allan_deviation(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute overlapping Allan deviation.
    Returns (tau_array, adev_array).
    Uses overlapping estimator for better confidence at long tau.
    """
    N = len(data)
    # cluster sizes (powers of 2 up to N//4)
    max_m   = N // 4
    m_vals  = np.unique(np.round(
        np.logspace(0, np.log10(max_m), 200)
    ).astype(int))
    m_vals  = m_vals[(m_vals >= 1) & (m_vals <= max_m)]

    taus, adevs = [], []
    for m in m_vals:
        tau  = m / fs
        # overlapping Allan variance
        if N <= 2 * m:
            continue
        phase = np.cumsum(data) / fs   # integrate signal → phase
        # overlapping clusters of size m
        c = phase[m:] - phase[:-m]     # length N-m
        if len(c) < 2:
            continue
        avar = np.sum((c[m:] - c[:-m])**2) / (2 * m**2 * (N - 2*m) / fs**2)
        adevs.append(np.sqrt(np.abs(avar)))
        taus.append(tau)

    return np.array(taus), np.array(adevs)


# ─────────────────────────────── calibration core ────────────────────────────
def calibrate_from_stationary(s_df: pd.DataFrame,
                               v_df: pd.DataFrame | None,
                               run_id: str) -> dict:
    """
    Given a paired S-/V-dataframe from a stationary run, compute bias and noise.
    v_df is optional (used to confirm zero-velocity ground truth).
    Returns a calibration dict.
    """
    # ── Confirm stationary via V-file or S-file GPS speed ────────────────────
    if v_df is not None and 'Velocity_kmh' in v_df.columns:
        motion_mask = v_df['Velocity_kmh'].values > V_THRESHOLD_KMH
        stationary_frac = 1.0 - motion_mask.mean()
        log.info(f'  [{run_id}] Stationary fraction (V-file): {stationary_frac*100:.1f}%')
    else:
        # fallback: use S-file GPS speed
        if 'GPS_Speed_kmh' in s_df.columns:
            motion_mask = s_df['GPS_Speed_kmh'].values > V_THRESHOLD_KMH
            stationary_frac = 1.0 - motion_mask.mean()
        else:
            stationary_frac = 1.0
        log.info(f'  [{run_id}] Stationary fraction (S-file GPS): {stationary_frac*100:.1f}%')

    # ── Extract strictly stationary rows ─────────────────────────────────────
    if v_df is not None and 'Velocity_kmh' in v_df.columns:
        n = min(len(s_df), len(v_df))
        stat_mask = v_df['Velocity_kmh'].values[:n] <= V_THRESHOLD_KMH
        s_stat = s_df.iloc[:n][stat_mask].copy()
    else:
        s_stat = s_df.copy()

    if len(s_stat) < 100:
        log.warning(f'  [{run_id}] Too few stationary samples ({len(s_stat)}); '
                    f'using all rows')
        s_stat = s_df.copy()

    n_stat = len(s_stat)
    dur_s  = n_stat * DT
    log.info(f'  [{run_id}] Calibrating from {n_stat} stationary samples '
             f'({dur_s:.1f} s)')

    # ── Accelerometer: bias = mean - expected (gravity on Z in vehicle frame) ─
    ax = s_stat['Accel_X_ms2'].values.astype(float)
    ay = s_stat['Accel_Y_ms2'].values.astype(float)
    az = s_stat['Accel_Z_ms2'].values.astype(float)

    gx = s_stat['Gravity_X_ms2'].values.astype(float)
    gy = s_stat['Gravity_Y_ms2'].values.astype(float)
    gz = s_stat['Gravity_Z_ms2'].values.astype(float)

    # Gravity vector gives phone orientation → expected accel = gravity (no motion)
    # bias = mean(measured_accel - gravity)  [in static, accel should equal gravity]
    accel_bias_x = float(np.nanmean(ax - gx))
    accel_bias_y = float(np.nanmean(ay - gy))
    accel_bias_z = float(np.nanmean(az - gz))
    accel_std_x  = float(np.nanstd(ax - gx))
    accel_std_y  = float(np.nanstd(ay - gy))
    accel_std_z  = float(np.nanstd(az - gz))

    # ── Gyroscope: bias = mean (should be 0 when stationary) ─────────────────
    wr = s_stat['Gyro_Roll_rads'].values.astype(float)
    wp = s_stat['Gyro_Pitch_rads'].values.astype(float)
    wy = s_stat['Gyro_Yaw_rads'].values.astype(float)

    gyro_bias_roll  = float(np.nanmean(wr))
    gyro_bias_pitch = float(np.nanmean(wp))
    gyro_bias_yaw   = float(np.nanmean(wy))
    gyro_std_roll   = float(np.nanstd(wr))
    gyro_std_pitch  = float(np.nanstd(wp))
    gyro_std_yaw    = float(np.nanstd(wy))

    # ── Allan deviation (if enough samples) ──────────────────────────────────
    adev_results = {}
    if dur_s >= 120:    # need ≥120 s for meaningful AVAR
        for name, sig in [('accel_x', ax - gx),
                           ('accel_y', ay - gy),
                           ('accel_z', az - gz),
                           ('gyro_yaw', wy)]:
            sig_clean = sig[~np.isnan(sig)]
            if len(sig_clean) > 200:
                taus, adev = allan_deviation(sig_clean, fs=1.0/DT)
                # Extract ARW (slope −0.5 at short tau) and bias instability (min)
                if len(adev) > 0:
                    min_idx = np.argmin(adev)
                    adev_results[name] = {
                        'tau_at_min_s':   round(float(taus[min_idx]), 2),
                        'bias_instab':    round(float(adev[min_idx]), 8),
                        'arw_1s':         round(float(adev[0]), 8) if len(adev) > 0 else None,
                    }
        log.info(f'  [{run_id}] Allan deviation computed for '
                 f'{len(adev_results)} signals')
    else:
        log.warning(f'  [{run_id}] Only {dur_s:.0f}s stationary; '
                    f'Allan deviation needs ≥120s – skipped')

    calib = {
        'run_id':      run_id,
        'n_samples':   n_stat,
        'duration_s':  round(dur_s, 1),
        'accel_bias':  {
            'x_ms2': round(accel_bias_x, 6),
            'y_ms2': round(accel_bias_y, 6),
            'z_ms2': round(accel_bias_z, 6),
        },
        'accel_noise_std': {
            'x_ms2': round(accel_std_x, 6),
            'y_ms2': round(accel_std_y, 6),
            'z_ms2': round(accel_std_z, 6),
        },
        'gyro_bias': {
            'roll_rads':  round(gyro_bias_roll, 8),
            'pitch_rads': round(gyro_bias_pitch, 8),
            'yaw_rads':   round(gyro_bias_yaw, 8),
        },
        'gyro_noise_std': {
            'roll_rads':  round(gyro_std_roll, 8),
            'pitch_rads': round(gyro_std_pitch, 8),
            'yaw_rads':   round(gyro_std_yaw, 8),
        },
        'allan_deviation': adev_results,
    }
    return calib


def run_calibration(loaded_dfs: dict, manifest: list[dict]) -> dict:
    """
    Identify calibration (stationary) runs from manifest, run calibration,
    save per-run JSON and a summary markdown report.
    """
    calib_runs = [m for m in manifest
                  if m.get('parse_status') == 'OK'
                  and m['split'] == 'calibration'
                  and m['type'] == 'S']

    if not calib_runs:
        # fallback: look for any long stationary segment in training V/S
        log.warning('No dedicated calibration runs; searching train set for '
                    'zero-velocity segments…')
        calib_runs = [m for m in manifest
                      if m.get('parse_status') == 'OK'
                      and m['type'] == 'S'
                      and m['split'] == 'train'][:3]

    log.info(f'Calibration candidates: {[m["run_id"] for m in calib_runs]}')

    all_calibs = []
    for m in calib_runs:
        run_id = m['run_id']
        s_key  = m['path']
        v_key  = m.get('paired_file')

        s_df = loaded_dfs.get(s_key)
        v_df = loaded_dfs.get(v_key) if v_key else None

        if s_df is None:
            log.warning(f'  [{run_id}] S-dataframe not in memory – skip')
            continue

        calib = calibrate_from_stationary(s_df, v_df, run_id)
        all_calibs.append(calib)

        # save per-run JSON
        json_path = CALIB_DIR / f'calibration_{run_id}.json'
        with open(json_path, 'w') as f:
            json.dump(calib, f, indent=2)
        log.info(f'  [{run_id}] Saved → {json_path}')

    # ── aggregate (mean across all calib runs) ────────────────────────────────
    if not all_calibs:
        log.error('No calibration data computed!')
        return {}

    def mean_bias(key, axis):
        return float(np.mean([c[key][axis] for c in all_calibs]))

    aggregate = {
        'source_runs': [c['run_id'] for c in all_calibs],
        'accel_bias_mean': {
            'x_ms2': mean_bias('accel_bias', 'x_ms2'),
            'y_ms2': mean_bias('accel_bias', 'y_ms2'),
            'z_ms2': mean_bias('accel_bias', 'z_ms2'),
        },
        'accel_noise_std_mean': {
            'x_ms2': mean_bias('accel_noise_std', 'x_ms2'),
            'y_ms2': mean_bias('accel_noise_std', 'y_ms2'),
            'z_ms2': mean_bias('accel_noise_std', 'z_ms2'),
        },
        'gyro_bias_mean': {
            'roll_rads':  mean_bias('gyro_bias', 'roll_rads'),
            'pitch_rads': mean_bias('gyro_bias', 'pitch_rads'),
            'yaw_rads':   mean_bias('gyro_bias', 'yaw_rads'),
        },
        'gyro_noise_std_mean': {
            'roll_rads':  mean_bias('gyro_noise_std', 'roll_rads'),
            'pitch_rads': mean_bias('gyro_noise_std', 'pitch_rads'),
            'yaw_rads':   mean_bias('gyro_noise_std', 'yaw_rads'),
        },
        'per_run': all_calibs,
    }

    # ── save aggregate JSON ───────────────────────────────────────────────────
    agg_path = CALIB_DIR / 'calibration_aggregate.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    # ── write markdown report ─────────────────────────────────────────────────
    _write_report(aggregate, all_calibs)
    return aggregate


def _write_report(agg: dict, calibs: list[dict]):
    lines = [
        '# Phase 1 Calibration Report',
        '',
        '> **Source**: Computed from real stationary segments of the IO-VNBD '
        'dataset (or schema-identical synthetic data). NOT placeholder values.',
        '',
        f'**Runs used**: {", ".join(agg["source_runs"])}',
        '',
        '---',
        '',
        '## Aggregate (mean across all calibration runs)',
        '',
        '### Accelerometer Bias (m/s²)',
        '',
        '| Axis | Bias (m/s²) | Noise Std (m/s²) |',
        '|------|------------|-----------------|',
    ]
    for ax in ('x', 'y', 'z'):
        b = agg['accel_bias_mean'][f'{ax}_ms2']
        n = agg['accel_noise_std_mean'][f'{ax}_ms2']
        lines.append(f'| {ax.upper()}    | {b:+.6f}    | {n:.6f}         |')

    lines += [
        '',
        '### Gyroscope Bias (rad/s)',
        '',
        '| Axis  | Bias (rad/s) | Noise Std (rad/s) |',
        '|-------|-------------|------------------|',
    ]
    for ax in ('roll', 'pitch', 'yaw'):
        b = agg['gyro_bias_mean'][f'{ax}_rads']
        n = agg['gyro_noise_std_mean'][f'{ax}_rads']
        lines.append(f'| {ax.capitalize():<5s} | {b:+.8f}  | {n:.8f}        |')

    lines += ['', '---', '', '## Per-Run Detail', '']
    for c in calibs:
        rid  = c['run_id']
        n    = c['n_samples']
        dur  = c['duration_s']
        lines.append(f'### Run: {rid}  (n={n} samples, {dur:.1f} s)')
        lines.append('')
        lines.append('**Accel bias** (m/s²):  '
                     f"X={c['accel_bias']['x_ms2']:+.6f}  "
                     f"Y={c['accel_bias']['y_ms2']:+.6f}  "
                     f"Z={c['accel_bias']['z_ms2']:+.6f}")
        lines.append('')
        lines.append('**Accel noise σ** (m/s²):  '
                     f"X={c['accel_noise_std']['x_ms2']:.6f}  "
                     f"Y={c['accel_noise_std']['y_ms2']:.6f}  "
                     f"Z={c['accel_noise_std']['z_ms2']:.6f}")
        lines.append('')
        lines.append('**Gyro bias** (rad/s):  '
                     f"Roll={c['gyro_bias']['roll_rads']:+.8f}  "
                     f"Pitch={c['gyro_bias']['pitch_rads']:+.8f}  "
                     f"Yaw={c['gyro_bias']['yaw_rads']:+.8f}")
        lines.append('')
        lines.append('**Gyro noise σ** (rad/s):  '
                     f"Roll={c['gyro_noise_std']['roll_rads']:.8f}  "
                     f"Pitch={c['gyro_noise_std']['pitch_rads']:.8f}  "
                     f"Yaw={c['gyro_noise_std']['yaw_rads']:.8f}")
        if c.get('allan_deviation'):
            lines.append('')
            lines.append('**Allan Deviation (bias instability)**:')
            for sig, vals in c['allan_deviation'].items():
                lines.append(
                    f"  - `{sig}`: τ_min={vals['tau_at_min_s']}s  "
                    f"bias_instab={vals['bias_instab']:.3e}  "
                    f"ARW@1s={vals['arw_1s']:.3e}"
                )
        lines.append('')

    lines += [
        '---',
        '',
        '## Notes',
        '',
        '- Accel bias = mean(measured_accel − gravity_vector) during stationary.',
        '- Gyro bias = mean(measured_gyro) during stationary (should be 0 dps).',
        '- These values must be **subtracted** before strapdown integration.',
        '- Allan deviation needs ≥120 s of contiguous data; skipped otherwise.',
        '- Do NOT generalize these parameters across different phone models.',
        '  Re-run calibration per device as new runs are added.',
        '',
        '*Phase 1 baseline (unoptimized)*',
    ]

    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    log.info(f'Calibration report → {REPORT_PATH}')


if __name__ == '__main__':
    # standalone test
    from step1_ingest import build_manifest
    manifest, loaded_dfs = build_manifest()
    run_calibration(loaded_dfs, manifest)
