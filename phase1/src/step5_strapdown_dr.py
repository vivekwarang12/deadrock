"""
step5_strapdown_dr.py  –  STEP 5: Classical Strapdown Dead Reckoning
=====================================================================
Implements standard strapdown INS mechanization:
  1. Gyro integration for orientation (yaw only for 2D navigation)
  2. Accel integration (after gravity removal + bias correction) → velocity
  3. Velocity integration → position

This is the NAIVE BASELINE. It is expected to drift badly.
Do NOT artificially tune it to look good. The point is to measure real drift.

Input:  vehicle-frame-aligned S-dataframe (from step4_alignment.py)
Output: (lat_est, lon_est, speed_est) arrays
"""

import numpy as np
import pandas as pd

DT      = 0.1
G       = 9.81
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
R_EARTH = 6_371_000.0   # metres


def strapdown_dead_reckoning(s_aligned: pd.DataFrame,
                              init_lat_deg: float,
                              init_lon_deg: float,
                              init_heading_deg: float = 0.0,
                              init_speed_ms: float = 0.0) -> dict:
    """
    Classical strapdown 2D dead-reckoning on aligned phone IMU data.

    Parameters
    ----------
    s_aligned       : S-file DataFrame with Accel_veh_X/Y/Z_ms2 and
                      Gyro_veh_yaw_rads columns (from step4_alignment).
    init_lat_deg    : Starting latitude (degrees)
    init_lon_deg    : Starting longitude (degrees)
    init_heading_deg: Initial heading (degrees, 0=North, CW)
    init_speed_ms   : Initial speed (m/s)

    Returns
    -------
    dict with arrays:
        lat_deg, lon_deg, speed_ms, heading_deg, time_s
    """
    n = len(s_aligned)

    # ── extract aligned IMU ───────────────────────────────────────────────────
    ax = s_aligned['Accel_veh_X_ms2'].values.astype(float)   # forward
    ay = s_aligned['Accel_veh_Y_ms2'].values.astype(float)   # lateral (left)
    # az not used in 2D

    gyro_yaw = s_aligned['Gyro_veh_yaw_rads'].values.astype(float)

    # fill NaNs with forward interpolation then 0
    def fill_nan(arr):
        df_tmp = pd.Series(arr)
        return df_tmp.ffill().fillna(0.0).values

    ax       = fill_nan(ax)
    ay       = fill_nan(ay)
    gyro_yaw = fill_nan(gyro_yaw)

    # ── state arrays ──────────────────────────────────────────────────────────
    lat_rad     = np.zeros(n)
    lon_rad     = np.zeros(n)
    speed       = np.zeros(n)
    heading_rad = np.zeros(n)

    lat_rad[0]     = init_lat_deg * DEG2RAD
    lon_rad[0]     = init_lon_deg * DEG2RAD
    heading_rad[0] = init_heading_deg * DEG2RAD
    speed[0]       = init_speed_ms

    # ── integration loop ─────────────────────────────────────────────────────
    for i in range(1, n):
        # 1. Heading update (gyro integration)
        heading_rad[i] = heading_rad[i-1] + gyro_yaw[i-1] * DT

        # 2. Velocity update (forward acceleration only; lateral ignored for speed)
        # NOTE: no zupt, no gravity de-rotation beyond what alignment did
        v_prev   = speed[i-1]
        # forward accel in vehicle frame = ax (already gravity-free from alignment)
        dv       = ax[i-1] * DT
        speed[i] = max(0.0, v_prev + dv)

        # 3. Position update
        # Use midpoint heading for stability
        h_mid = (heading_rad[i-1] + heading_rad[i]) / 2.0
        d     = speed[i-1] * DT      # distance this step
        dlat  = d * np.cos(h_mid) / R_EARTH
        dlon  = d * np.sin(h_mid) / (R_EARTH * np.cos(lat_rad[i-1]))
        lat_rad[i] = lat_rad[i-1] + dlat
        lon_rad[i] = lon_rad[i-1] + dlon

    return {
        'lat_deg':     lat_rad * RAD2DEG,
        'lon_deg':     lon_rad * RAD2DEG,
        'speed_ms':    speed,
        'heading_deg': (heading_rad * RAD2DEG) % 360.0,
        'time_s':      np.arange(n) * DT,
    }


def run_on_segment(s_aligned: pd.DataFrame,
                   v_df: pd.DataFrame,
                   start_idx: int,
                   end_idx: int) -> dict:
    """
    Run strapdown DR on a masked segment [start_idx, end_idx).
    Initialise from GPS/V-file ground truth at start_idx.
    Returns DR estimates + ground truth for that window.
    """
    seg_s = s_aligned.iloc[start_idx:end_idx].reset_index(drop=True)
    seg_v = v_df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Initialise from V-file GPS at start
    init_lat  = float(v_df['Latitude_deg'].iloc[start_idx])
    init_lon  = float(v_df['Longitude_deg'].iloc[start_idx])
    init_head = float(v_df['Heading_deg'].iloc[start_idx])
    init_spd  = float(v_df['Velocity_kmh'].iloc[start_idx]) / 3.6

    dr = strapdown_dead_reckoning(seg_s, init_lat, init_lon,
                                   init_head, init_spd)

    # Ground truth from V-file
    gt = {
        'lat_deg':   seg_v['Latitude_deg'].values.astype(float),
        'lon_deg':   seg_v['Longitude_deg'].values.astype(float),
        'speed_ms':  seg_v['Velocity_kmh'].values.astype(float) / 3.6,
        'heading_deg': seg_v['Heading_deg'].values.astype(float),
    }

    return {'dr': dr, 'gt': gt,
            'start_idx': start_idx, 'end_idx': end_idx}
