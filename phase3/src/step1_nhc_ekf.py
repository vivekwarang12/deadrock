"""
step1_nhc_ekf.py  –  Phase 3 Step 1
====================================
Extended Error-State Kalman Filter with Non-Holonomic Constraints (NHC).

Land vehicle dynamics dictate that, under non-skidding conditions:
  v_lateral_body  ≈ 0
  v_vertical_body ≈ 0

In the 2D navigation frame (East, North) with heading angle psi (rad from North, clockwise):
  v_forward =  v_E * sin(psi) + v_N * cos(psi)
  v_lateral =  v_E * cos(psi) - v_N * sin(psi) ≈ 0

When forward velocity exceeds a threshold (e.g. 0.5 m/s), NHC provides a continuous
pseudo-measurement update to the EKF:
  z_nhc = 0 = [cos(psi), -sin(psi)] * [v_E, v_N]^T + v_noise
  H_nhc = [0, 0, cos(psi), -sin(psi), -v_E*sin(psi) - v_N*cos(psi)]

This dramatically suppresses lateral drift during extended GNSS outages.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DT = 0.1
R_EARTH = 6_371_000.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    dlat = (lat2 - lat1) * DEG2RAD
    dlon = (lon2 - lon1) * DEG2RAD
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat1 * DEG2RAD) * np.cos(lat2 * DEG2RAD) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cumulative_distance_m(lat, lon) -> np.ndarray:
    dist = np.zeros(len(lat))
    for i in range(1, len(lat)):
        dist[i] = dist[i-1] + haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    return dist


def lat_lon_to_local_m(lat, lon, lat0, lon0):
    E = (lon - lon0) * DEG2RAD * R_EARTH * np.cos(lat0 * DEG2RAD)
    N = (lat - lat0) * DEG2RAD * R_EARTH
    return E, N


def local_m_to_lat_lon(E, N, lat0, lon0):
    lat = lat0 + (N / R_EARTH) * RAD2DEG
    lon = lon0 + (E / (R_EARTH * np.cos(lat0 * DEG2RAD))) * RAD2DEG
    return lat, lon


def _imu_window_features(imu_win: np.ndarray) -> np.ndarray:
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


class ESEKFWithNHC:
    """
    Error-State Extended Kalman Filter fusing:
      - Process: ML velocity prediction + gyro yaw integration
      - GNSS Updates: Position (East, North) and velocity when GPS available
      - NHC Updates: Zero lateral velocity pseudo-measurement when moving
      - ZUPT Updates: Zero velocity when classifier detects idle/stationary
    """
    def __init__(self, lat0: float, lon0: float,
                 pos_noise: float = 3.0,
                 vel_noise: float = 0.8,
                 nhc_noise: float = 0.15,
                 gyro_noise: float = 0.02):
        self.lat0 = lat0
        self.lon0 = lon0

        # State vector x: [pos_E, pos_N, vel_E, vel_N, heading_rad]
        self.x = np.zeros(5, dtype=np.float64)

        # Covariance P
        self.P = np.diag([
            pos_noise**2, pos_noise**2,
            vel_noise**2, vel_noise**2,
            (5.0 * DEG2RAD)**2
        ])

        # Measurement variances
        self.R_gps_pos = np.diag([pos_noise**2, pos_noise**2])
        self.R_gps_vel = np.diag([vel_noise**2, vel_noise**2])
        self.R_nhc = np.array([[nhc_noise**2]])
        self.R_zupt = np.diag([(0.05)**2, (0.05)**2])

        # Process noise Q
        self.Q_base = np.diag([
            0.05**2, 0.05**2,    # pos process noise
            0.20**2, 0.20**2,    # vel process noise
            (0.5 * DEG2RAD)**2   # heading process noise
        ])

    def initialize(self, lat: float, lon: float, heading_deg: float, speed_ms: float = 0.0):
        E, N = lat_lon_to_local_m(lat, lon, self.lat0, self.lon0)
        psi = heading_deg * DEG2RAD
        self.x = np.array([
            E, N,
            speed_ms * np.sin(psi),
            speed_ms * np.cos(psi),
            psi
        ], dtype=np.float64)

    def predict(self, speed_pred_ms: float, yaw_rate_rads: float, dt: float = DT,
                noise_scale: float = 1.0):
        """
        Propagate nominal state using ML speed and gyro yaw rate.
        """
        pos_E, pos_N, vel_E, vel_N, psi = self.x

        # Update heading
        psi_new = (psi + yaw_rate_rads * dt) % (2.0 * np.pi)

        # Nominal velocity along heading
        vE_new = speed_pred_ms * np.sin(psi_new)
        vN_new = speed_pred_ms * np.cos(psi_new)

        # Position integration (trapezoidal / average)
        pos_E_new = pos_E + 0.5 * (vel_E + vE_new) * dt
        pos_N_new = pos_N + 0.5 * (vel_N + vN_new) * dt

        self.x = np.array([pos_E_new, pos_N_new, vE_new, vN_new, psi_new], dtype=np.float64)

        # State transition Jacobian F
        F = np.eye(5, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = speed_pred_ms * np.cos(psi_new)
        F[3, 4] = -speed_pred_ms * np.sin(psi_new)

        Q = self.Q_base * noise_scale * dt
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, gps_lat: float, gps_lon: float,
                    gps_speed_ms: float = None, gps_heading_deg: float = None,
                    accuracy_m: float = 3.0):
        """
        Kalman measurement update with GNSS fix.
        """
        E_meas, N_meas = lat_lon_to_local_m(gps_lat, gps_lon, self.lat0, self.lon0)
        z_pos = np.array([E_meas, N_meas])
        H_pos = np.zeros((2, 5))
        H_pos[0, 0] = 1.0
        H_pos[1, 1] = 1.0

        R_pos = np.eye(2) * max(accuracy_m**2, 1.0)
        y = z_pos - self.x[:2]
        S = H_pos @ self.P @ H_pos.T + R_pos
        K = self.P @ H_pos.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H_pos) @ self.P

        # Optional GNSS velocity update
        if gps_speed_ms is not None and gps_heading_deg is not None and gps_speed_ms > 1.0:
            psi_gps = gps_heading_deg * DEG2RAD
            vE_meas = gps_speed_ms * np.sin(psi_gps)
            vN_meas = gps_speed_ms * np.cos(psi_gps)
            z_vel = np.array([vE_meas, vN_meas])
            H_vel = np.zeros((2, 5))
            H_vel[0, 2] = 1.0
            H_vel[1, 3] = 1.0

            y_v = z_vel - self.x[2:4]
            S_v = H_vel @ self.P @ H_vel.T + self.R_gps_vel
            K_v = self.P @ H_vel.T @ np.linalg.inv(S_v)
            self.x = self.x + K_v @ y_v
            self.P = (np.eye(5) - K_v @ H_vel) @ self.P

    def update_nhc(self):
        """
        Non-Holonomic Constraint update: zero lateral velocity in vehicle frame.
        v_lateral = v_E * cos(psi) - v_N * sin(psi) ≈ 0
        """
        vel_E, vel_N, psi = self.x[2], self.x[3], self.x[4]
        forward_speed = vel_E * np.sin(psi) + vel_N * np.cos(psi)

        # Apply NHC only when moving forward to avoid stationary noise locking
        if forward_speed > 0.5:
            # Measurement: 0 = v_lateral
            z_nhc = 0.0
            v_lat = vel_E * np.cos(psi) - vel_N * np.sin(psi)
            y = np.array([z_nhc - v_lat])

            H_nhc = np.zeros((1, 5))
            H_nhc[0, 2] = np.cos(psi)
            H_nhc[0, 3] = -np.sin(psi)
            H_nhc[0, 4] = -vel_E * np.sin(psi) - vel_N * np.cos(psi)

            S = H_nhc @ self.P @ H_nhc.T + self.R_nhc
            K = self.P @ H_nhc.T / S[0, 0]

            self.x = self.x + (K.ravel() * y[0])
            self.P = (np.eye(5) - K @ H_nhc) @ self.P

    def update_zupt(self):
        """
        Zero Velocity Update: enforce vel_E = 0, vel_N = 0 when vehicle is stationary.
        """
        z_zero = np.array([0.0, 0.0])
        H_zupt = np.zeros((2, 5))
        H_zupt[0, 2] = 1.0
        H_zupt[1, 3] = 1.0

        y = z_zero - self.x[2:4]
        S = H_zupt @ self.P @ H_zupt.T + self.R_zupt
        K = self.P @ H_zupt.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H_zupt) @ self.P

    def get_position_lat_lon(self) -> tuple[float, float]:
        lat, lon = local_m_to_lat_lon(self.x[0], self.x[1], self.lat0, self.lon0)
        return lat, lon

    def get_speed_and_heading(self) -> tuple[float, float]:
        speed = float(np.sqrt(self.x[2]**2 + self.x[3]**2))
        heading_deg = float(self.x[4] * RAD2DEG) % 360.0
        return speed, heading_deg


def run_nhc_ekf_fusion(s_aligned: pd.DataFrame,
                       gps_df: pd.DataFrame,
                       v_predictor,
                       artifact_gate=None,
                       adaptive_noise_net=None,
                       outage_mask: np.ndarray = None,
                       window: int = 30) -> dict:
    """
    Run full ESEKF with NHC on an aligned IMU + GPS sequence.
    outage_mask: boolean array (True where GPS is available, False during outage)
    """
    n = len(s_aligned)
    if outage_mask is None:
        outage_mask = np.ones(n, dtype=bool)

    lat_col = next((c for c in ['Latitude_deg', 'GPS_Latitude_deg', 'GPS_Lat', 'lat'] if c in gps_df.columns), None)
    lon_col = next((c for c in ['Longitude_deg', 'GPS_Longitude_deg', 'GPS_Lon', 'lon'] if c in gps_df.columns), None)
    hdg_col = next((c for c in ['Heading_deg', 'GPS_Bearing_deg', 'GPS_Bearing', 'bearing'] if c in gps_df.columns), None)
    spd_col = next((c for c in ['Velocity_kmh', 'GPS_Speed_ms', 'GPS_Speed_kmh'] if c in gps_df.columns), None)

    lat0 = float(gps_df[lat_col].dropna().iloc[0]) if lat_col else 52.3850
    lon0 = float(gps_df[lon_col].dropna().iloc[0]) if lon_col else -1.5600
    init_hdg = float(gps_df[hdg_col].dropna().iloc[0]) if hdg_col else 0.0

    ekf = ESEKFWithNHC(lat0, lon0)
    ekf.initialize(lat0, lon0, init_hdg)

    est_lat, est_lon = np.zeros(n), np.zeros(n)
    est_spd, est_hdg = np.zeros(n), np.zeros(n)
    est_modes = []

    # Pre-extract IMU window buffer
    feat_cols = [
        'Accel_veh_X_ms2', 'Accel_veh_Y_ms2', 'Accel_veh_Z_ms2',
        'Gyro_veh_yaw_rads', 'Gyro_veh_pitch_rads', 'Gyro_veh_roll_rads'
    ]
    imu_feats = s_aligned[feat_cols].fillna(0.0).values.astype(np.float32)

    for t in range(n):
        # 1. IMU window extraction for velocity prediction
        w_start = max(0, t - window + 1)
        win = imu_feats[w_start:t+1]
        speed_pred = 0.0
        if v_predictor is not None:
            try:
                speed_pred = float(v_predictor.predict_batch(win[np.newaxis])[0])
                speed_pred = max(0.0, speed_pred)
            except Exception:
                speed_pred = float(np.sqrt(ekf.x[2]**2 + ekf.x[3]**2))

        yaw_rate = float(imu_feats[t, 3])  # Gyro_veh_yaw_rads

        # 2. Artifact classifier gating
        is_idle = False
        if artifact_gate is not None:
            try:
                feat_vec = _imu_window_features(win)
                cls_idx, conf = artifact_gate.predict(feat_vec)
                scale = artifact_gate.velocity_scale_factor(cls_idx, conf)
                speed_pred *= scale
                is_idle = (cls_idx == 3)  # 3 is idle
            except Exception:
                pass

        # 3. Dynamic adaptive noise scale
        noise_scale = 1.0
        if adaptive_noise_net is not None:
            noise_scale = float(adaptive_noise_net.predict_scale(win))

        # 4. EKF Time Update (Prediction)
        ekf.predict(speed_pred, yaw_rate, dt=DT, noise_scale=noise_scale)

        # 5. NHC Update or ZUPT
        if is_idle:
            ekf.update_zupt()
        else:
            ekf.update_nhc()

        # 6. GNSS Update (if available and not simulated outage)
        gps_avail = (lat_col is not None and lon_col is not None and
                     outage_mask[t] and not np.isnan(gps_df[lat_col].iloc[t]))
        if gps_avail:
            g_lat = float(gps_df[lat_col].iloc[t])
            g_lon = float(gps_df[lon_col].iloc[t])
            g_spd = None
            if spd_col and not np.isnan(gps_df[spd_col].iloc[t]):
                raw_spd = float(gps_df[spd_col].iloc[t])
                g_spd = raw_spd / 3.6 if 'kmh' in spd_col else raw_spd
            g_hdg = float(gps_df[hdg_col].iloc[t]) if hdg_col and not np.isnan(gps_df[hdg_col].iloc[t]) else None
            g_acc = float(gps_df['GPS_Accuracy_m'].iloc[t]) if 'GPS_Accuracy_m' in gps_df else 3.0
            ekf.update_gnss(g_lat, g_lon, g_spd, g_hdg, accuracy_m=g_acc)
            est_modes.append('GNSS-AIDED')
        else:
            est_modes.append('PURE-DR')

        cur_lat, cur_lon = ekf.get_position_lat_lon()
        cur_spd, cur_hdg = ekf.get_speed_and_heading()
        est_lat[t] = cur_lat
        est_lon[t] = cur_lon
        est_spd[t] = cur_spd
        est_hdg[t] = cur_hdg

    return {
        'lat': est_lat,
        'lon': est_lon,
        'speed': est_spd,
        'heading': est_hdg,
        'modes': est_modes,
        'lat0': lat0,
        'lon0': lon0
    }
