"""
synthetic_data_generator.py
============================
Generates synthetic IO-VNBD-schema-compliant V- and S- CSV files for pipeline
validation.  When real LFS-resolved CSVs are present the pipeline will use those
instead (auto-detected by file size > 10 kB).

Schema source: README_1.pdf (Data in Brief, 2021 – Onyekpe et al.)

V-file columns (29 total, 10 Hz CAN-bus):
  0  GPS_Satellites, 1  Time_s, 2  Latitude_deg, 3  Longitude_deg,
  4  Velocity_kmh, 5  Heading_deg, 6  Height_km, 7  VertVelocity_kmh,
  8  SamplePeriod_s, 9  SteeringAngle_deg,
  10 WheelSpeedFL_rads, 11 WheelSpeedFR_rads,
  12 WheelSpeedRL_rads, 13 WheelSpeedRR_rads,
  14 YawRate_degs, 15 IndicatedSpeed_kmh,
  16 LongAccel_g, 17 LatAccel_g,
  18 Handbrake, 19 GearRequested, 20 Gear,
  21 EngineSpeed_rpm, 22 CoolantTemp_C, 23 ClutchPosition,
  24 BrakePressure_psi, 25 BrakePosition, 26 BatteryVoltage_V,
  27 AirTemp_C, 28 AcceleratorPedal_pct

S-file columns (24 total, 10 Hz smartphone; GPS at 1 Hz):
  0  GPS_Latitude_deg, 1  GPS_Longitude_deg, 2  GPS_Altitude_m,
  3  GPS_Speed_kmh, 4  GPS_Accuracy_m, 5  GPS_Orientation_deg,
  6  GPS_SatellitesInRange, 7  TimeSinceStart_ms, 8  Date,
  9  Accel_X_ms2, 10 Accel_Y_ms2, 11 Accel_Z_ms2,
  12 Gravity_X_ms2, 13 Gravity_Y_ms2, 14 Gravity_Z_ms2,
  15 Gyro_Yaw_rads, 16 Gyro_Pitch_rads, 17 Gyro_Roll_rads,
  18 MagField_X_uT, 19 MagField_Y_uT, 20 MagField_Z_uT,
  21 Orientation_Yaw_deg, 22 Orientation_Roll_deg, 23 Orientation_Pitch_deg
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ─── constants ────────────────────────────────────────────────────────────────
DT          = 0.1          # 10 Hz sample period
G           = 9.81         # m/s²
R_EARTH     = 6371000.0    # metres
DEG2RAD     = np.pi / 180.0
RAD2DEG     = 180.0 / np.pi
# wheel radius Ford Fiesta (approx)
WHEEL_RADIUS_M = 0.295

# ─── V-file column names (must match real dataset header) ──────────────────────
V_COLS = [
    'GPS_Satellites', 'Time_s', 'Latitude_deg', 'Longitude_deg',
    'Velocity_kmh', 'Heading_deg', 'Height_km', 'VertVelocity_kmh',
    'SamplePeriod_s', 'SteeringAngle_deg',
    'WheelSpeedFL_rads', 'WheelSpeedFR_rads',
    'WheelSpeedRL_rads', 'WheelSpeedRR_rads',
    'YawRate_degs', 'IndicatedSpeed_kmh',
    'LongAccel_g', 'LatAccel_g',
    'Handbrake', 'GearRequested', 'Gear',
    'EngineSpeed_rpm', 'CoolantTemp_C', 'ClutchPosition',
    'BrakePressure_psi', 'BrakePosition', 'BatteryVoltage_V',
    'AirTemp_C', 'AcceleratorPedal_pct'
]

# ─── S-file column names ──────────────────────────────────────────────────────
S_COLS = [
    'GPS_Latitude_deg', 'GPS_Longitude_deg', 'GPS_Altitude_m',
    'GPS_Speed_kmh', 'GPS_Accuracy_m', 'GPS_Orientation_deg',
    'GPS_SatellitesInRange', 'TimeSinceStart_ms', 'Date',
    'Accel_X_ms2', 'Accel_Y_ms2', 'Accel_Z_ms2',
    'Gravity_X_ms2', 'Gravity_Y_ms2', 'Gravity_Z_ms2',
    'Gyro_Yaw_rads', 'Gyro_Pitch_rads', 'Gyro_Roll_rads',
    'MagField_X_uT', 'MagField_Y_uT', 'MagField_Z_uT',
    'Orientation_Yaw_deg', 'Orientation_Roll_deg', 'Orientation_Pitch_deg'
]

# ─────────────────────────────────────────────────────────────────────────────
class TrajectorySimulator:
    """Simulate a realistic vehicle trajectory with configurable scenarios."""

    def __init__(self, seed: int = 42,
                 origin_lat: float = 52.406,   # Coventry, UK
                 origin_lon: float = -1.510,
                 scenario: str = 'urban'):
        self.rng = np.random.default_rng(seed)
        self.lat0  = origin_lat
        self.lon0  = origin_lon
        self.scenario = scenario

        # phone-frame to vehicle-frame rotation (yaw offset + tilt)
        # Realistic: phone mounted ~15° yaw, 5° pitch, 2° roll
        self.phone_yaw_offset   =  15.0  # degrees
        self.phone_pitch_offset =   5.0  # degrees
        self.phone_roll_offset  =   2.0  # degrees

        # Per-axis IMU biases (device-specific)
        self.accel_bias = np.array([0.04, -0.02, 0.05])   # m/s²
        self.gyro_bias  = np.array([0.003, -0.002, 0.001]) # rad/s

        # Noise std (realistic MEMS)
        self.accel_noise_std = 0.05   # m/s²
        self.gyro_noise_std  = 0.002  # rad/s

    def _build_scenario_profile(self, duration_s: float):
        """Return arrays of target speed (m/s) and yaw_rate (rad/s) at 10 Hz."""
        n = int(duration_s / DT)
        t = np.arange(n) * DT

        speed   = np.zeros(n)
        yaw_dot = np.zeros(n)

        if self.scenario == 'stationary':
            # pure stationary — for calibration
            speed[:] = 0.0

        elif self.scenario == 'motorway':
            # accelerate to ~110 km/h, stay, gentle curves
            v_target = 30.0   # m/s (~108 kmh)
            accel_phase = int(60 / DT)
            speed[:accel_phase] = np.linspace(0, v_target, accel_phase)
            speed[accel_phase:] = v_target + self.rng.normal(0, 0.5, n - accel_phase)
            speed = np.clip(speed, 0, 35)
            # gentle sinusoidal yaw
            yaw_dot = 0.02 * np.sin(2 * np.pi * t / 120.0)

        elif self.scenario == 'urban':
            # stop-start city driving with roundabouts and turns
            segments = []
            remaining = n
            while remaining > 0:
                # random phase: straight, turn, stop
                phase = self.rng.choice(['straight', 'turn', 'stop'],
                                        p=[0.5, 0.3, 0.2])
                seg_len = int(self.rng.uniform(50, 200))
                seg_len = min(seg_len, remaining)
                if phase == 'straight':
                    v = self.rng.uniform(8, 15)   # 30–54 km/h
                    segments.append((seg_len, v, 0.0))
                elif phase == 'turn':
                    v = self.rng.uniform(3, 8)
                    direction = self.rng.choice([-1, 1])
                    yd = direction * self.rng.uniform(0.2, 0.5)  # rad/s
                    segments.append((seg_len, v, yd))
                else:
                    segments.append((seg_len, 0.0, 0.0))
                remaining -= seg_len

            idx = 0
            for seg_len, v, yd in segments:
                speed[idx:idx + seg_len]   = v
                yaw_dot[idx:idx + seg_len] = yd
                idx += seg_len

            # smooth speed
            from scipy.ndimage import uniform_filter1d
            speed = uniform_filter1d(speed.astype(float), size=20)
            speed = np.clip(speed, 0, 20)

        return speed, yaw_dot

    def simulate(self, duration_s: float = 600.0):
        """
        Returns (v_df, s_df) — DataFrames with V- and S- schemas.
        Ground truth trajectory is stored in _gt_lat, _gt_lon, _gt_speed,
        _gt_heading as instance attributes for evaluation harness use.
        """
        n = int(duration_s / DT)
        t = np.arange(n) * DT

        speed_ms, yaw_dot_rads = self._build_scenario_profile(duration_s)

        # ── integrate heading and position ────────────────────────────────────
        heading_rad = np.zeros(n)
        lat_rad     = np.zeros(n)
        lon_rad     = np.zeros(n)
        lat_rad[0]  = self.lat0 * DEG2RAD
        lon_rad[0]  = self.lon0 * DEG2RAD

        for i in range(1, n):
            heading_rad[i] = heading_rad[i-1] + yaw_dot_rads[i-1] * DT
            d = speed_ms[i-1] * DT
            dlat = d * np.cos(heading_rad[i-1]) / R_EARTH
            dlon = d * np.sin(heading_rad[i-1]) / (R_EARTH * np.cos(lat_rad[i-1]))
            lat_rad[i] = lat_rad[i-1] + dlat
            lon_rad[i] = lon_rad[i-1] + dlon

        lat_deg     = lat_rad * RAD2DEG
        lon_deg     = lon_rad * RAD2DEG
        heading_deg = (heading_rad * RAD2DEG) % 360.0

        # store ground truth for evaluation harness
        self._gt_lat     = lat_deg
        self._gt_lon     = lon_deg
        self._gt_speed   = speed_ms
        self._gt_heading = heading_deg
        self._t          = t

        # ── vehicle longitudinal accel (g) ────────────────────────────────────
        speed_diff = np.gradient(speed_ms, DT)
        long_accel_g = speed_diff / G
        lat_accel_g  = (speed_ms * yaw_dot_rads) / G   # centripetal

        # ── wheel speeds (rad/s) ──────────────────────────────────────────────
        ws = speed_ms / WHEEL_RADIUS_M   # simple approximation
        ws_noise = self.rng.normal(0, 0.02, (n, 4))
        wheel_speeds = np.column_stack([ws]*4) + ws_noise

        # ── build V-file DataFrame ────────────────────────────────────────────
        v_data = {
            'GPS_Satellites':       self.rng.integers(6, 12, n).astype(float),
            'Time_s':               t,
            'Latitude_deg':         lat_deg,
            'Longitude_deg':        lon_deg,
            'Velocity_kmh':         speed_ms * 3.6,
            'Heading_deg':          heading_deg,
            'Height_km':            np.full(n, 0.09) + self.rng.normal(0, 0.0002, n),
            'VertVelocity_kmh':     self.rng.normal(0, 0.1, n),
            'SamplePeriod_s':       np.full(n, DT),
            'SteeringAngle_deg':    yaw_dot_rads * RAD2DEG * 15.0,  # approximate rack ratio
            'WheelSpeedFL_rads':    wheel_speeds[:, 0],
            'WheelSpeedFR_rads':    wheel_speeds[:, 1],
            'WheelSpeedRL_rads':    wheel_speeds[:, 2],
            'WheelSpeedRR_rads':    wheel_speeds[:, 3],
            'YawRate_degs':         yaw_dot_rads * RAD2DEG,
            'IndicatedSpeed_kmh':   speed_ms * 3.6 + self.rng.normal(0, 0.3, n),
            'LongAccel_g':          long_accel_g + self.rng.normal(0, 0.01, n),
            'LatAccel_g':           lat_accel_g  + self.rng.normal(0, 0.01, n),
            'Handbrake':            np.zeros(n),
            'GearRequested':        np.clip(np.floor(speed_ms / 7) + 1, 1, 5),
            'Gear':                 np.clip(np.floor(speed_ms / 7) + 1, 1, 5),
            'EngineSpeed_rpm':      speed_ms * 80 + 800 + self.rng.normal(0, 50, n),
            'CoolantTemp_C':        np.full(n, 88.0) + self.rng.normal(0, 1, n),
            'ClutchPosition':       np.zeros(n),
            'BrakePressure_psi':    np.where(speed_diff < -0.5,
                                             np.abs(speed_diff) * 20, 0),
            'BrakePosition':        np.where(speed_diff < -0.5, 1, 0).astype(float),
            'BatteryVoltage_V':     np.full(n, 12.6) + self.rng.normal(0, 0.1, n),
            'AirTemp_C':            np.full(n, 18.0) + self.rng.normal(0, 0.5, n),
            'AcceleratorPedal_pct': np.clip(speed_diff * 10 + 20, 0, 100),
        }
        v_df = pd.DataFrame(v_data, columns=V_COLS)

        # ── phone-frame IMU (with realistic misalignment + bias + noise) ──────
        yaw_off_r   = self.phone_yaw_offset   * DEG2RAD
        pitch_off_r = self.phone_pitch_offset * DEG2RAD
        roll_off_r  = self.phone_roll_offset  * DEG2RAD

        # Rotation matrix: phone→vehicle  (ZYX Euler: yaw, pitch, roll)
        cy, sy = np.cos(yaw_off_r),   np.sin(yaw_off_r)
        cp, sp = np.cos(pitch_off_r), np.sin(pitch_off_r)
        cr, sr = np.cos(roll_off_r),  np.sin(roll_off_r)
        R_pv = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr            ]
        ])
        # vehicle accel in vehicle frame (x=forward, y=left, z=up)
        accel_v = np.column_stack([
            long_accel_g * G,       # forward
            lat_accel_g  * G,       # lateral
            np.full(n, G)           # gravity component (vehicle frame, approx level)
        ])
        accel_v += self.rng.normal(0, self.accel_noise_std, accel_v.shape)

        # rotate to phone frame
        accel_p = (R_pv.T @ accel_v.T).T   # phone = R_pv^-1 @ veh

        # gravity in phone frame
        grav_v = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, G)])
        grav_p = (R_pv.T @ grav_v.T).T

        # gyro in phone frame
        gyro_v = np.column_stack([
            np.zeros(n),            # roll rate (simplified)
            np.zeros(n),            # pitch rate
            yaw_dot_rads            # yaw rate
        ])
        gyro_p = (R_pv.T @ gyro_v.T).T

        # add biases and noise
        accel_measured = accel_p + self.accel_bias + \
                         self.rng.normal(0, self.accel_noise_std, accel_p.shape)
        gyro_measured  = gyro_p  + self.gyro_bias  + \
                         self.rng.normal(0, self.gyro_noise_std,  gyro_p.shape)

        # GPS sub-sampled to 1 Hz (every 10 samples)
        gps_lat = np.where(np.arange(n) % 10 == 0,
                           lat_deg  + self.rng.normal(0, 0.00003, n),
                           lat_deg[0])
        gps_lon = np.where(np.arange(n) % 10 == 0,
                           lon_deg  + self.rng.normal(0, 0.00004, n),
                           lon_deg[0])
        gps_spd = np.where(np.arange(n) % 10 == 0,
                           speed_ms * 3.6 + self.rng.normal(0, 0.5, n), 0)
        gps_acc = np.where(np.arange(n) % 10 == 0, 3.0 + self.rng.uniform(0,2,n), 0)

        # Orientation from heading (phone yaw = vehicle heading + offset)
        orient_yaw   = (heading_deg + self.phone_yaw_offset) % 360.0
        orient_pitch = np.full(n, self.phone_pitch_offset)
        orient_roll  = np.full(n, self.phone_roll_offset)

        # magnetometer (simplified, UK declination ~-2°)
        mag_x = np.cos(heading_rad + yaw_off_r) * 20 + self.rng.normal(0, 1, n)
        mag_y = np.sin(heading_rad + yaw_off_r) * 20 + self.rng.normal(0, 1, n)
        mag_z = np.full(n, -45.0)             # vertical component at UK latitude

        s_data = {
            'GPS_Latitude_deg':      gps_lat,
            'GPS_Longitude_deg':     gps_lon,
            'GPS_Altitude_m':        np.full(n, 90.0) + self.rng.normal(0, 2, n),
            'GPS_Speed_kmh':         gps_spd,
            'GPS_Accuracy_m':        gps_acc,
            'GPS_Orientation_deg':   np.where(np.arange(n) % 10 == 0, heading_deg, 0),
            'GPS_SatellitesInRange': self.rng.integers(6, 12, n).astype(float),
            'TimeSinceStart_ms':     (t * 1000).astype(int),
            'Date':                  '2020-01-15 10:00:00_000',
            'Accel_X_ms2':           accel_measured[:, 0],
            'Accel_Y_ms2':           accel_measured[:, 1],
            'Accel_Z_ms2':           accel_measured[:, 2],
            'Gravity_X_ms2':         grav_p[:, 0],
            'Gravity_Y_ms2':         grav_p[:, 1],
            'Gravity_Z_ms2':         grav_p[:, 2],
            'Gyro_Yaw_rads':         gyro_measured[:, 2],
            'Gyro_Pitch_rads':       gyro_measured[:, 1],
            'Gyro_Roll_rads':        gyro_measured[:, 0],
            'MagField_X_uT':         mag_x,
            'MagField_Y_uT':         mag_y,
            'MagField_Z_uT':         mag_z,
            'Orientation_Yaw_deg':   orient_yaw,
            'Orientation_Roll_deg':  orient_roll,
            'Orientation_Pitch_deg': orient_pitch,
        }
        s_df = pd.DataFrame(s_data, columns=S_COLS)
        return v_df, s_df


# ─── dataset manifest entry metadata ─────────────────────────────────────────
def compute_gps_distance_km(lat, lon):
    """Haversine distance in km along a GPS track."""
    if len(lat) < 2:
        return 0.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(np.sum(c) * R_EARTH / 1000.0)


# ─── generation config ────────────────────────────────────────────────────────
SYNTHETIC_RUNS = [
    # (run_id, driver, duration_s, scenario, seed)
    # TRAIN: Drivers A, B, D
    ('S1',    'A', 900,  'urban',     10),
    ('S2',    'A', 720,  'motorway',  11),
    ('S3a',   'A', 600,  'urban',     12),
    ('M',     'B', 800,  'urban',     20),
    ('Y1',    'D', 650,  'urban',     40),
    # STATIONARY segments for calibration (Driver E's style)
    ('Vta1a', 'E', 1800, 'stationary', 50),  # >20 min stationary
    # VAL: Driver C (simulated)
    ('S3b',   'A', 500,  'urban',     13),    # treat as "C" subset for structure
    # TEST: Driver E motorway runs
    ('Vfa01', 'E', 750,  'motorway',  60),
    ('Vfa02', 'E', 820,  'motorway',  61),
    ('Vtb1',  'E', 600,  'urban',     70),
    ('Vtb2',  'E', 550,  'urban',     71),
    ('Vw1',   'E', 700,  'urban',     80),
]

# Driver-to-split mapping
DRIVER_SPLIT = {
    'A': 'train',
    'B': 'train',
    'D': 'train',
    'C': 'val',
    'E': 'test',      # E's later runs = test; stationary stays in train/calibration
}


def generate_all(out_dir: str = '../data/synthetic') -> list[dict]:
    """Generate all synthetic runs, return manifest entries."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    for run_id, driver, dur, scenario, seed in SYNTHETIC_RUNS:
        sim = TrajectorySimulator(seed=seed, scenario=scenario)
        v_df, s_df = sim.simulate(duration_s=float(dur))

        v_file = out_path / f'V-{run_id}.csv'
        s_file = out_path / f'S-{run_id}.csv'
        v_df.to_csv(v_file, index=False)
        s_df.to_csv(s_file, index=False)

        n = len(v_df)
        dist = compute_gps_distance_km(v_df['Latitude_deg'].values,
                                       v_df['Longitude_deg'].values)
        split = DRIVER_SPLIT.get(driver, 'train')
        # stationary always goes to calibration regardless
        if scenario == 'stationary':
            split = 'calibration'

        for ftype, fpath, df in [('V', str(v_file), v_df), ('S', str(s_file), s_df)]:
            manifest.append({
                'path':          str(fpath),
                'type':          ftype,
                'run_id':        run_id,
                'driver':        driver,
                'split':         split,
                'sync_status':   'Y',
                'scenario':      scenario,
                'rows':          n,
                'duration_s':    round(n * DT, 1),
                'distance_km':   round(dist, 3),
                'source':        'SYNTHETIC',
                'paired_file':   str(s_file if ftype == 'V' else v_file),
            })

        print(f'  [{run_id}] driver={driver} split={split} '
              f'dur={dur}s rows={n} dist={dist:.2f} km  scenario={scenario}')

    return manifest


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else '../data/synthetic'
    print(f'Generating synthetic IO-VNBD dataset → {out}')
    m = generate_all(out)
    mf_path = Path(out) / 'synthetic_manifest.json'
    with open(mf_path, 'w') as f:
        json.dump(m, f, indent=2)
    print(f'\nManifest: {mf_path}  ({len(m)} entries)')
