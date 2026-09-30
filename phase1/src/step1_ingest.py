"""
step1_ingest.py  –  STEP 1: Data ingestion & manifest builder
==============================================================
Scans both the real IO-VNBD tree (LFS files, identified by size > 10 kB)
and the synthetic fallback directory, loads every V- and S- CSV into a
consistent schema, and writes a unified manifest.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

REPO_ROOT      = Path('/Users/vivekwarang/sih/IO-VNBD-master')
SYNTHETIC_ROOT = Path('/Users/vivekwarang/sih/phase1/data/synthetic')
OUTPUT_DIR     = Path('/Users/vivekwarang/sih/phase1/outputs')
MANIFEST_PATH  = OUTPUT_DIR / 'dataset_manifest.json'

DT = 0.1      # 10 Hz
LFS_POINTER_MAX_BYTES = 200   # anything ≤ this is a pointer, not real data

# ──────────────────── column name sets ────────────────────────────────────────
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

R_EARTH = 6_371_000.0  # m


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> float:
    lat_r = np.radians(lat);  lon_r = np.radians(lon)
    dlat  = np.diff(lat_r);   dlon  = np.diff(lon_r)
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon/2)**2
    return float(np.sum(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * R_EARTH / 1000.0)


# ──────────────────── driver / scenario inference from filename ───────────────
DRIVER_MAP = {
    'S1': 'A', 'S2': 'A', 'S3a': 'A', 'S3b': 'A', 'S3c': 'A', 'S4': 'A',
    'M':  'B',
    'Y1': 'D',
    'Vfa01': 'E', 'Vfa02': 'E',
    'Vta': 'E',   # prefix for vta series
    'Vtb': 'E',   # prefix for vtb series
    'Vw':  'E',   # prefix for vw series
}

SCENARIO_KEYWORDS = {
    'motorway': ['vfa', 'S2'],
    'urban':    ['S1', 'S3', 'S4', 'M', 'Y1', 'vtb', 'vw'],
    'stationary': ['vta1a', 'Vta1a', 'stationary'],
}


def infer_driver(stem: str) -> str:
    """Best-effort driver inference from file stem (e.g. 'V-S1' → 'A')."""
    stem_clean = stem.replace('V-', '').replace('S-', '')
    for key, drv in DRIVER_MAP.items():
        if stem_clean.startswith(key):
            return drv
    # vta / vtb / vw prefix
    for prefix in ('vta', 'vtb', 'Vw', 'Vta', 'Vtb', 'Vfa', 'Vfb'):
        if stem_clean.lower().startswith(prefix.lower()):
            return 'E'
    return 'Unknown'


def infer_scenario(stem: str) -> str:
    s = stem.lower()
    if any(k.lower() in s for k in SCENARIO_KEYWORDS['stationary']):
        return 'stationary'
    if any(k.lower() in s for k in SCENARIO_KEYWORDS['motorway']):
        return 'motorway'
    return 'urban'


# driver→split mapping (by-driver split, no leakage)
DRIVER_SPLIT = {
    'A': 'train',
    'B': 'train',
    'D': 'train',
    'C': 'val',
    'E': 'test',
}


def is_lfs_pointer(path: Path) -> bool:
    return path.stat().st_size <= LFS_POINTER_MAX_BYTES


# ──────────────────── loader ─────────────────────────────────────────────────
def find_col(df, keywords):
    for c in df.columns:
        c_clean = str(c).strip().lower()
        if all(kw.lower() in c_clean for kw in keywords):
            return c
    return None

def load_csv(path: Path, file_type: str) -> pd.DataFrame | None:
    """
    Load a V- or S- CSV. Handles:
    - Fuzzy column header matching for raw IO-VNBD headers
    - Explicit unit conversions (V- yaw rate deg/s -> rad/s, V- accel g -> m/s², V- height, S- units)
    - LFS pointers (returns None + logs warning)
    - Corrupt / malformed files (returns None + logs error)
    """
    if is_lfs_pointer(path):
        log.warning(f'LFS pointer (not real data): {path.name}')
        return None
    try:
        df_raw = pd.read_csv(path, encoding='latin1', on_bad_lines='skip', low_memory=False)
        if len(df_raw) == 0:
            return None
            
        df = pd.DataFrame(index=df_raw.index)
        
        if file_type == 'V':
            # Map V- columns
            col_map = {
                'GPS_Satellites': find_col(df_raw, ['gps', 'satellite']),
                'Time_s': find_col(df_raw, ['time', 'start']),
                'Latitude_deg': find_col(df_raw, ['latitude']),
                'Longitude_deg': find_col(df_raw, ['longitude']),
                'Velocity_kmh': find_col(df_raw, ['velocity']),
                'Heading_deg': find_col(df_raw, ['heading']),
                'Height_km': find_col(df_raw, ['height']),
                'VertVelocity_kmh': find_col(df_raw, ['vertical', 'velocity']),
                'SamplePeriod_s': find_col(df_raw, ['sample', 'period']),
                'SteeringAngle_deg': find_col(df_raw, ['steering', 'angle']),
                'WheelSpeedFL_rads': find_col(df_raw, ['wheel', 'front', 'left']),
                'WheelSpeedFR_rads': find_col(df_raw, ['wheel', 'front', 'right']),
                'WheelSpeedRL_rads': find_col(df_raw, ['wheel', 'rear', 'left']),
                'WheelSpeedRR_rads': find_col(df_raw, ['wheel', 'rear', 'right']),
                'YawRate_degs': find_col(df_raw, ['yaw', 'rate']),
                'IndicatedSpeed_kmh': find_col(df_raw, ['indicated', 'speed']),
                'LongAccel_g': find_col(df_raw, ['longitudinal', 'acceleration']),
                'LatAccel_g': find_col(df_raw, ['lateral', 'acceleration']),
                'Handbrake': find_col(df_raw, ['handbrake']),
                'GearRequested': find_col(df_raw, ['gear', 'requested']),
                'Gear': find_col(df_raw, ['gear']),
                'EngineSpeed_rpm': find_col(df_raw, ['engine', 'speed']),
                'CoolantTemp_C': find_col(df_raw, ['coolant']),
                'ClutchPosition': find_col(df_raw, ['clutch']),
                'BrakePressure_psi': find_col(df_raw, ['brake', 'pressure']),
                'BrakePosition': find_col(df_raw, ['brake', 'position']),
                'BatteryVoltage_V': find_col(df_raw, ['battery']),
                'AirTemp_C': find_col(df_raw, ['air', 'temp']),
                'AcceleratorPedal_pct': find_col(df_raw, ['accelerator']),
            }
            for col_name in V_COLS:
                raw_c = col_map.get(col_name)
                if raw_c and raw_c in df_raw.columns:
                    df[col_name] = pd.to_numeric(df_raw[raw_c], errors='coerce')
                else:
                    df[col_name] = np.nan
                    
            # Apply conversions to standardized internal units
            # 1. YawRate_degs -> convert to rad/s in new column YawRate_rads (or convert YawRate_degs to rad/s)
            if 'YawRate_degs' in df.columns:
                df['YawRate_rads'] = np.radians(df['YawRate_degs'])
            # 2. Accel g -> m/s²
            if 'LongAccel_g' in df.columns:
                df['LongAccel_ms2'] = df['LongAccel_g'] * 9.80665
            if 'LatAccel_g' in df.columns:
                df['LatAccel_ms2'] = df['LatAccel_g'] * 9.80665
            # 3. Height_km -> auto-check if values are in meters or km
            if 'Height_km' in df.columns:
                h_vals = df['Height_km'].dropna()
                if len(h_vals) > 0 and h_vals.max() < 10.0 and h_vals.max() > 0:
                    df['Height_m'] = df['Height_km'] * 1000.0
                else:
                    df['Height_m'] = df['Height_km'] # already in meters

        else:
            # Map S- columns
            col_map = {
                'GPS_Latitude_deg': find_col(df_raw, ['gps', 'latitude']),
                'GPS_Longitude_deg': find_col(df_raw, ['gps', 'longitude']),
                'GPS_Altitude_m': find_col(df_raw, ['gps', 'altitude']),
                'GPS_Speed_kmh': find_col(df_raw, ['gps', 'speed']),
                'GPS_Accuracy_m': find_col(df_raw, ['gps', 'accuracy']),
                'GPS_Orientation_deg': find_col(df_raw, ['gps', 'orientation']),
                'GPS_SatellitesInRange': find_col(df_raw, ['satellites']),
                'TimeSinceStart_ms': find_col(df_raw, ['time', 'start']),
                'Date': find_col(df_raw, ['date']),
                'Accel_X_ms2': find_col(df_raw, ['accelerometer', 'x']),
                'Accel_Y_ms2': find_col(df_raw, ['accelerometer', 'y']),
                'Accel_Z_ms2': find_col(df_raw, ['accelerometer', 'z']),
                'Gravity_X_ms2': find_col(df_raw, ['gravity', 'x']),
                'Gravity_Y_ms2': find_col(df_raw, ['gravity', 'y']),
                'Gravity_Z_ms2': find_col(df_raw, ['gravity', 'z']),
                'Gyro_Yaw_rads': find_col(df_raw, ['gyroscope', 'yaw']),
                'Gyro_Pitch_rads': find_col(df_raw, ['gyroscope', 'pitch']),
                'Gyro_Roll_rads': find_col(df_raw, ['gyroscope', 'roll']),
                'MagField_X_uT': find_col(df_raw, ['magnetic', 'x']),
                'MagField_Y_uT': find_col(df_raw, ['magnetic', 'y']),
                'MagField_Z_uT': find_col(df_raw, ['magnetic', 'z']),
                'Orientation_Yaw_deg': find_col(df_raw, ['orientation', 'yaw']),
                'Orientation_Roll_deg': find_col(df_raw, ['orientation', 'roll']),
                'Orientation_Pitch_deg': find_col(df_raw, ['orientation', 'pitch']),
            }
            for col_name in S_COLS:
                raw_c = col_map.get(col_name)
                if raw_c and raw_c in df_raw.columns:
                    if col_name != 'Date':
                        df[col_name] = pd.to_numeric(df_raw[raw_c], errors='coerce')
                    else:
                        df[col_name] = df_raw[raw_c]
                else:
                    df[col_name] = np.nan

        # Drop rows that are entirely NaN
        df.dropna(how='all', inplace=True)
        return df
    except Exception as exc:
        log.error(f'{path.name}: parse failed – {exc}')
        return None


# ──────────────────── scanner ─────────────────────────────────────────────────
def scan_directory(base: Path, source_tag: str) -> list[dict]:
    """Walk a directory tree and collect all V-*.csv and S-*.csv files."""
    entries = []
    for path in sorted(base.rglob('*.csv')):
        stem = path.stem          # e.g. 'V-M'  or  'S-S1'
        if stem.startswith('V-'):
            ftype = 'V'
        elif stem.startswith('S-'):
            ftype = 'S'
        else:
            continue              # skip non-V/S files (e.g. 'd_t1_start1.csv')
        entries.append((path, ftype, stem, source_tag))
    return entries


# ──────────────────── main ────────────────────────────────────────────────────
def build_manifest() -> tuple[list[dict], dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. collect file paths ─────────────────────────────────────────────────
    file_entries = []
    if REPO_ROOT.exists():
        file_entries += scan_directory(REPO_ROOT, 'REAL')
    elif Path('/Users/vivekwarang/sih/IO-VNBD-fresh').exists():
        file_entries += scan_directory(Path('/Users/vivekwarang/sih/IO-VNBD-fresh'), 'REAL')

    log.info(f'Found {len(file_entries)} V-/S- CSV paths to process')

    # ── 2. build paired-file index (V→S and S→V) ──────────────────────────────
    # Map run_id → {V: path, S: path}
    run_index: dict[str, dict] = {}
    for path, ftype, stem, src in file_entries:
        run_id = stem[2:]   # strip 'V-' or 'S-'
        if run_id not in run_index:
            run_index[run_id] = {}
        run_index[run_id][ftype] = path

    # ── 3. load each file, compute stats, build manifest ─────────────────────
    manifest       = []
    loaded_dfs     = {}   # path_str → DataFrame  (for downstream use)
    parse_failures = []

    for path, ftype, stem, src in file_entries:
        run_id  = stem[2:]
        driver  = infer_driver(stem)
        scenario = infer_scenario(stem)
        split   = DRIVER_SPLIT.get(driver, 'train')
        if scenario == 'stationary':
            split = 'calibration'

        paired_path = None
        other_type  = 'S' if ftype == 'V' else 'V'
        if run_id in run_index and other_type in run_index[run_id]:
            paired_path = str(run_index[run_id][other_type])

        # skip LFS pointers
        if is_lfs_pointer(path):
            manifest.append({
                'path': str(path), 'type': ftype, 'run_id': run_id,
                'driver': driver, 'split': split,
                'sync_status': 'Y' if paired_path else 'N',
                'scenario': scenario, 'rows': None, 'duration_s': None,
                'distance_km': None, 'source': src + '_LFS_POINTER',
                'paired_file': paired_path, 'parse_status': 'LFS_POINTER'
            })
            continue

        df = load_csv(path, ftype)
        if df is None:
            parse_failures.append(str(path))
            manifest.append({
                'path': str(path), 'type': ftype, 'run_id': run_id,
                'driver': driver, 'split': split,
                'sync_status': 'Y' if paired_path else 'N',
                'scenario': scenario, 'rows': None, 'duration_s': None,
                'distance_km': None, 'source': src,
                'paired_file': paired_path, 'parse_status': 'FAILED'
            })
            continue

        # compute stats
        n       = len(df)
        dur_s   = round(n * DT, 1)
        dist_km = 0.0
        if ftype == 'V':
            lat = df['Latitude_deg'].dropna().values
            lon = df['Longitude_deg'].dropna().values
            if len(lat) > 1:
                dist_km = haversine_km(lat, lon)
        else:
            lat = df['GPS_Latitude_deg'].dropna().values
            lon = df['GPS_Longitude_deg'].dropna().values
            if len(lat) > 1:
                dist_km = haversine_km(lat, lon)

        loaded_dfs[str(path)] = df
        manifest.append({
            'path':        str(path),
            'type':        ftype,
            'run_id':      run_id,
            'driver':      driver,
            'split':       split,
            'sync_status': 'Y' if paired_path else 'N',
            'scenario':    scenario,
            'rows':        int(n),
            'duration_s':  dur_s,
            'distance_km': round(dist_km, 3),
            'source':      src,
            'paired_file': paired_path,
            'parse_status':'OK'
        })
        log.info(f'  OK  {path.name:<25s} rows={n:>7,} dur={dur_s:>7.1f}s '
                 f'dist={dist_km:>6.2f}km  driver={driver} split={split}')

    # ── 4. summary stats ──────────────────────────────────────────────────────
    ok      = [m for m in manifest if m['parse_status'] == 'OK']
    lfs_ptr = [m for m in manifest if 'LFS_POINTER' in m.get('parse_status','')]
    failed  = [m for m in manifest if m['parse_status'] == 'FAILED']

    summary = {
        'total_paths':    len(manifest),
        'loaded_ok':      len(ok),
        'lfs_pointers':   len(lfs_ptr),
        'parse_failures': len(failed),
        'failure_paths':  parse_failures,
        'split_counts':   {s: sum(1 for m in ok if m['split'] == s)
                           for s in ('train','val','test','calibration')},
        'type_counts':    {t: sum(1 for m in ok if m['type'] == t)
                           for t in ('V','S')},
    }

    log.info('\n── MANIFEST SUMMARY ──')
    for k, v in summary.items():
        log.info(f'  {k}: {v}')

    # ── 5. save ───────────────────────────────────────────────────────────────
    with open(MANIFEST_PATH, 'w') as f:
        json.dump({'summary': summary, 'files': manifest}, f, indent=2)
    log.info(f'\nManifest saved → {MANIFEST_PATH}')

    return manifest, loaded_dfs


if __name__ == '__main__':
    build_manifest()
