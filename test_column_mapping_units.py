import os
import pandas as pd
import numpy as np

V_TEST_FILE = '/Users/vivekwarang/sih/IO-VNBD-master/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/V-Vw4.csv'
S_TEST_FILE = '/Users/vivekwarang/sih/IO-VNBD-master/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw04/S-Vw4.csv'

def find_col(df, keywords):
    for c in df.columns:
        c_clean = c.strip().lower()
        if all(kw.lower() in c_clean for kw in keywords):
            return c
    return None

def test_v_file_mapping():
    print("="*70)
    print("STEP 2A: V- FILE COLUMN MAPPING AND UNIT CONVERSION TEST")
    print("="*70)
    df_v = pd.read_csv(V_TEST_FILE, encoding='latin1')
    
    yaw_rate_col = find_col(df_v, ['yaw rate'])
    height_col   = find_col(df_v, ['height'])
    long_acc_col = find_col(df_v, ['longitudinal acceleration'])
    lat_acc_col  = find_col(df_v, ['lateral acceleration'])
    vel_col      = find_col(df_v, ['velocity'])
    
    # 1. Yaw rate: deg/s -> rad/s
    yaw_rate_deg = df_v[yaw_rate_col].values
    yaw_rate_rad = np.radians(yaw_rate_deg)
    
    # 2. GPS Height: raw values are already in METERS despite 'Height (km)' column header
    # Check max value: if max value < 10.0 then it's in km, else if > 10.0 it's already in meters!
    height_raw = df_v[height_col].values
    if height_raw.max() < 10.0 and height_raw.max() > 0:
        height_m = height_raw * 1000.0 # true km -> m
        height_unit_note = "Converted from km to m (* 1000)"
    else:
        height_m = height_raw # already in meters
        height_unit_note = "Preserved as meters (header 'km' mislabel auto-detected)"
        
    # 3. Accel: g -> m/s²
    long_accel_g = df_v[long_acc_col].values
    lat_accel_g = df_v[lat_acc_col].values
    long_accel_ms2 = long_accel_g * 9.80665
    lat_accel_ms2 = lat_accel_g * 9.80665
    
    # 4. Velocity: km/h -> m/s
    vel_kmh = df_v[vel_col].values
    vel_ms = vel_kmh / 3.6

    print("\nCONVERTED VALUES SANITY CHECK (V- File: V-Vw4.csv):")
    print(f"  Yaw Rate ('{yaw_rate_col}'): raw range [{yaw_rate_deg.min():.2f}, {yaw_rate_deg.max():.2f}] deg/s --> [{yaw_rate_rad.min():.4f}, {yaw_rate_rad.max():.4f}] rad/s")
    print(f"  GPS Height ('{height_col}'): raw range [{height_raw.min():.2f}, {height_raw.max():.2f}] --> [{height_m.min():.2f}, {height_m.max():.2f}] m ({height_unit_note})")
    print(f"  Long Accel ('{long_acc_col}'): raw range [{long_accel_g.min():.4f}, {long_accel_g.max():.4f}] g --> [{long_accel_ms2.min():.2f}, {long_accel_ms2.max():.2f}] m/s²")
    print(f"  Lat Accel ('{lat_acc_col}'): raw range [{lat_accel_g.min():.4f}, {lat_accel_g.max():.4f}] g --> [{lat_accel_ms2.min():.2f}, {lat_accel_ms2.max():.2f}] m/s²")
    print(f"  Velocity ('{vel_col}'): raw range [{vel_kmh.min():.2f}, {vel_kmh.max():.2f}] km/h --> [{vel_ms.min():.2f}, {vel_ms.max():.2f}] m/s")

def test_s_file_mapping():
    print("\n" + "="*70)
    print("STEP 2B: S- FILE COLUMN MAPPING AND UNIT CONVERSION TEST")
    print("="*70)
    df_s = pd.read_csv(S_TEST_FILE, encoding='latin1')
    
    gyro_yaw_col = find_col(df_s, ['gyroscope', 'yaw'])
    alt_col      = find_col(df_s, ['gps', 'altitude'])
    acc_x_col    = find_col(df_s, ['accelerometer', 'x'])
    acc_y_col    = find_col(df_s, ['accelerometer', 'y'])
    speed_col    = find_col(df_s, ['gps', 'speed'])
    
    # 1. Gyro Yaw: rad/s (already rad/s)
    gyro_yaw_rad = df_s[gyro_yaw_col].values
    
    # 2. Accel: m/s² (already m/s²)
    accel_x = df_s[acc_x_col].values
    accel_y = df_s[acc_y_col].values
    
    # 3. GPS Altitude: m (already m)
    gps_alt_m = df_s[alt_col].values
    
    # 4. GPS Speed: Kmh -> m/s
    gps_speed_kmh = df_s[speed_col].values
    gps_speed_ms = gps_speed_kmh / 3.6

    print("\nCONVERTED VALUES SANITY CHECK (S- File: S-Vw4.csv):")
    print(f"  Gyro Yaw ('{gyro_yaw_col}'): range [{gyro_yaw_rad.min():.4f}, {gyro_yaw_rad.max():.4f}] rad/s (preserved)")
    print(f"  GPS Altitude ('{alt_col}'): range [{gps_alt_m.min():.2f}, {gps_alt_m.max():.2f}] m (preserved)")
    print(f"  Accel X ('{acc_x_col}'): range [{accel_x.min():.2f}, {accel_x.max():.2f}] m/s² (preserved)")
    print(f"  Accel Y ('{acc_y_col}'): range [{accel_y.min():.2f}, {accel_y.max():.2f}] m/s² (preserved)")
    print(f"  GPS Speed ('{speed_col}'): range [{gps_speed_kmh.min():.2f}, {gps_speed_kmh.max():.2f}] km/h --> [{gps_speed_ms.min():.2f}, {gps_speed_ms.max():.2f}] m/s")

if __name__ == '__main__':
    test_v_file_mapping()
    test_s_file_mapping()
