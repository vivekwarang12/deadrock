import os
import sys
import glob
import pandas as pd
import numpy as np

MASTER_DIR = '/Users/vivekwarang/sih/IO-VNBD-master'

# Expected columns per paper Table 3 (V- files) and Table 5 (S- files)
V_EXPECTED_COLS = 29
S_EXPECTED_COLS = 24

# Appendix expected counts from IO-VNBD paper Tables A1-1 through A7
APPENDIX_TARGETS = {
    'V-M': 105995,
    'V-S1': 51790,
    'V-S2': 57608,
    'V-S3a': 39201,
    'V-S3b': 21853,
    'V-S3c': 11739,
    'V-S4': 25553,
    'V-Vw4': 126573,
    'V-Vw14a': 76541,
    'V-Vfa01': 11361,
    'V-Vfa02': 67761,
    'S-M': 105995,
    'S-S1': 51790,
    'S-S2': 57608,
    'S-Vw4': 126573,
    'S-Vfa01': 11361,
    'S-Vfa02': 67761,
}

def verify_all_files():
    print("=" * 80)
    print("STEP 1: FULL-SCALE VERIFICATION OF ALL 564 FILES IN IO-VNBD-MASTER")
    print("=" * 80)
    
    all_files = []
    for root, dirs, files in os.walk(MASTER_DIR):
        for f in files:
            if f.endswith('.csv'):
                all_files.append(os.path.join(root, f))
                
    print(f"Total CSV files found: {len(all_files)}")
    
    results = []
    v_total_rows = 0
    s_total_rows = 0
    v_file_count = 0
    s_file_count = 0
    
    matched_appendix = {}

    for idx, path in enumerate(all_files, 1):
        rel_path = os.path.relpath(path, MASTER_DIR)
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        size = os.path.getsize(path)
        
        status = "PASS"
        reason = ""
        rows = 0
        cols = 0
        file_type = "UNKNOWN"
        
        # 1. Size & LFS stub check
        if size < 500:
            status = "FAIL"
            reason = "LFS pointer stub (< 500B)"
        else:
            try:
                # Read head to check LFS text
                with open(path, 'r', encoding='latin1', errors='ignore') as f:
                    first_line = f.readline()
                if "version https://git-lfs" in first_line:
                    status = "FAIL"
                    reason = "LFS pointer text detected"
                else:
                    df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip', low_memory=False)
                    rows = len(df)
                    cols = len(df.columns)
                    
                    if fname.startswith('V-') or fname.lower().startswith('v-') or fname.startswith('V'):
                        file_type = "V- (ECU/VBOX)"
                    elif fname.startswith('S-') or fname.lower().startswith('s-') or fname.startswith('S'):
                        file_type = "S- (Smartphone)"
                        
                    # Determine column count match
                    if fname.startswith('V-') or fname.startswith('V') or 'V-Dataset' in path:
                        file_type = "V-"
                        v_file_count += 1
                        v_total_rows += rows
                        if cols != V_EXPECTED_COLS:
                            status = "WARNING"
                            reason = f"Col count {cols} != expected {V_EXPECTED_COLS}"
                    elif fname.startswith('S-') or fname.startswith('S') or 'S-Dataset' in path:
                        file_type = "S-"
                        s_file_count += 1
                        s_total_rows += rows
                        if cols != S_EXPECTED_COLS:
                            status = "WARNING"
                            reason = f"Col count {cols} != expected {S_EXPECTED_COLS}"
                            
                    # GPS Bounding box check
                    lat_col = [c for c in df.columns if 'lat' in c.lower()]
                    lon_col = [c for c in df.columns if 'lon' in c.lower()]
                    if lat_col and lon_col:
                        lats = pd.to_numeric(df[lat_col[0]], errors='coerce').dropna()
                        lons = pd.to_numeric(df[lon_col[0]], errors='coerce').dropna()
                        if len(lats) > 0 and len(lons) > 0:
                            min_lat, max_lat = lats.min(), lats.max()
                            min_lon, max_lon = lons.min(), lons.max()
                            # UK / Western Europe / Nigeria range check
                            if not (40.0 <= max_lat <= 65.0 and -12.0 <= min_lon <= 15.0) and not (4.0 <= max_lat <= 15.0 and 2.0 <= min_lon <= 15.0):
                                if min_lat != 0.0 or max_lat != 0.0:
                                    status = "WARNING"
                                    reason += f" | GPS bounds out of range: Lat [{min_lat:.2f}, {max_lat:.2f}], Lon [{min_lon:.2f}, {max_lon:.2f}]"

                    # Check Appendix target matching
                    # Strip extension
                    key = fname.replace('.csv', '')
                    if key in APPENDIX_TARGETS:
                        matched_appendix[key] = {
                            'actual_rows': rows,
                            'target_rows': APPENDIX_TARGETS[key],
                            'delta': rows - APPENDIX_TARGETS[key],
                            'pct_diff': ((rows - APPENDIX_TARGETS[key]) / APPENDIX_TARGETS[key]) * 100
                        }

            except Exception as e:
                status = "FAIL"
                reason = f"Parse error: {str(e)}"
                
        results.append({
            'rel_path': rel_path,
            'fname': fname,
            'file_type': file_type,
            'size_bytes': size,
            'rows': rows,
            'cols': cols,
            'status': status,
            'reason': reason
        })

    results_df = pd.DataFrame(results)
    
    # Save full audit table
    audit_csv = '/Users/vivekwarang/sih/FULL_DATASET_AUDIT_TABLE.csv'
    results_df.to_csv(audit_csv, index=False)
    print(f"\nSaved full audit table to: {audit_csv}")
    
    # Print summary
    pass_count = len(results_df[results_df['status'] == 'PASS'])
    warn_count = len(results_df[results_df['status'] == 'WARNING'])
    fail_count = len(results_df[results_df['status'] == 'FAIL'])
    
    print("\n" + "="*80)
    print("AUDIT SUMMARY STATS:")
    print("="*80)
    print(f"Total Files Audited: {len(results_df)}")
    print(f"PASS (Clean Header & Data):    {pass_count}")
    print(f"WARNING (Minor Col/GPS dev):  {warn_count}")
    print(f"FAIL (Corrupt/LFS Stub):      {fail_count}")
    
    print("\n" + "="*80)
    print("AGGREGATE ROW SUMS VS PAPER STATEMENTS:")
    print("="*80)
    print(f"S- (Smartphone) Total Files: {s_file_count}")
    print(f"S- Total Rows:               {s_total_rows:,} rows")
    print(f"S- Paper Target (~2.2M):      2,200,000 rows")
    print(f"S- Delta:                     {s_total_rows - 2200000:+,} rows ({((s_total_rows - 2200000)/2200000)*100:+.2f}%)")
    print("-" * 50)
    print(f"V- (ECU / VBOX) Total Files: {v_file_count}")
    print(f"V- Total Rows:               {v_total_rows:,} rows")
    print(f"V- Paper Target (~1.4M):      1,400,000 rows")
    print(f"V- Delta:                     {v_total_rows - 1400000:+,} rows ({((v_total_rows - 1400000)/1400000)*100:+.2f}%)")
    
    print("\n" + "="*80)
    print("APPENDIX TABLE ROW-COUNT CROSS-REFERENCE MATCHES:")
    print("="*80)
    print(f"{'Dataset Name':<15} | {'Actual Rows':<12} | {'Paper Target':<12} | {'Delta':<10} | {'% Diff':<8}")
    print("-" * 65)
    for k, v in matched_appendix.items():
        print(f"{k:<15} | {v['actual_rows']:<12,} | {v['target_rows']:<12,} | {v['delta']:<+10,} | {v['pct_diff']:+.2f}%")
        
    return results_df

if __name__ == '__main__':
    verify_all_files()
