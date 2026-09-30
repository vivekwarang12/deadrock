import os
import shutil
import glob
import re

SYNTHETIC_DIR = '/Users/vivekwarang/sih/phase1/data/synthetic'
QUARANTINE_DIR = '/Users/vivekwarang/sih/phase1/data/synthetic_quarantine'

OUTPUT_DIRS = [
    '/Users/vivekwarang/sih/phase1/outputs',
    '/Users/vivekwarang/sih/phase2/outputs',
    '/Users/vivekwarang/sih/phase2/models',
    '/Users/vivekwarang/sih/phase3/outputs',
    '/Users/vivekwarang/sih/phase4/outputs',
]

def purge_synthetic_and_cache():
    print("="*70)
    print("STEP 3: PURGE SYNTHETIC DATA & STUB-ERA CACHED ARTIFACTS")
    print("="*70)
    
    # 1. Quarantine synthetic data
    if os.path.exists(SYNTHETIC_DIR):
        if os.path.exists(QUARANTINE_DIR):
            shutil.rmtree(QUARANTINE_DIR)
        shutil.move(SYNTHETIC_DIR, QUARANTINE_DIR)
        print(f"✅ Quarantined synthetic data directory to: {QUARANTINE_DIR}")
    else:
        print("ℹ️ Synthetic data directory already removed or non-existent.")
        
    # 2. Clear cached outputs & models
    cleared_files = 0
    for out_dir in OUTPUT_DIRS:
        if os.path.exists(out_dir):
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    # Remove cached pkl, pt, npy, json, h5, parquet files
                    if f.endswith(('.pt', '.npy', '.pkl', '.json', '.h5', '.parquet', '.csv', '.png')):
                        fpath = os.path.join(root, f)
                        os.remove(fpath)
                        cleared_files += 1
            print(f"✅ Cleared cache in: {out_dir}")
            
    print(f"\nTotal stub-era cache files purged: {cleared_files}")

if __name__ == '__main__':
    purge_synthetic_and_cache()
