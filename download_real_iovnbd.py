import os
import urllib.parse
import urllib.request
import ssl
import certifi
import time
import sys

BASE_DIR = '/Users/vivekwarang/sih/IO-VNBD-master'
REPO_BASE_URL = 'https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/'

ctx = ssl.create_default_context(cafile=certifi.where())

def get_relative_path(full_path, base_path):
    rel = os.path.relpath(full_path, base_path)
    return rel

def download_file(rel_path, target_path):
    # Encode each path component
    parts = rel_path.split(os.sep)
    encoded_parts = [urllib.parse.quote(p) for p in parts]
    encoded_url_path = '/'.join(encoded_parts)
    download_url = REPO_BASE_URL + encoded_url_path
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    req = urllib.request.Request(download_url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
            # Verify data is real CSV (not HTML or LFS pointer)
            if len(data) < 500:
                print(f"❌ [STUB/SHORT] {rel_path} - size: {len(data)} bytes")
                return False
            
            first_line = data[:200].decode('utf-8', errors='ignore').split('\n')[0]
            if "version https://git-lfs" in first_line:
                print(f"❌ [LFS POINTER] {rel_path}")
                return False
            
            with open(target_path, 'wb') as f:
                f.write(data)
            
            print(f"✅ [SUCCESS] {rel_path} ({len(data):,} bytes)")
            return True
    except Exception as e:
        print(f"❌ [ERROR] {rel_path}: {e}")
        return False

def main():
    print(f"Starting real dataset acquisition for {BASE_DIR}...")
    stubs_to_download = []
    
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith('.csv'):
                full_path = os.path.join(root, f)
                if os.path.getsize(full_path) < 500:
                    stubs_to_download.append(full_path)
    
    print(f"Found {len(stubs_to_download)} LFS CSV stubs to replace.")
    
    success_count = 0
    fail_count = 0
    
    for idx, full_path in enumerate(stubs_to_download, 1):
        rel_path = get_relative_path(full_path, BASE_DIR)
        print(f"[{idx}/{len(stubs_to_download)}] Downloading: {rel_path} ...", end=" ")
        
        # Try download up to 3 times
        success = False
        for attempt in range(3):
            if download_file(rel_path, full_path):
                success = True
                break
            time.sleep(1)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            
        # Mild throttle to be polite
        time.sleep(0.1)

    print("\n" + "="*60)
    print(f"DOWNLOAD COMPLETE: {success_count} succeeded, {fail_count} failed out of {len(stubs_to_download)} total files.")
    print("="*60)

if __name__ == '__main__':
    main()
