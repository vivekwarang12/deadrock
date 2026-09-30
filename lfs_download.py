#!/usr/bin/env python3
"""
Download Git-LFS objects directly from GitHub using the LFS Batch API.
Replaces pointer stubs with real CSV data in-place.

Usage:
    python3 lfs_download.py [--sync-only] [--dry-run]
"""
import os, glob, json, sys, hashlib, time, ssl
import urllib.request, urllib.error

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

REPO_OWNER = "onyekpeu"
REPO_NAME  = "IO-VNBD"
LFS_URL    = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git/info/lfs/objects/batch"
BASE_DIRS  = [
    "/Users/vivekwarang/sih/IO-VNBD-master",
    "/Users/vivekwarang/sih/IO-VNBD-fresh",
]
BATCH_SIZE = 50

def parse_pointer(path):
    try:
        with open(path, "r", errors="replace") as fh:
            content = fh.read(512)
        if "git-lfs.github.com" not in content:
            return None
        info = {"path": path}
        for line in content.splitlines():
            if line.startswith("oid sha256:"):
                info["oid"] = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                info["size"] = int(line.split()[1])
        return info if "oid" in info else None
    except Exception:
        return None

def collect_pointers(sync_only=False):
    pointers = []
    for base in BASE_DIRS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != ".git"]
            if sync_only and "Synchronised" not in root:
                continue
            for f in files:
                if f.endswith(".csv"):
                    p = parse_pointer(os.path.join(root, f))
                    if p:
                        pointers.append(p)
    return pointers

def batch_fetch(pointers):
    objects = [{"oid": p["oid"], "size": p["size"]} for p in pointers]
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": objects,
    }).encode("utf-8")
    req = urllib.request.Request(
        LFS_URL,
        data=payload,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  [BATCH ERROR] HTTP {e.code}: {e.read()[:200]}")
        return None
    except Exception as e:
        print(f"  [BATCH ERROR] {e}")
        return None

def download_file(url, dest_path, expected_oid, expected_size, retries=3):
    tmp_path = dest_path + ".lfs_tmp"
    for attempt in range(retries):
        try:
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=SSL_CTX))
            with opener.open(url, timeout=120) as resp, open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
            # legacy urlretrieve replaced above
            if False: urllib.request.urlretrieve(url, tmp_path)
            actual_size = os.path.getsize(tmp_path)
            if actual_size != expected_size:
                os.remove(tmp_path)
                continue
            h = hashlib.sha256()
            with open(tmp_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_oid:
                os.remove(tmp_path)
                continue
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            print(f"    [attempt {attempt+1}] {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(2 ** attempt)
    return False

def main():
    sync_only = "--sync-only" in sys.argv
    dry_run   = "--dry-run"   in sys.argv

    print(f"=== IO-VNBD LFS Downloader ===")
    print(f"  sync_only={sync_only}  dry_run={dry_run}")
    print("Collecting LFS pointer files...")
    pointers = collect_pointers(sync_only=sync_only)
    print(f"  Found {len(pointers)} pointer files to resolve")

    if not pointers:
        print("Nothing to do.")
        return

    total_bytes = sum(p["size"] for p in pointers)
    print(f"  Total download: {total_bytes/1e6:.1f} MB")

    if dry_run:
        for p in pointers[:10]:
            print(f"  {os.path.basename(p['path'])} | {p['size']/1e3:.1f}KB | {p['oid'][:16]}...")
        return

    oid_to_pointers = {}
    for p in pointers:
        oid_to_pointers.setdefault(p["oid"], []).append(p)

    unique_oids = list(oid_to_pointers.keys())
    unique_pointers = [oid_to_pointers[oid][0] for oid in unique_oids]

    downloaded = 0; failed = 0; skipped = 0
    start_time = time.time()

    for i in range(0, len(unique_pointers), BATCH_SIZE):
        batch = unique_pointers[i:i + BATCH_SIZE]
        print(f"\nBatch {i//BATCH_SIZE + 1}  ({len(batch)} objects)...")

        resp = batch_fetch(batch)
        if resp is None:
            failed += len(batch)
            continue

        for obj in resp.get("objects", []):
            oid = obj["oid"]
            dest_files = [p["path"] for p in oid_to_pointers.get(oid, [])]

            if "error" in obj:
                print(f"  [API ERROR] {oid[:12]}: {obj['error']}")
                failed += 1
                continue

            download_url = obj.get("actions", {}).get("download", {}).get("href")
            if not download_url:
                print(f"  [NO URL] {oid[:12]}")
                skipped += 1
                continue

            size = obj["size"]
            primary = dest_files[0]
            basename = os.path.basename(primary)

            try:
                existing_content = open(primary, errors="replace").read(50)
                already_real = (os.path.getsize(primary) == size and
                               "version" not in existing_content)
            except Exception:
                already_real = False

            if already_real:
                skipped += 1
                continue

            print(f"  {basename} ({size/1e3:.0f}KB)...", end=" ", flush=True)
            ok = download_file(download_url, primary, oid, size)
            if ok:
                downloaded += 1
                print("OK")
                import shutil
                for alt in dest_files[1:]:
                    shutil.copy2(primary, alt)
            else:
                failed += 1
                print("FAIL")

    elapsed = time.time() - start_time
    print(f"\n=== Done in {elapsed:.1f}s === downloaded={downloaded} skipped={skipped} failed={failed}")

if __name__ == "__main__":
    main()
