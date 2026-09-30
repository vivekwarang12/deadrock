#!/usr/bin/env python3
"""
io_vnbd_blob_downloader.py
==========================
Download IO-VNBD real CSV files using GitHub's web blob ?raw=true endpoint,
which bypasses the LFS batch API quota limit.

The ?raw=true endpoint serves real LFS content from GitHub's CDN cache when
available, independent of the per-repo LFS bandwidth quota.

Usage:
    python3 io_vnbd_blob_downloader.py [--sync-only] [--limit N] [--dest DIR]
"""
import os, sys, ssl, time, hashlib, json, urllib.request, urllib.error, urllib.parse
import certifi

REPO          = "onyekpeu/IO-VNBD"
BRANCH        = "master"
BASE_BLOB_URL = f"https://github.com/{REPO}/blob/{BRANCH}"
BASE_DIR      = "/Users/vivekwarang/sih/IO-VNBD-master"
TREE_CACHE    = "/tmp/io_vnbd_tree.json"
CTX           = ssl.create_default_context(cafile=certifi.where())

def build_opener():
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=CTX))

def load_tree():
    """Load saved tree or fetch from API."""
    if os.path.exists(TREE_CACHE):
        with open(TREE_CACHE) as f:
            data = json.load(f)
        return data["tree"]
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.github.v3+json"
    })
    with urllib.request.urlopen(req, timeout=30, context=CTX) as resp:
        data = json.load(resp)
    with open(TREE_CACHE, "w") as f:
        json.dump(data, f)
    return data["tree"]

def parse_lfs_pointer(local_path):
    """Return (oid, size) if file is LFS pointer, else None."""
    try:
        with open(local_path, errors="replace") as f:
            content = f.read(512)
        if "git-lfs.github.com" not in content:
            return None
        oid, size = None, None
        for line in content.splitlines():
            if line.startswith("oid sha256:"):
                oid = line.split(":", 1)[1].strip()
            elif line.startswith("size "):
                size = int(line.split()[1])
        return (oid, size) if oid and size else None
    except Exception:
        return None

def download_blob(repo_path, dest_path, opener, retries=3):
    """
    Download a file via GitHub's web blob ?raw=true endpoint.
    Returns (success, bytes_downloaded, error_msg).
    """
    # URL-encode each path component (not the slashes)
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in repo_path.split("/"))
    url = f"{BASE_BLOB_URL}/{encoded}?raw=true"
    tmp  = dest_path + ".dl_tmp"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=60) as resp:
                total = 0
                with open(tmp, "wb") as out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                        total += len(chunk)
            # Check not LFS pointer
            with open(tmp, errors="replace") as f:
                head = f.read(60)
            if "git-lfs.github.com" in head:
                os.remove(tmp)
                return False, 0, "still_lfs_pointer"
            if total < 1000:
                os.remove(tmp)
                return False, total, f"suspiciously_small_{total}B"
            os.replace(tmp, dest_path)
            return True, total, None
        except urllib.error.HTTPError as e:
            err = f"HTTP_{e.code}"
            if e.code in (403, 404):
                if os.path.exists(tmp): os.remove(tmp)
                return False, 0, err  # no point retrying
        except Exception as e:
            err = str(e)[:80]
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(2 ** attempt)

    return False, 0, err

def verify_csv(path):
    """Quick content verification: row count, header check, sample values."""
    import csv as csvmod
    result = {"path": path, "ok": False}
    try:
        with open(path, newline="", errors="replace") as f:
            reader = csvmod.reader(f)
            headers = next(reader)
            rows = 0
            sample_rows = []
            for row in reader:
                rows += 1
                if rows <= 3:
                    sample_rows.append(row)
        result["row_count"] = rows
        result["headers"] = headers[:8]  # first 8 cols
        result["sample"] = sample_rows[:2]
        result["ok"] = rows > 100  # real files have thousands of rows
    except Exception as e:
        result["error"] = str(e)
    return result

def main():
    sync_only = "--sync-only" in sys.argv
    limit_arg = next((sys.argv[i+1] for i, a in enumerate(sys.argv) if a == "--limit"), None)
    limit = int(limit_arg) if limit_arg else None

    print("=== IO-VNBD Blob Downloader ===")
    print(f"  sync_only={sync_only}  limit={limit}")

    tree = load_tree()
    csv_items = [item for item in tree
                 if item["path"].endswith(".csv") and item["type"] == "blob"]
    if sync_only:
        csv_items = [i for i in csv_items if "Synchronised" in i["path"]]
    if limit:
        csv_items = csv_items[:limit]

    print(f"  Target files: {len(csv_items)}")

    opener  = build_opener()
    results = {"downloaded": [], "skipped": [], "failed": [], "verified": []}

    for idx, item in enumerate(csv_items):
        repo_path = item["path"]
        local_path = os.path.join(BASE_DIR, repo_path)
        basename   = os.path.basename(repo_path)

        # Check if already real (not LFS pointer)
        if os.path.exists(local_path):
            ptr = parse_lfs_pointer(local_path)
            if ptr is None:
                # Already real
                results["skipped"].append(local_path)
                if (idx + 1) % 20 == 0:
                    print(f"  [{idx+1}/{len(csv_items)}] {basename} — already real, skip")
                continue
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  [{idx+1}/{len(csv_items)}] {basename} ...", end=" ", flush=True)
        ok, nbytes, err = download_blob(repo_path, local_path, opener)

        if ok:
            print(f"OK ({nbytes/1024:.0f}KB)")
            results["downloaded"].append(local_path)
        else:
            print(f"FAIL — {err}")
            results["failed"].append({"path": local_path, "error": err})

        # Verify every 5th downloaded file
        if ok and len(results["downloaded"]) % 5 == 1:
            v = verify_csv(local_path)
            results["verified"].append(v)
            print(f"    ✓ Verify: {v['row_count']} rows | headers: {v['headers'][:4]}")

    # Summary
    print(f"\n=== Done ===")
    print(f"  Downloaded: {len(results['downloaded'])}")
    print(f"  Skipped (already real): {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")
    if results["failed"]:
        errors = {}
        for f in results["failed"]:
            errors[f["error"]] = errors.get(f["error"], 0) + 1
        for err, cnt in errors.items():
            print(f"    {err}: {cnt}x")

    with open("/tmp/blob_download_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Full results → /tmp/blob_download_results.json")
    return results

if __name__ == "__main__":
    main()
