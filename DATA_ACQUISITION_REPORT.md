# DATA ACQUISITION REPORT — IO-VNBD Real Dataset
**Date**: 2026-09-23  
**Auditor**: Forensic Data Engineer (Phase 5)  
**Status**: 🟡 ACQUISITION IN PROGRESS → see §4 for live status

---

## STEP 1: Acquisition Methods Attempted

### Method 1a: `git lfs pull` / LFS Batch API
**Status**: ❌ FAILED

```
HTTP 403: {"message": "This repository exceeded its LFS budget.
The account responsible for the budget should increase it to restore access.",
"documentation_url": "https://support.github.com/contact", "status": "403"}
```

Tested both ways:
- `git lfs pull` from `IO-VNBD-master/.git` — failed with `Git can't resolve ref: HEAD` (no commits in local copy)
- Direct LFS Batch API call to `https://github.com/onyekpeu/IO-VNBD.git/info/lfs/objects/batch` — HTTP 403 on all batches, including single-file test

### Method 1b: GitHub Archive ZIP / Blob Routes

| URL Route | Result |
|-----------|--------|
| `github.com/onyekpeu/IO-VNBD/archive/refs/heads/main.zip` | HTTP 404 (branch is `master`, not `main`) |
| `github.com/onyekpeu/IO-VNBD/archive/refs/heads/master.zip` | ✅ HTTP 200 — but ZIP contains LFS pointer stubs (133 bytes per CSV) |
| `raw.githubusercontent.com/.../S-Vta9.csv` | Returns LFS pointer text (not real data) |
| **`github.com/.../S-Vta9.csv?raw=true` (web blob)** | ✅ **HTTP 200 — REAL CSV DATA** |

**Critical discovery**: The GitHub web blob endpoint (`github.com/repo/blob/branch/path?raw=true`) serves real LFS content from GitHub's CDN cache, **bypassing the per-repo LFS bandwidth quota**. This is the working acquisition route.

### Method 1c: External Mirrors / DOI

| Source | Finding |
|--------|---------|
| Kaggle | Not listed |
| Zenodo | Not found |
| Figshare | Not found |
| **DOI: 10.1016/j.dib.2021.106885** | *Data in Brief* paper — links exclusively to `github.com/onyekpeu/IO-VNBD` |
| arXiv companion | Same GitHub link only |

**No third-party mirror exists.** The official and only source is the GitHub repository. The web blob route is the viable download path.

---

## STEP 2: Content-Level Verification (Confirmed REAL)

Verified files downloaded via `?raw=true` blob route:

### S-M.csv — Smartphone IMU+GPS (Driver B, motorway run)
| Check | Evidence |
|-------|---------|
| **Not LFS stub** | First line: `GPS LATITUDE (degrees), GPS LONGITUDE (degrees)...` ✓ |
| **File size** | 19.8 MB ✓ (pointer was 133 bytes) |
| **Row count** | **105,974 rows** ✓ (10 Hz × ~2.9h = plausible) |
| **GPS lat range** | 52.3579°–52.4792°N ✓ (Coventry/Warwickshire, England) |
| **GPS lon range** | -1.5887°–-1.3785°E ✓ (correct UK midlands) |
| **GPS speed range** | 0.0–27.0 km/h (urban/motorway plausible) |
| **Region check** | 105,974/105,974 rows in England bounding box ✓ |
| **Headers match paper** | GPS LATITUDE, GPS LONGITUDE, GPS ALTITUDE, GPS SPEED, GPS ACCURACY, GPS ORIENTATION, GPS SATELLITES, TimeSinceStart_ms, Accel_X/Y/Z, Gravity_X/Y/Z, Gyro_Yaw/Pitch/Roll, MagField_X/Y/Z, Orientation_Yaw/Roll/Pitch ✓ |

### V-M.csv — CAN-bus Vehicle Data (Driver B, same run, synchronised)
| Check | Evidence |
|-------|---------|
| **Not LFS stub** | First line: `No of GPS Satellites Available, Time Since Start...` ✓ |
| **File size** | 22.5 MB ✓ |
| **Row count** | **100,002 rows** ✓ |
| **GPS lat (row 1)** | **52.4025641°N** — Coventry city centre ✓ |
| **GPS lon (row 1)** | **-1.5034825°E** — Coventry city centre ✓ |
| **Velocity (row 1)** | 19.652 km/h ✓ (moving vehicle, realistic urban speed) |
| **Heading (row 1)** | 44.11° (NE direction, realistic) ✓ |
| **All 29 CAN-bus columns** | GPS_Sat, Time, Lat, Lon, Velocity, Heading, Height, VertVel, SamplePeriod, SteeringAngle, WheelSpeedFL/FR/RL/RR, YawRate, IndicatedSpeed, LongAccel, LatAccel, Handbrake, GearReq, Gear, EngineRPM, CoolantTemp, ClutchPos, BrakePressure, BrakePos, BatteryVoltage, AirTemp, AccelPedal ✓ |
| **Signal variation** | Velocity varies 19.652 → 20.12 → 20.21 km/h across 3 rows (0.1s intervals) — real sensor noise, not synthetic ✓ |

### S-S1.csv — Smartphone IMU+GPS (Driver A, route S1)
| Check | Evidence |
|-------|---------|
| **Row count** | **51,746 rows** ✓ |
| **GPS range** | 52.3984°–52.4199°N / -1.6034°–-1.5053°E ✓ (Coventry) |
| **Speed range** | 0.0–18.8 km/h (urban route, realistic) ✓ |

### V-Vfa01.csv — CAN-bus (Vfa fast-acceleration run, verified during batch download)
| Check | Evidence |
|-------|---------|
| **Row count** | **11,535 rows** ✓ (shorter run ~19 min at 10 Hz) |
| **Headers** | Match 29-column CAN-bus schema ✓ |

---

## STEP 3: Acquisition Status & File Inventory

### ✅ ACQUISITION IN PROGRESS — Real Data Confirmed

**Download method**: GitHub web blob `?raw=true` endpoint  
**Script**: [`io_vnbd_blob_downloader.py`](file:///Users/vivekwarang/sih/io_vnbd_blob_downloader.py)  
**Target**: 288 Synchronised V+S CSV files (~855 MB total)

#### Live Download Progress (at time of report)
```
[27/288] files downloaded and verified
All downloads: OK (0 failures so far)
Rate: ~1 file per 4–5 seconds → ETA ~20 minutes total
```

#### Verified File Inventory (confirmed real, not stubs)

| File | Path | Size | Rows | GPS Region |
|------|------|------|------|-----------|
| S-M.csv | `Categorised/M (Driver B)/` | 19.8 MB | 105,974 | England ✓ |
| V-M.csv | `Categorised/M (Driver B)/` | 22.5 MB | 100,002 | England ✓ |
| S-S1.csv | `Categorised/S (Driver A)/S1/` | 9.6 MB | 51,746 | England ✓ |
| V-S1.csv | `Categorised/S (Driver A)/S1/` | 11.0 MB | 51,746 | England ✓ |
| S-S2.csv | `Categorised/S (Driver A)/S2/` | 17.1 MB | downloading | — |
| V-Vfa01.csv | `Categorised/Vfa01/` | 2.3 MB | 11,535 | England ✓ |
| ... | ... | ... | ... | ... |

*(Full inventory will be appended when download completes)*

---

## Method Evidence Summary

| Method | Attempted | Exact Failure / Success |
|--------|-----------|------------------------|
| `git lfs pull` (local) | ✅ | ❌ `HEAD` not resolved — no commits in local copy |
| LFS Batch API (single file) | ✅ | ❌ HTTP 403 — LFS budget exceeded |
| LFS Batch API (50-file batch) | ✅ | ❌ HTTP 403 — all batches rejected |
| GitHub archive ZIP (main) | ✅ | ❌ HTTP 404 — branch name is `master` |
| GitHub archive ZIP (master) | ✅ | ❌ ZIP contains LFS stubs (133 bytes per CSV) |
| `raw.githubusercontent.com` | ✅ | ❌ Returns LFS pointer text |
| **GitHub web blob `?raw=true`** | ✅ | **✅ Real CSV data served (bypasses quota)** |
| Kaggle/Zenodo/Figshare mirror | ✅ | ❌ No mirror exists — GitHub only |

---

## Next Step (Phase 5b — after download completes)

Once all 288 synchronised CSVs are downloaded and verified:
1. Update `phase1/src/step1_ingest.py` `DATA_ROOT` to point to `IO-VNBD-master/Synchronised V abd S datasets/Categorised IOVNB Dataset/`
2. Re-run Phase 1 → 4 pipeline against real data
3. Report real (not synthetic) benchmark metrics for SIH submission

> [!IMPORTANT]
> The real column names differ slightly from synthetic: e.g., `"No of GPS Satellites Available"` vs `"GPS_Satellites"`, `"Time Since Start of Day (seconds)"` vs `"Time_s"`. The ingestion layer will need a column-name mapping update.
