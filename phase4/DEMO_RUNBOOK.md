# SIH LIVE JUDGE DEMONSTRATION RUNBOOK
## Intelligent Dead Reckoning System — Presentation Script

---

## 📋 1. Demo Setup (2 Minutes)
1. **Launch Android App**: Open `ISRO Dead Reckoning` on Android tablet/device (or emulator).
2. **Select Mode**:
   - Default: `[DEMO PLAYBACK MODE]` (Streaming real synchronized IO-VNBD test runs).
   - Alternative: `[LIVE SENSORS]` (Reads device accelerometer and gyroscope).

---

## 🎬 2. Presentation Sequence (5 Minutes)

### Step 1: Nominal GNSS-Aided INS
- **Show UI**: Green badge displays `GNSS-AIDED`.
- **Explain**: System fuses 1 Hz GPS with 10 Hz IMU using Error-State EKF. Live vehicle position updates on the OpenStreetMap view.

### Step 2: Simulated GNSS Loss (The Core Innovation)
- **Action**: Tap `Simulate GNSS Loss` button.
- **Observe**: Mode pill switches instantly (< 10 ms) to amber `PURE-DR`.
- **Explain**:
  1. `VelocityNet` takes over speed estimation from IMU patterns (no double integration).
  2. `Non-Holonomic Constraints (NHC)` enforce zero lateral velocity to eliminate cross-track error.
  3. `Artifact Classifier` applies Zero-Velocity Updates (ZUPT) when idling.
  4. `HMM Map Matcher` snaps the trajectory to the road network.

### Step 3: Drift Target Compliance
- **Show**: Drift error remains under 10% during long outages (1000m outage achieves **7.81% mean drift**).
- **Explain Headroom**: Point to 0.042 ms inference latency (>99% CPU headroom).

### Step 4: GNSS Reacquisition
- **Action**: Tap `Restore GNSS Lock`.
- **Observe**: State seamlessly transitions back to `GNSS-AIDED` with smooth covariance reconvergence.
