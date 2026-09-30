# ISRO SIH Intelligent Dead Reckoning – Judge Demonstration Runbook

**Target System:** AI-ML Based Intelligent Dead Reckoning Navigation System  
**Target Organization:** ISRO (Smart India Hackathon)  
**Audience:** Technical Evaluators & Hackathon Judges  
**Server Command:** `python3 phase3/server.py`  
**Web Dashboard URL:** `http://localhost:8050/`

---

## Executive Summary & Honesty Guarantee

This runbook provides the step-by-step procedure for conducting a live technical demonstration of the **AI-ML Intelligent Dead Reckoning System** before ISRO judges.

Every visualization and metric displayed on the dashboard is directly driven by the **real IO-VNBD dataset pipeline** (`velocity_net.onnx`, 15-state ESEKF, gravity alignment, and state machine mode-switcher). Nothing in this interface is scripted, mocked, or hand-animated.

---

## 1. Environment Setup & Launch

### Step 1.1: Start Real Pipeline Server
Open a terminal in the project root directory and execute:
```bash
python3 phase3/server.py
```
*Expected Output:*
```text
INFO  Processing real pipeline for dataset key: Vfa01
INFO  Pipeline computation complete for Vfa01: 11486 ticks, Endpoint Error=325.12m (48.81% drift)
INFO  ===========================================================================
INFO  ISRO SIH Real Pipeline Web Dashboard Server Running at:
INFO    http://localhost:8050/
INFO    REPLAY SOURCE: IO-VNBD Vfa01 (Highway, 70 km/h)
INFO  ===========================================================================
```

### Step 1.2: Launch Web Dashboard
Open any modern browser (Chrome, Firefox, Safari) and navigate to:
```text
http://localhost:8050/
```

---

## 2. Step-by-Step Judge Demonstration Script

### Phase 1: Replay Mode Transparency & System Initialisation
1. **Point to Top Banner:** Highlight the prominent banner `[REPLAY MODE: IO-VNBD Vfa01 (Highway, 70 km/h)]`.  
   *Narration:* *"To ensure 100% evaluation honesty, we clearly label that this demo replays real recorded IO-VNBD drive logs (Driver E, highway scenario) sample-by-sample through our ONNX edge model and ESEKF engine."*
2. **Point to Calibration Panel:** Show the computed phone orientation: `Pitch: -0.01°`, `Roll: -0.00°`, Status `CALIBRATED (Gravity Vector Aligned)`.  
   *Narration:* *"Before navigation begins, our in-vehicle calibration module uses static gravity vectors to auto-align the smartphone body frame to the vehicle driving frame."*

### Phase 2: Normal Navigation (GNSS-Aided Integration)
1. **Point to Mode Badge:** Highlight the green badge `GNSS-AIDED-INS`.
2. **Observe Trajectory Map:** Show the blue line on the Leaflet map tracing CAN-bus ground-truth GPS coordinates.  
   *Narration:* *"While GNSS signals are healthy, the 15-state Error-State EKF continuously fuses smartphone IMU readings with GNSS positions to bound accelerometer and gyroscope bias."*

### Phase 3: GNSS Outage Simulation & AI-DR Takeover
1. **Transition Point:** Allow the scrubber to reach sample index **2000** ($t = 200.0\text{ s}$).
2. **Observe Mode Switch:** Watch the top mode badge instantly switch to **`DEAD-RECKONING (GNSS OUTAGE)`** with a pulsing red indicator.
3. **Observe Trajectory Divergence:** Point to the red dashed line appearing on the Leaflet map, departing from the blue ground-truth path.  
   *Narration:* *"At t=200s, GNSS signals are completely masked out. The automated mode-switching state machine detects signal loss within 100ms and transitions to pure AI Dead Reckoning. VelocityNet directly estimates forward velocity from 20-sample IMU windows."*

### Phase 4: Live Metric Observation & Auditing
During the outage window ($t = 200\text{s}$ to $230\text{s}$):
1. **Live Speed:** Point to the AI Estimated Speed panel ($11.4\text{ m/s}$ vs CAN GT $22.5\text{ m/s}$).
2. **Endpoint Position Error:** Point to the Live Drift panel showing position error increasing smoothly to **$52.44\text{ m}$** at $100\text{m}$ outage distance ($47.98\%$ drift).
3. **Confidence Score:** Note that confidence drops from $95\%$ to $80\%$ during pure DR.

### Phase 5: GNSS Reacquisition & Session Summary
1. **Reacquisition:** At sample index **2300** ($t = 230.0\text{ s}$), watch the mode switch back to `GNSS-AIDED-INS`.
2. **Verified Session Summary Panel:** Point to the final summary card:
   - **Outage Distance:** `666.05 m`
   - **Endpoint Error:** `325.12 m`
   - **Real Drift Percentage:** `48.81 %`
   - **Velocity Model RMSE:** `11.02 m/s`  
   *Narration:* *"All numbers shown on this dashboard match the exact empirical values documented in our audit report `SIH_COMPLIANCE_MATRIX.md`."*

---

## 3. Fallback Contingency Protocols

If any technical issue arises during live judging:

| Contingency Scenario | Primary Action | Fallback Demonstration |
|----------------------|----------------|-----------------------|
| **Port 8050 Unavailable / In Use** | Edit `phase3/server.py` `port = 8055` and restart. | Open `file:///.../phase3/dashboard/index.html` directly in browser. |
| **Python Server Crash** | Relaunch command `python3 phase3/server.py`. | Show offline generated trajectory plot [`phase1/outputs/reports/screening_trajectory_plot.png`](file:///Users/vivekwarang/vehicle%20sih/phase1/outputs/reports/screening_trajectory_plot.png). |
| **Android Device Disconnect** | Switch to Web Dashboard at `http://localhost:8050/`. | Both Web Dashboard and Android App execute identical pipeline contracts. |
