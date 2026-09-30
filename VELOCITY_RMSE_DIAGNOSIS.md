# Velocity Model RMSE Diagnostic Report

**System Component:** `velocity_net.onnx` / `step1_velocity_model.py`  
**Evaluation File:** IO-VNBD Dataset `Vfa01` (`S-Vfa01.csv` & `V-vfa01.csv`)  
**Auditor:** Senior ML & Validation Engineer  
**Date:** September 29, 2026  
**Diagnostic Result:** **NO CODE BUG FOUND — GENUINE MODEL UNDERPERFORMANCE / MEAN COLLAPSE**

---

## Executive Summary

An in-depth diagnostic investigation was performed to determine whether the high **11.02 m/s RMSE** observed on `velocity_net.onnx` against CAN-bus ground truth speed was caused by a unit mismatch bug ($\text{km/h}$ vs $\text{m/s}$), a timestamp alignment lag, or a genuine ML model generalization failure.

### Key Finding:
- **No unit scaling bug exists:** Model predictions and CAN ground-truth targets are both confirmed in $\text{m/s}$.
- **No timestamp alignment bug exists:** Aligning `S` and `V` timestamps via linear time interpolation yields **8.93 m/s RMSE** (vs **8.92 m/s** naive index match on `Vfa01`).
- **Root Cause Identified:** **Mean-Collapse Generalization Failure.** The ~17K parameter `VelocityNet` model predicts a narrow speed range ($\approx 9\text{–}12\text{ m/s}$) regardless of whether the vehicle is driving at $3\text{ m/s}$ (over-predicting) or $20\text{ m/s}$ (under-predicting). This confirms **Remediation Roadmap Item 3** (dataset expansion & model retraining).

---

## 1. Step-by-Step Diagnostic Verification

### Check 1: 20 Real Paired Samples (ONNX Output vs Ground Truth)
The table below shows 20 consecutive timestamps ($dt = 0.1\text{s}$) comparing `velocity_net.onnx` output to timestamp-interpolated CAN-bus ground truth speed:

| Index | Timestamp (s) | ONNX Prediction (m/s) | Naive CAN GT (m/s) | Aligned CAN GT (m/s) | Aligned CAN GT (km/h) | Ratio (GT/ONNX) |
|-------|---------------|----------------------|--------------------|----------------------|-----------------------|-----------------|
| 500   | 33676.236     | 12.57                | 19.91              | 19.42                | 69.91                 | 1.55            |
| 501   | 33676.336     | 12.25                | 19.68              | 19.37                | 69.75                 | 1.58            |
| 502   | 33676.436     | 11.14                | 19.75              | 19.38                | 69.77                 | 1.74            |
| 503   | 33676.536     | 11.53                | 19.77              | 19.40                | 69.83                 | 1.68            |
| 504   | 33676.637     | 12.09                | 19.86              | 19.38                | 69.76                 | 1.60            |
| 505   | 33676.736     | 10.26                | 19.66              | 19.36                | 69.69                 | 1.89            |
| 506   | 33676.836     | 10.96                | 19.68              | 19.36                | 69.69                 | 1.77            |
| 507   | 33676.936     | 11.76                | 19.73              | 19.36                | 69.68                 | 1.65            |
| 508   | 33677.036     | 12.13                | 19.66              | 19.34                | 69.61                 | 1.59            |
| 509   | 33677.136     | 10.36                | 19.69              | 19.34                | 69.62                 | 1.87            |
| 510   | 33677.236     | 11.34                | 19.69              | 19.37                | 69.75                 | 1.71            |
| 511   | 33677.336     | 12.24                | 19.61              | 19.36                | 69.70                 | 1.58            |
| 512   | 33677.436     | 12.25                | 19.66              | 19.35                | 69.67                 | 1.58            |
| 513   | 33677.537     | 11.22                | 19.57              | 19.40                | 69.84                 | 1.73            |
| 514   | 33677.636     | 12.38                | 19.60              | 19.41                | 69.87                 | 1.57            |
| 515   | 33677.736     | 10.31                | 19.63              | 19.42                | 69.93                 | 1.88            |
| 516   | 33677.836     | 10.03                | 19.61              | 19.41                | 69.89                 | 1.94            |
| 517   | 33677.936     | 11.96                | 19.66              | 19.47                | 70.08                 | 1.63            |
| 518   | 33678.036     | 10.76                | 19.57              | 19.50                | 70.19                 | 1.81            |
| 519   | 33678.136     | 11.56                | 19.60              | 19.51                | 70.23                 | 1.69            |

*Observation:* Ratios range from 1.55 to 1.94 (mean ~1.7x), inconsistent with a 3.6x unit mismatch bug.

---

### Check 2: Training Label Pipeline Verification
Code line inspection of [`dataset_builder.py`](file:///Users/vivekwarang/vehicle%20sih/phase2/src/dataset_builder.py#L57):
```python
# Line 57 of dataset_builder.py:
speed_s = v_df[LABEL_COL_V].iloc[:n].ffill().bfill().fillna(0.0) / 3.6
```
- Ground-truth CAN speeds (`Velocity_kmh`) are explicitly divided by $3.6$ to convert to $\text{m/s}$ prior to model training.
- The `VelocityNet` architecture outputs non-negative scalar values in $\text{m/s}$ via a `Softplus` head layer.
- **Verdict:** No training label conversion error.

---

### Check 3: Timestamp Alignment Verification
- `S-Vfa01` start time: **33626.236 s** (09:20:26.236 AM)
- `V-vfa01` start time: **33622.000 s** (09:20:22.000 AM)
- A **4.236-second (42-sample)** start time offset exists between streams.
- Re-evaluating ONNX inference with exact time-based linear interpolation yields:
  * **Naive Row Matching RMSE:** `8.92 m/s` (on `Vfa01`)
  * **Time-Aligned Interpolated RMSE:** `8.93 m/s` (on `Vfa01`)
- **Verdict:** Time alignment offset has minimal effect on RMSE during highway cruising.

---

### Check 4: Speed Regime Behavior (Low vs High Speed Response)
Comparing ONNX predictions against CAN GT across different speed regimes:
- **Low Speed Regime ($\text{CAN GT} < 5\text{ m/s}$):**
  * CAN GT: $3.28\text{ m/s} \text{ -- } 4.94\text{ m/s}$ ($\approx 12\text{--}18\text{ km/h}$)
  * ONNX Prediction: $9.39\text{ m/s} \text{ -- } 10.42\text{ m/s}$ ($\approx 34\text{--}37\text{ km/h}$) — **Over-predicts low speeds by ~2x**.
- **High Speed Regime ($\text{CAN GT} > 20\text{ m/s}$):**
  * CAN GT: $20.02\text{ m/s} \text{ -- } 20.46\text{ m/s}$ ($\approx 72\text{--}74\text{ km/h}$)
  * ONNX Prediction: $8.90\text{ m/s} \text{ -- } 12.56\text{ m/s}$ ($\approx 32\text{--}45\text{ km/h}$) — **Under-predicts high speeds by ~2x**.

---

## 2. Conclusion & Next Steps

1. **Conclusion:** The 11.02 m/s RMSE is **not** a software bug. It represents true model underperformance caused by mean-collapse on an unseen driver/vehicle partition (`Driver E`).
2. **Action on Outage Drift Numbers:** Because no code bug was found and the velocity RMSE did not materially change, the current honest drift numbers (**48.81% outage drift**, **48.03% 50m drift**) are retained without alteration.
3. **Validation of Remediation Roadmap:** This diagnostic confirms **Remediation Roadmap Item 3** (fine-tuning `velocity_net.onnx` on the expanded 15-driver IO-VNBD dataset partition) as the correct technical fix.
