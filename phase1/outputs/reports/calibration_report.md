# Phase 1 Calibration Report

> **Source**: Computed from real stationary segments of the IO-VNBD dataset (or schema-identical synthetic data). NOT placeholder values.

**Runs used**: Vta1a, Vta1a, Vta1a

---

## Aggregate (mean across all calibration runs)

### Accelerometer Bias (m/s²)

| Axis | Bias (m/s²) | Noise Std (m/s²) |
|------|------------|-----------------|
| X    | +0.099457    | 0.224381         |
| Y    | -0.252874    | 0.304735         |
| Z    | +0.035513    | 0.078819         |

### Gyroscope Bias (rad/s)

| Axis  | Bias (rad/s) | Noise Std (rad/s) |
|-------|-------------|------------------|
| Roll  | +nan  | nan        |
| Pitch | +nan  | nan        |
| Yaw   | +nan  | nan        |

---

## Per-Run Detail

### Run: Vta1a  (n=167 samples, 16.7 s)

**Accel bias** (m/s²):  X=+0.099457  Y=-0.252874  Z=+0.035513

**Accel noise σ** (m/s²):  X=0.224381  Y=0.304735  Z=0.078819

**Gyro bias** (rad/s):  Roll=+0.00034850  Pitch=-0.00119880  Yaw=-0.00084910

**Gyro noise σ** (rad/s):  Roll=0.01319338  Pitch=0.01974742  Yaw=0.01056067

### Run: Vta1a  (n=167 samples, 16.7 s)

**Accel bias** (m/s²):  X=+0.099457  Y=-0.252874  Z=+0.035513

**Accel noise σ** (m/s²):  X=0.224381  Y=0.304735  Z=0.078819

**Gyro bias** (rad/s):  Roll=+nan  Pitch=+nan  Yaw=+nan

**Gyro noise σ** (rad/s):  Roll=nan  Pitch=nan  Yaw=nan

### Run: Vta1a  (n=167 samples, 16.7 s)

**Accel bias** (m/s²):  X=+0.099457  Y=-0.252874  Z=+0.035513

**Accel noise σ** (m/s²):  X=0.224381  Y=0.304735  Z=0.078819

**Gyro bias** (rad/s):  Roll=+0.00034850  Pitch=-0.00119880  Yaw=-0.00084910

**Gyro noise σ** (rad/s):  Roll=0.01319338  Pitch=0.01974742  Yaw=0.01056067

---

## Notes

- Accel bias = mean(measured_accel − gravity_vector) during stationary.
- Gyro bias = mean(measured_gyro) during stationary (should be 0 dps).
- These values must be **subtracted** before strapdown integration.
- Allan deviation needs ≥120 s of contiguous data; skipped otherwise.
- Do NOT generalize these parameters across different phone models.
  Re-run calibration per device as new runs are added.

*Phase 1 baseline (unoptimized)*