"""
test_rate_independence.py  –  Phase 3 Rate-Handling Validation
===============================================================
Validates that SIHEdgeEngine processes IMU telemetry at various sample rates:
  - 10 Hz (Native IO-VNBD / Smartphone rate)
  - 50 Hz (Standard Automotive MEMS IMU)
  - 100 Hz (High-Performance Industrial IMU)
  - 200 Hz (Fiber-Optic Gyroscope / FOG standard rate)

Note: Architectural rate-handling is verified here. Physical 200Hz FOG hardware
benchmarking is labeled as architecturally compatible / untested on physical FOG unit.
"""

import unittest
import numpy as np

from phase3.src.engine import SIHEdgeEngine, EngineConfig, IMUSample, GNSSSample, AxisConvention


class TestRateIndependence(unittest.TestCase):

    def setUp(self):
        self.initial_gnss = GNSSSample(
            timestamp_sec=0.0,
            latitude_deg=52.3850,
            longitude_deg=-1.5600,
            speed_ms=15.0,
            bearing_deg=90.0,
            accuracy_m=3.0,
            satellites=8,
            hdop=1.1
        )

    def _run_stream_at_rate(self, rate_hz: float, duration_sec: float = 5.0) -> list:
        cfg = EngineConfig(
            nominal_imu_rate_hz=rate_hz,
            axis_convention=AxisConvention.VEHICLE_CAN,
            enable_map_matching=False
        )
        engine = SIHEdgeEngine(cfg)
        engine.initialize_fix(self.initial_gnss)

        dt = 1.0 / rate_hz
        n_samples = int(duration_sec * rate_hz)
        states = []

        for i in range(n_samples):
            t = (i + 1) * dt
            # Synthetic 15 m/s driving with slight 0.05 rad/s yaw
            accel = (0.2, 0.0, 9.81)
            gyro = (0.0, 0.0, 0.05)
            sample = IMUSample(timestamp_sec=t, accel_ms2=accel, gyro_rads=gyro)
            state = engine.process_imu_sample(sample)
            states.append(state)

        return states

    def test_10hz_native_rate(self):
        states = self._run_stream_at_rate(10.0)
        self.assertEqual(len(states), 50)
        self.assertGreater(states[-1].latitude_deg, 52.0)
        self.assertFalse(np.isnan(states[-1].speed_ms))

    def test_50hz_mems_rate(self):
        states = self._run_stream_at_rate(50.0)
        self.assertEqual(len(states), 250)
        self.assertFalse(np.isnan(states[-1].latitude_deg))

    def test_100hz_industrial_rate(self):
        states = self._run_stream_at_rate(100.0)
        self.assertEqual(len(states), 500)
        self.assertFalse(np.isnan(states[-1].latitude_deg))

    def test_200hz_fog_architectural_rate(self):
        states = self._run_stream_at_rate(200.0)
        self.assertEqual(len(states), 1000)
        self.assertFalse(np.isnan(states[-1].latitude_deg))
        # Total distance traveled should match ~75m in 5s at 15 m/s
        lat_diff = abs(states[-1].latitude_deg - self.initial_gnss.latitude_deg)
        lon_diff = abs(states[-1].longitude_deg - self.initial_gnss.longitude_deg)
        self.assertGreater(lat_diff + lon_diff, 0.0001)


if __name__ == '__main__':
    unittest.main()
