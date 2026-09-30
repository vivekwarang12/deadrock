"""
config.py  –  SIH Edge Engine Configuration
=============================================
Configuration schema for device-agnostic navigation engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from .types import AxisConvention


@dataclass
class EngineConfig:
    # Model Paths
    onnx_model_path: Optional[str] = None
    normalizer_path: Optional[str] = None
    classifier_path: Optional[str] = None

    # Sensor & Timing Properties
    nominal_imu_rate_hz: float = 10.0   # Supports 10Hz, 50Hz, 100Hz, 200Hz
    window_duration_sec: float = 3.0    # 3.0s window for velocity inference
    axis_convention: AxisConvention = AxisConvention.VEHICLE_CAN

    # Filter Tunings
    pos_noise_m: float = 3.0
    vel_noise_ms: float = 0.8
    nhc_noise_ms: float = 0.15
    gyro_noise_rads: float = 0.02
    zupt_noise_ms: float = 0.05

    # Mode Switcher Thresholds
    max_gps_accuracy_m: float = 10.0
    min_satellites: int = 4
    max_hdop: float = 3.0
    hysteresis_count: int = 3
    gnss_timeout_sec: float = 1.1

    # Map Matching
    enable_map_matching: bool = True
    map_matching_sigma_z: float = 8.0
    map_matching_beta: float = 3.0

    @property
    def window_samples(self) -> int:
        return max(int(self.nominal_imu_rate_hz * self.window_duration_sec), 5)
