"""
step3_mode_switcher.py  –  Phase 3 Step 3
===========================================
Seamless GNSS / Dead-Reckoning Mode-Switching State Machine.

Requirements:
  - Sub-millisecond transition latency upon GNSS loss.
  - Multi-criterion signal quality monitoring:
      * GPS fix status (valid coordinates)
      * HDOP / Estimated Horizontal Accuracy (threshold: 10.0m)
      * Satellite Count (threshold: 4 satellites)
      * EKF Innovation Vector anomaly detection
  - Hysteresis counter to prevent chattering / rapid mode oscillations.
  - Live transition latency logging and timing measurements.
"""

import time
import logging
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class NavigationMode(str, Enum):
    GNSS_AIDED_INS = "GNSS-AIDED-INS"
    DEGRADED_GNSS = "DEGRADED-GNSS"
    PURE_INERTIAL_DR = "PURE-INERTIAL-DR"


class ModeSwitcherConfig:
    def __init__(self,
                 max_accuracy_m: float = 10.0,
                 min_satellites: int = 4,
                 max_hdop: float = 3.0,
                 hysteresis_count: int = 3,
                 timeout_seconds: float = 1.1):
        self.max_accuracy_m = max_accuracy_m
        self.min_satellites = min_satellites
        self.max_hdop = max_hdop
        self.hysteresis_count = hysteresis_count
        self.timeout_seconds = timeout_seconds


class ModeSwitcher:
    """
    Finite State Machine with Hysteresis for Seamless GNSS / DR Mode Switching.
    """
    def __init__(self, config: Optional[ModeSwitcherConfig] = None):
        self.cfg = config or ModeSwitcherConfig()
        self.current_mode = NavigationMode.GNSS_AIDED_INS
        self.last_valid_gnss_time: float = 0.0

        # Hysteresis counters
        self.good_gnss_streak = 0
        self.bad_gnss_streak = 0

        # Latency tracking
        self.transition_events: List[Dict[str, Any]] = []

    def evaluate_sample(self,
                        t_sec: float,
                        has_gnss: bool,
                        accuracy_m: Optional[float] = None,
                        sat_count: Optional[int] = None,
                        hdop: Optional[float] = None,
                        innovation_m: Optional[float] = None) -> NavigationMode:
        """
        Evaluate current sensor telemetry and transition states if necessary.
        Returns the active NavigationMode.
        """
        is_healthy = True

        if not has_gnss:
            is_healthy = False
        else:
            if accuracy_m is not None and (np.isnan(accuracy_m) or accuracy_m > self.cfg.max_accuracy_m):
                is_healthy = False
            if sat_count is not None and sat_count < self.cfg.min_satellites:
                is_healthy = False
            if hdop is not None and hdop > self.cfg.max_hdop:
                is_healthy = False
            if innovation_m is not None and innovation_m > 30.0:  # EKF divergence check
                is_healthy = False

        prev_mode = self.current_mode

        if is_healthy:
            self.good_gnss_streak += 1
            self.bad_gnss_streak = 0
            self.last_valid_gnss_time = t_sec

            if self.good_gnss_streak >= self.cfg.hysteresis_count:
                self.current_mode = NavigationMode.GNSS_AIDED_INS
        else:
            self.bad_gnss_streak += 1
            self.good_gnss_streak = 0

            # Immediate switch to DR on total signal loss; degraded if only marginal accuracy drop
            if not has_gnss or (t_sec - self.last_valid_gnss_time) > self.cfg.timeout_seconds:
                self.current_mode = NavigationMode.PURE_INERTIAL_DR
            elif self.bad_gnss_streak >= self.cfg.hysteresis_count:
                self.current_mode = NavigationMode.DEGRADED_GNSS

        # Log transition event
        if self.current_mode != prev_mode:
            self.transition_events.append({
                'time_sec': t_sec,
                'from_mode': prev_mode.value,
                'to_mode': self.current_mode.value,
                'trigger_reason': 'signal_loss' if not has_gnss else 'quality_drop'
            })

        return self.current_mode


def measure_mode_switch_latency(gps_series: pd.Series,
                                outage_start_idx: int,
                                sample_dt: float = 0.1) -> Dict[str, float]:
    """
    Measures the exact latency between simulated GPS loss and state-machine transition.
    """
    switcher = ModeSwitcher()
    t_start = outage_start_idx * sample_dt

    # Simulate sequence up to and past the outage
    switch_idx = -1
    for i in range(max(0, outage_start_idx - 10), outage_start_idx + 20):
        t = i * sample_dt
        has_gnss = (i < outage_start_idx)
        acc = 3.0 if has_gnss else None

        t0_perf = time.perf_counter()
        mode = switcher.evaluate_sample(t, has_gnss=has_gnss, accuracy_m=acc)
        eval_us = (time.perf_counter() - t0_perf) * 1e6

        if i >= outage_start_idx and mode == NavigationMode.PURE_INERTIAL_DR and switch_idx < 0:
            switch_idx = i
            break

    delay_samples = max(0, switch_idx - outage_start_idx)
    delay_ms = delay_samples * sample_dt * 1000.0

    return {
        'outage_start_sample': outage_start_idx,
        'switched_sample': switch_idx,
        'latency_samples': delay_samples,
        'latency_ms': delay_ms,
        'state_eval_time_us': eval_us
    }
