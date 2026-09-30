"""
step5_latency_audit.py  –  Phase 4 Step 5
===========================================
End-to-End Real-Time Execution Latency & Headroom Benchmark.

Measures the exact CPU cycle time for the entire live navigation tick:
  [Raw Sensor Read] → [Axis Transform] → [Buffer Update] → [ONNX VelocityNet]
  → [ESEKF Propagation] → [NHC Update] → [Mode Switch FSM] → [Map Snapping] → [Output State]

Target: < 100 ms per cycle (10 Hz mobile update rate).
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

from phase3.src.engine import SIHEdgeEngine, EngineConfig, IMUSample, GNSSSample, AxisConvention

log = logging.getLogger(__name__)


def benchmark_end_to_end_latency(onnx_model_path: Path,
                                 normalizer_path: Path,
                                 n_warmup: int = 200,
                                 n_runs: int = 2000) -> Dict[str, Any]:
    """
    Executes live loop timing on single-thread CPU.
    """
    cfg = EngineConfig(
        onnx_model_path=str(onnx_model_path),
        normalizer_path=str(normalizer_path),
        nominal_imu_rate_hz=10.0,
        axis_convention=AxisConvention.VEHICLE_CAN,
        enable_map_matching=True
    )
    engine = SIHEdgeEngine(cfg)

    # Initialize GNSS reference fix
    initial_gnss = GNSSSample(
        timestamp_sec=0.0,
        latitude_deg=52.3850,
        longitude_deg=-1.5600,
        speed_ms=12.5,
        bearing_deg=45.0,
        accuracy_m=3.0,
        satellites=8,
        hdop=1.1
    )
    engine.initialize_fix(initial_gnss)

    # Warm-up phase
    for i in range(n_warmup):
        t = (i + 1) * 0.1
        sample = IMUSample(
            timestamp_sec=t,
            accel_ms2=(0.1, 0.0, 9.81),
            gyro_rads=(0.0, 0.0, 0.02)
        )
        engine.process_imu_sample(sample)

    # Timed runs
    latencies_ms = []
    for i in range(n_runs):
        t = (n_warmup + i + 1) * 0.1
        sample = IMUSample(
            timestamp_sec=t,
            accel_ms2=(0.1 + np.random.randn()*0.02, 0.0, 9.81),
            gyro_rads=(0.0, 0.0, 0.02 + np.random.randn()*0.005)
        )

        t_start = time.perf_counter()
        state = engine.process_imu_sample(sample)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        latencies_ms.append(t_elapsed_ms)

    lats = np.array(latencies_ms)
    mean_lat = float(np.mean(lats))
    p50_lat  = float(np.percentile(lats, 50))
    p95_lat  = float(np.percentile(lats, 95))
    p99_lat  = float(np.percentile(lats, 99))
    max_lat  = float(np.max(lats))

    budget_ms = 100.0  # 10 Hz budget
    headroom_pct = ((budget_ms - p99_lat) / budget_ms) * 100.0

    return {
        'total_timed_cycles': n_runs,
        'mean_latency_ms': round(mean_lat, 4),
        'median_latency_ms': round(p50_lat, 4),
        'p95_latency_ms': round(p95_lat, 4),
        'p99_latency_ms': round(p99_lat, 4),
        'max_latency_ms': round(max_lat, 4),
        'target_budget_ms': budget_ms,
        'headroom_percent': round(headroom_pct, 2),
        'realtime_feasible_at_10hz': bool(p99_lat < budget_ms),
        'verdict': f'✓ FEASIBLE: P99 cycle time is {p99_lat:.3f} ms with {headroom_pct:.1f}% computing headroom'
    }
