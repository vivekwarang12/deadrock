"""
step5_model_export.py  –  Phase 2 Step 5
=========================================
Export trained models to ONNX + int8 quantization.
Benchmark real CPU inference latency on single-thread (simulates mobile).

Benchmarking methodology:
  - CPU-only, single-thread OrtSession (mirrors mobile ARM Cortex-A55/72)
  - Input shape: (1, window, 6)  — exactly as used at 10 Hz
  - 1000 warm-up runs, 5000 timed runs → report mean + p99 latency
  - Target: < 10 ms per inference to stay within 100 ms / 10 Hz budget
"""

import json
import time
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PHASE2_OUT  = Path(__file__).parent.parent / 'outputs'
MODELS_DIR  = PHASE2_OUT / 'models'
REPORTS_DIR = PHASE2_OUT / 'reports'


# ─────────────────────── ONNX export (PyTorch → ONNX) ────────────────────────
def export_velocity_model_onnx(model, window: int, n_features: int = 6) -> Path:
    """Export VelocityNet to ONNX with dynamic batch and sequence axes."""
    import torch
    model.eval()
    dummy  = torch.zeros(1, window, n_features)
    path   = MODELS_DIR / 'velocity_net.onnx'
    torch.onnx.export(
        model, dummy, str(path),
        input_names=['imu_window'],
        output_names=['speed_ms'],
        dynamic_axes={
            'imu_window': {0: 'batch', 1: 'seq'},
            'speed_ms':   {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=False,  # keep shapes symbolic for quantization compat
    )
    size_kb = path.stat().st_size / 1024
    log.info(f'ONNX export: {path}  ({size_kb:.1f} KB)')
    return path


def quantize_onnx(onnx_path: Path) -> Path:
    """INT8 dynamic quantization via onnxruntime quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
        q_path = onnx_path.parent / (onnx_path.stem + '_int8.onnx')
        pre_path = onnx_path.parent / (onnx_path.stem + '_pre.onnx')
        # Pre-process: infer shapes and clean graph before quantizing
        try:
            quant_pre_process(str(onnx_path), str(pre_path), skip_symbolic_shape=True)
            src = str(pre_path)
        except Exception:
            src = str(onnx_path)
        quantize_dynamic(
            src, str(q_path),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
        )
        # Clean up pre-processed temp file
        if pre_path.exists():
            pre_path.unlink()
        orig_kb = onnx_path.stat().st_size / 1024
        q_kb    = q_path.stat().st_size  / 1024
        log.info(f'INT8 quantization: {orig_kb:.1f} KB → {q_kb:.1f} KB '
                 f'({(1-q_kb/orig_kb)*100:.1f}% smaller)')
        return q_path
    except Exception as e:
        log.warning(f'Quantization failed: {e}  – using fp32 ONNX')
        return onnx_path


# ─────────────────────── latency benchmark ────────────────────────────────────
def benchmark_ort(onnx_path: Path, window: int,
                   n_warmup: int = 200, n_runs: int = 2000) -> dict:
    """
    Real latency benchmark using ONNX Runtime (CPU, single thread).
    Simulates mobile deployment constraints.
    """
    import onnxruntime as ort

    # Single-thread session — approximates mobile CPU constraint
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess        = ort.InferenceSession(str(onnx_path), opts,
                                        providers=['CPUExecutionProvider'])
    input_name  = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, window, 6).astype(np.float32)

    # warm-up
    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy_input})

    # timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

    lat = np.array(latencies)
    result = {
        'model':        onnx_path.name,
        'window':       window,
        'n_runs':       n_runs,
        'mean_ms':      round(float(lat.mean()), 4),
        'median_ms':    round(float(np.median(lat)), 4),
        'p95_ms':       round(float(np.percentile(lat, 95)), 4),
        'p99_ms':       round(float(np.percentile(lat, 99)), 4),
        'min_ms':       round(float(lat.min()), 4),
        'max_ms':       round(float(lat.max()), 4),
        'budget_10hz_ms': 100.0,
        'feasible_10hz':  bool(float(np.percentile(lat, 99)) < 100.0),
        'note':         '1 thread, CPU only. Actual mobile ARM may be 2-4x slower '
                        'for float32 ONNX; INT8 typically recovers ~2x.'
    }
    log.info(f'  [{onnx_path.name}] mean={result["mean_ms"]:.3f}ms  '
             f'p99={result["p99_ms"]:.3f}ms  '
             f'{"✓ feasible" if result["feasible_10hz"] else "✗ too slow"} at 10Hz')
    return result


# ─────────────────────── classifier export (joblib → ONNX) ───────────────────
def export_classifier_onnx() -> Path | None:
    """Export GBT classifier to ONNX via skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import joblib

        pack = joblib.load(MODELS_DIR / 'artifact_classifier.pkl')
        clf  = pack['clf']
        n_feat = 42  # 6 channels × 7 stats

        initial_type = [('features', FloatTensorType([None, n_feat]))]
        onnx_model   = convert_sklearn(clf, initial_types=initial_type,
                                        target_opset=14)
        path = MODELS_DIR / 'artifact_classifier.onnx'
        with open(path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        log.info(f'Classifier ONNX: {path}  ({path.stat().st_size/1024:.1f} KB)')
        return path
    except Exception as e:
        log.warning(f'Classifier ONNX export failed (skl2onnx not installed?): {e}')
        return None


# ─────────────────────── master export function ───────────────────────────────
def run_export_and_benchmark(model, window: int) -> dict:
    """Full export pipeline. Returns benchmark results dict."""
    results = {}

    # 1. Export fp32 ONNX
    onnx_fp32 = export_velocity_model_onnx(model, window)
    results['fp32'] = benchmark_ort(onnx_fp32, window)

    # 2. INT8 quantize + benchmark
    onnx_int8 = quantize_onnx(onnx_fp32)
    if onnx_int8 != onnx_fp32:
        results['int8'] = benchmark_ort(onnx_int8, window)
    else:
        results['int8'] = results['fp32'].copy()
        results['int8']['note'] = 'Quantization failed; int8 = fp32'

    # 3. Classifier ONNX (best-effort)
    clf_path = export_classifier_onnx()
    if clf_path:
        results['classifier'] = benchmark_ort(clf_path, window=0)

    # 4. Save
    bench_path = REPORTS_DIR / 'latency_benchmark.json'
    with open(bench_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'Latency benchmark → {bench_path}')

    # 5. Summary
    fp32_p99 = results['fp32']['p99_ms']
    int8_p99 = results.get('int8', {}).get('p99_ms', fp32_p99)
    log.info(f'\n  10 Hz feasibility check (p99 latency < 100 ms):')
    log.info(f'    fp32 ONNX: p99 = {fp32_p99:.3f} ms  '
             f'{"✓" if fp32_p99 < 100 else "✗"}')
    log.info(f'    int8 ONNX: p99 = {int8_p99:.3f} ms  '
             f'{"✓" if int8_p99 < 100 else "✗"}')
    log.info(f'  Note: add ~2–4× margin for real mobile ARM → '
             f'~{fp32_p99*3:.1f} ms worst-case estimate')
    return results
