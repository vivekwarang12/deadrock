# REPRODUCIBILITY GUIDE
## ISRO SIH Dead Reckoning Benchmark

To reproduce every metric, model, table, and plot from scratch:

```bash
# 1. Run Unit Tests (Rate-independence, mode switcher, EKF)
python3 -m unittest discover -s phase3/tests

# 2. Run Complete Phase 3 Master Pipeline
python3 phase3/src/run_phase3.py
```

All plots will be written to `phase3/outputs/plots/` and all benchmark numbers to `phase3/FINAL_RESULTS.md`.
