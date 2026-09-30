# Train / Validation / Test Split — IO-VNBD Phase 1

## Split Strategy: By Driver (not by row)

Splitting by **driver identity** prevents data leakage: a model trained on
rows from the same driver but different time windows would learn driver-specific
motion artifacts rather than generalizable navigation patterns.

---

## Split Assignments

| Driver | ID | Driving Style | Split | Rationale |
|--------|----|--------------|-------|-----------|
| A      | S1–S4 | Defensive | **Train** | Largest labelled set; urban + motorway |
| B      | M      | Defensive | **Train** | Urban; good roundabout coverage |
| D      | Y1     | Defensive | **Train** | Short urban; good for calibration diversity |
| C      | (unsync) | Defensive | **Val** | Held-out driver; no sync V-file pair required |
| E      | Vfa, Vtb, Vw | Aggressive | **Test** | Only aggressive driver; hardest case |
| E (Vta1a) | Stationary | — | **Calibration** | >20 min no-motion; bias estimation only |

---

## Scenario Coverage

| Scenario | Train | Val | Test |
|----------|-------|-----|------|
| Urban / low-speed | ✓ S1, S3, M, Y1 | ✓ S3b | ✓ Vtb, Vw |
| Motorway / high-speed | ✓ S2 | — | ✓ Vfa01, Vfa02 |
| Roundabouts | ✓ M, S3 | ✓ | ✓ Vw |
| Hard braking | ✓ S4 | ✓ | ✓ Vtb |
| Stationary (calibration) | ✓ Vta1a | — | — |

Both motorway and urban scenarios appear in **all three splits** ✓

---

## Rationale for Choosing Driver E as Test

1. **Aggressive driving style** is the hardest case for dead reckoning
   (high lateral accelerations, rapid yaw changes, higher speeds).
2. Driver E has the **largest number of synchronized V-/S- runs** in the
   dataset, providing many independent test windows.
3. E's runs cover both motorway (Vfa) and urban (Vtb, Vw) — essential for
   testing generalisation across road types.

---

## Leakage Prevention

- No rows from a test-split driver appear in train or val.
- Calibration runs (Vta1a stationary) use E's phone but are excluded from
  DR evaluation (used only for bias estimation).
- The `split` field in `dataset_manifest.json` enforces these boundaries
  programmatically.

---

*Phase 1 baseline (unoptimized)*
