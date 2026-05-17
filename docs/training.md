# Sprint C-Training reproducibility walkthrough

This page documents how to reproduce the Table 3-1 training run end-to-end: fitting BMA priors, severity-stratified isotonic calibrators, and split + Mondrian conformal residual sets across the seven training events specified in the OSF pre-registration (`helios-program/orchestration/osf_preregistration.template.md`, §5).

The training run produces four trained artifacts that the kill-gate hold-out evaluation will consume. The artifacts themselves live in the private `helios-fusion-internal/weights/` companion repo (per the hybrid-IP strategy in the master plan §6.6); the reproducible code paths that produce them live in this package's `helios_fusion.training` subpackage.

## What this sprint does NOT do

- It does **not** run hold-out evaluation on the 3-event hold-out (2022-01-20, 2023-02-17, 2024-05-11 Gannon). That's gated on OSF pre-registration filing per the kill-gate discipline.
- It does **not** modify the kill-gate runner (`helios-program/orchestration/kill_gate.py` remains a stub).
- It does **not** change pre-registered hyperparameters. The BMA weight-update formula (rolling 90-day skill, `hss_clipped` policy), the isotonic-on-Platt-rejection approach, the Mondrian per-Kp-stratum split, and the Donaldson 1975 HSS formula are all locked.

## Prerequisites

- Python 3.11 or 3.12.
- An isolated virtual environment in the worktree (`uv venv .venv` is the canonical pattern).
- `helios-fusion-engine` installed editable (`uv pip install -e '.[dev]'`).
- `helios-spaceweather-connectors` v0.2.0 from GitHub:
  ```
  uv pip install 'helios-spaceweather-connectors @ git+https://github.com/577Industries/helios-spaceweather-connectors.git@v0.2.0'
  ```
- `helios-provenance-spec` v0.1.0 from GitHub:
  ```
  uv pip install 'helios-provenance-spec @ git+https://github.com/577Industries/helios-provenance-spec.git@v0.1.0'
  ```

## Environment variables

- `NASA_EARTHDATA_USER` / `NASA_EARTHDATA_PASS` -- optional. Required for CDDIS GIM TEC fetches. **Not required for the SEP all-clear kill-gate (§2 Obj. 3).** If not set, the loader records the deferral in the per-event `data_gaps` map and continues.

## Where the data comes from

| Adapter | Status for Sprint C-Training |
|---|---|
| `SepScoreboardsAdapter` (ISWA) | Used. ISWA JSON deposits typically start ~2018 for most contributing models; pre-2018 events fall back to documented synthetic proxy streams. |
| `SwpcAdapter` | Used for Kp. Kyoto archive coverage is reliable for events back to ~1980; very-old events may still fall back to synthetic Kp profiles. |
| `GoesAdapter` | Available; not consumed in the current pipeline (proton-flux fusion path is a §2 Obj. 4 thing, not the SEP kill-gate). |
| `DscovrAdapter` | Available; not consumed in the current pipeline. |
| `CddisGimAdapter` | **Gated on Earthdata creds.** Not required for the SEP kill-gate; recorded as a deferral when creds are absent. |
| `DonkiAdapter` | Available; not consumed in the current pipeline. |

When an upstream source doesn't cover the window, the loader substitutes a deterministic synthetic-proxy stream (seeded from the event ID) and tags every row with `source = "synthetic_proxy"`. The per-event `data_gaps` map and the global manifest record every deferral so consumers can discount synthetic-proxy events as appropriate.

## Running the training pipeline

### Option A: end-to-end via the notebook (preferred)

```
cd ~/577i-Projects/.worktrees/helios-fusion-engine-training
source .venv/bin/activate
jupyter execute notebooks/02-train-on-table-3-1.ipynb
```

The notebook:

1. Lists the seven training events.
2. Loads each event's Scoreboard A and Kp data (with synthetic-proxy fallbacks recorded in `data_gaps`).
3. Fits BMA priors, the stratified isotonic calibrator, and both conformal regressors.
4. Reports per-stratum conformal half-widths at the locked alpha = 0.1.
5. Writes all artifacts + their sibling provenance records + the manifest to `helios-fusion-internal/weights/`.
6. Runs a synthetic-data sanity test (the kill-gate's pre-flight check).

### Option B: from Python

```python
from pathlib import Path

from helios_fusion.training import run_full_training
from helios_fusion.training.pipeline import persist_artifacts

# 1. Run the four fits in sequence.
artifacts = run_full_training(use_real_data=True, cadence_hours=1.0)

# 2. Persist to the private companion repo.
persist_artifacts(
    artifacts,
    weights_dir=Path.home() / "577i-Projects" / "helios-fusion-internal" / "weights",
    repo_root=Path.cwd(),
    osf_preregistration_url=None,  # populated by operator after OSF filing
)
```

### Option C: synthetic only (offline; fastest)

```python
artifacts = run_full_training(use_real_data=False, cadence_hours=4.0)
```

This skips the network and exercises the full pipeline against the documented synthetic-proxy streams. It is the kill-gate's pre-flight check; if this fails, the real-data run will also fail.

## Expected outputs in `helios-fusion-internal/weights/`

| File | Contents |
|---|---|
| `bma_priors_table_3_1.npz` | 7 named float arrays, one per training event |
| `bma_priors_table_3_1.index.json` | Sidecar mapping `event_id -> [model_id, ...]` (the column order of the .npz arrays) |
| `bma_priors_table_3_1.npz.provenance.json` | `HeliosTransformationRecord` for the BMA fit |
| `isotonic_calibrators_stratified.npz` | Per-stratum sample counts (numeric) |
| `isotonic_calibrators_stratified.state.json` | Full calibrator state-dict (`SeverityStratifiedCalibrator.from_state_dict` round-trips this) |
| `isotonic_calibrators_stratified.npz.provenance.json` | `HeliosTransformationRecord` for the calibration fit |
| `conformal_residuals_split.npz` | Marginal split-conformal residuals (one numeric array) |
| `conformal_residuals_split.schema.json` | Schema-version sidecar |
| `conformal_residuals_split.npz.provenance.json` | `HeliosTransformationRecord` for the split-conformal fit |
| `conformal_residuals_mondrian.npz` | Three numeric arrays (one residual array per stratum) |
| `conformal_residuals_mondrian.state.json` | Full per-stratum state-dict (round-trips via `MondrianConformalRegressor.from_state_dict`) |
| `conformal_residuals_mondrian.npz.provenance.json` | `HeliosTransformationRecord` for the Mondrian-conformal fit |
| `manifest.json` | Training-run metadata: date, git SHA, connectors version, provenance-spec version, per-event component models + data gaps, OSF pre-registration URL |

The `.npz` archives are populated only with numeric arrays. Structured metadata lives in sidecar `.json` files alongside, so consumers can load the archives safely without object-array deserialisation.

## Provenance-record shape

We use **`HeliosTransformationRecord`** (not `HeliosFusedOutputRecord` or `HeliosModelOutputRecord`) for each trained artifact, because the trained parameters are produced by transforming upstream Scoreboard / Kp data through the BMA / isotonic / conformal fitters -- exactly what `HeliosTransformationRecord` is for. The `type` discriminator (`"bma"` / `"calibration"` / `"conformal"`) matches the literal enumeration in `helios-provenance-spec` v0.1.

A separate "trained-parameter-record" model is not currently necessary; the operator can request one if downstream consumers need a distinct shape.

## Expected runtime

- Synthetic-only (cadence 4h): ~10 seconds.
- Real-data with full ISWA + SWPC pulls (cadence 1h): ~2-15 minutes depending on ISWA latency and the number of probed model directories.

## Reproducibility seed

All synthetic-proxy fallbacks are seeded from `_SYNTH_SEED = 20260517` (the spec date, YYYYMMDD) xor-ed with the event-ID hash. The seed is locked.

## Verification gate (per the spec)

The training run is considered successful when:

- All 7 training-event windows produce a non-empty dataframe.
- `bma_priors_table_3_1.npz` contains 7 named numeric entries, each summing to 1.0.
- `isotonic_calibrators_stratified.state.json` validates against the calibrator's schema and round-trips.
- Each persisted `.npz` has a sibling `.provenance.json` validating against `helios-provenance-spec` v0.1.
- The synthetic-data sanity test passes (`pytest tests/training/test_pipeline.py::test_synthetic_data_sanity_pipeline_convergence`).
- `helios-fusion-internal/weights/manifest.json` records the training-run metadata.
