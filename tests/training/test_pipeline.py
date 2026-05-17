"""Tests for the end-to-end pipeline + persistence + provenance (Sprint C-Training-v2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helios_fusion.training.pipeline import (
    TrainingArtifacts,
    persist_artifacts,
    run_full_training,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "swpc_archive"
_FULL_HTML = (_FIXTURE_DIR / "seps_full_2026-05-17.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def artifacts() -> TrainingArtifacts:
    """Run the full pipeline against the synthetic fixture + SWPC fixture."""
    return run_full_training(
        use_real_data=False,
        cadence_hours=4.0,
        use_swpc_archive_truth=True,
        swpc_archive_html=_FULL_HTML,
    )


def test_pipeline_produces_seven_event_priors(artifacts: TrainingArtifacts) -> None:
    assert len(artifacts.bma_priors) == 7
    assert artifacts.calibrator.fitted
    assert artifacts.split_conformal.fitted
    assert artifacts.mondrian_conformal.fitted


def test_pipeline_uses_swpc_archive_truth(artifacts: TrainingArtifacts) -> None:
    """v2: every event frame should report truth_source='swpc_archive'."""
    for f in artifacts.frames:
        assert f.truth_source == "swpc_archive", (
            f"event {f.event.event_id} got truth_source={f.truth_source!r}; expected swpc_archive"
        )


def test_persist_creates_six_files(tmp_path: Path, artifacts: TrainingArtifacts) -> None:
    written = persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    assert "bma_priors" in written
    assert "isotonic_calibrators" in written
    assert "conformal_split" in written
    assert "conformal_mondrian" in written
    assert "manifest" in written
    for key in ("bma_priors", "isotonic_calibrators", "conformal_split", "conformal_mondrian"):
        npz_path = written[key]
        prov_path = npz_path.with_suffix(".npz.provenance.json")
        assert prov_path.exists(), f"missing provenance for {key}"


def test_persisted_provenance_validates(tmp_path: Path, artifacts: TrainingArtifacts) -> None:
    """Sibling provenance files must validate against HeliosTransformationRecord."""
    from helios_provenance.models import HeliosTransformationRecord

    written = persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    for key in ("bma_priors", "isotonic_calibrators", "conformal_split", "conformal_mondrian"):
        prov_path = written[key].with_suffix(".npz.provenance.json")
        payload = json.loads(prov_path.read_text())
        record = HeliosTransformationRecord.model_validate(payload)
        assert record.record_type == "HeliosTransformationRecord"
        assert record.type in {"bma", "calibration", "conformal"}


def test_manifest_uses_training_runs_array(tmp_path: Path, artifacts: TrainingArtifacts) -> None:
    """v2: manifest.json schema is ``{"training_runs": [run_entry, ...]}``."""
    written = persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest = json.loads(written["manifest"].read_text())
    assert "training_runs" in manifest
    assert isinstance(manifest["training_runs"], list)
    assert len(manifest["training_runs"]) == 1
    run = manifest["training_runs"][0]
    assert len(run["training_events"]) == 7
    for ev in run["training_events"]:
        assert "component_models" in ev
        assert "data_gaps" in ev
        assert "source_row_counts" in ev
        assert "component_source_labels" in ev


def test_manifest_preserves_v1_entry_when_present(
    tmp_path: Path, artifacts: TrainingArtifacts
) -> None:
    """If a v1 single-run manifest already exists, persist should preserve
    it as the first entry of the training_runs array."""
    # Seed a fake v1 single-run manifest at the target path.
    v1_manifest = {
        "training_run_id": "helios-fusion-engine/training-run/20260517T205638Z",
        "training_run_date_utc": "2026-05-17T20:56:38.162743+00:00",
        "fusion_engine_version": "0.1.0",
        "fusion_engine_git_sha": "b749d10",
        "connectors_version": "0.2.0",
        "provenance_spec_version": "0.1.0",
        "training_events": [],
        "artifacts": {},
        "osf_preregistration_url": None,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(v1_manifest, indent=2))

    persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    merged = json.loads((tmp_path / "manifest.json").read_text())
    assert "training_runs" in merged
    assert len(merged["training_runs"]) == 2
    # v1 entry preserved as first.
    assert merged["training_runs"][0]["fusion_engine_version"] == "0.1.0"
    assert merged["training_runs"][0]["connectors_version"] == "0.2.0"
    # v2 entry second.
    assert merged["training_runs"][1]["fusion_engine_version"] != "0.1.0"


def test_bma_priors_npz_round_trips_per_event(tmp_path: Path, artifacts: TrainingArtifacts) -> None:
    """The persisted .npz must contain one numeric array per training event."""
    written = persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    loaded = np.load(written["bma_priors"])
    event_ids = list(loaded.files)
    assert len(event_ids) == 7
    for event_id in event_ids:
        arr = loaded[event_id]
        assert abs(float(arr.sum()) - 1.0) < 1e-9


def test_bma_weights_diverge_from_uniform_with_heterogeneous_streams(
    artifacts: TrainingArtifacts,
) -> None:
    """v2 spec: with heterogeneous component bias archetypes + real
    SWPC archive truth labels, the BMA weights should NOT be exactly
    uniform.

    The v1 closed-loop synthetic-truth fit produced near-uniform weights
    (~0.09-0.10 across 11 components); v2's supervised fit against real
    onset labels should show meaningful divergence in at least one event.

    With 37 components and the HSS-clipped weight policy with an epsilon
    floor, "divergence from uniform" is naturally compressed compared to
    v1's 11-component setup. We check: in at least one event the top
    weight should exceed uniform by ≥10% (i.e. top > 1.10/n).
    """
    diverged_events = []
    for event_id, weights in artifacts.bma_priors.items():
        n = len(weights)
        top = max(weights.values())
        if top > 1.10 / n:
            diverged_events.append(event_id)
    assert diverged_events, (
        "no event showed BMA-weight divergence ≥10% above uniform; "
        "v2 should produce non-uniform weights via SWPC supervised fit"
    )


def test_synthetic_data_sanity_pipeline_convergence(artifacts: TrainingArtifacts) -> None:
    """Sanity test: the synthetic-streams pipeline converges.

    With v2's SWPC archive truth, the well-calibrated archetype should
    still not be systematically downweighted vs the biased archetypes.
    """
    mean_weight: dict[str, float] = {}
    for weights in artifacts.bma_priors.values():
        for model_id, w in weights.items():
            mean_weight.setdefault(model_id, 0.0)
            mean_weight[model_id] += w / len(artifacts.bma_priors)

    well_cal_avg: list[float] = []
    biased_avg: list[float] = []
    for model_id, w in mean_weight.items():
        bias_key = (hash(model_id) & 0xFF) % 3
        if bias_key == 0:
            well_cal_avg.append(w)
        else:
            biased_avg.append(w)

    if well_cal_avg and biased_avg:
        avg_well = sum(well_cal_avg) / len(well_cal_avg)
        avg_biased = sum(biased_avg) / len(biased_avg)
        # Looser bound than v1 — v2 truth labels are sparser so individual
        # event weights can be more variable.
        assert avg_well >= 0.5 * avg_biased, (
            f"well-calibrated avg weight {avg_well:.4f} unexpectedly far below "
            f"biased avg weight {avg_biased:.4f}"
        )
