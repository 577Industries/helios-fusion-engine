"""Tests for the end-to-end pipeline + persistence + provenance."""

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


@pytest.fixture(scope="module")
def artifacts() -> TrainingArtifacts:
    """Run the full pipeline against the synthetic fixture."""
    return run_full_training(use_real_data=False, cadence_hours=4.0)


def test_pipeline_produces_seven_event_priors(artifacts: TrainingArtifacts) -> None:
    assert len(artifacts.bma_priors) == 7
    assert artifacts.calibrator.fitted
    assert artifacts.split_conformal.fitted
    assert artifacts.mondrian_conformal.fitted


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
    # Each npz has a sibling provenance.json
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


def test_manifest_records_event_metadata(tmp_path: Path, artifacts: TrainingArtifacts) -> None:
    written = persist_artifacts(
        artifacts,
        weights_dir=tmp_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest = json.loads(written["manifest"].read_text())
    assert manifest["training_events"]
    assert len(manifest["training_events"]) == 7
    assert manifest["osf_preregistration_url"] is None  # left null until operator files
    for event in manifest["training_events"]:
        assert "component_models" in event
        assert "data_gaps" in event


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


def test_synthetic_data_sanity_pipeline_convergence(artifacts: TrainingArtifacts) -> None:
    """Sanity test: the synthetic-streams pipeline converges.

    The bias-key hash in :func:`_synthesize_proxy_stream` deterministically
    assigns each model_id to one of three bias archetypes. We compute the
    mean weight per model across all seven events and assert the
    well-calibrated archetype is not systematically downweighted relative
    to the biased archetypes.
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
        assert avg_well >= 0.75 * avg_biased, (
            f"well-calibrated avg weight {avg_well:.4f} unexpectedly far below "
            f"biased avg weight {avg_biased:.4f}"
        )
