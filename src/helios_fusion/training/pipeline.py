"""End-to-end Sprint C-Training pipeline orchestrator.

This module ties together the four fits (BMA -> isotonic -> split-conformal
-> Mondrian-conformal) into a single :func:`run_full_training` entry point.
It also exposes :func:`persist_artifacts` which writes the fitted artifacts
to disk as ``.npz`` archives paired with sibling ``.provenance.json`` files
that validate against ``helios-provenance-spec`` v0.1
(:class:`helios_provenance.models.HeliosTransformationRecord`).

Provenance-record shape decision
--------------------------------

We use :class:`HeliosTransformationRecord` for each trained artifact:

* ``record_type`` is "HeliosTransformationRecord"
* ``type`` is one of ``"bma"`` / ``"calibration"`` / ``"conformal"`` --
  matching the literal enumeration in the schema.
* ``parameters`` carries the fitted hyperparameters and training config
  (event list, fit policy, threshold, alpha-default).
* ``code_ref`` references the fusion-engine git SHA + module path.
* ``input_refs`` reference the seven training-event IDs (acting as
  provenance pointers to upstream data).
* ``output_refs`` reference the ``.npz`` filename.

This is the right shape because trained weights and calibrators ARE
transformations of the upstream Scoreboard / Kp data into a calibrated
model -- exactly what ``HeliosTransformationRecord`` is for. A separate
"trained-parameter-record" model would be redundant; the operator can
review and confirm or request a model-spec addition later.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic

from helios_fusion.training.fit_bma import fit_bma_priors
from helios_fusion.training.fit_conformal import fit_mondrian_conformal, fit_split_conformal
from helios_fusion.training.fit_isotonic import fit_stratified_calibrators
from helios_fusion.training.load_table_3_1 import load_all_training_events

if TYPE_CHECKING:
    from collections.abc import Iterable

    from helios_fusion.calibration.stratified import SeverityStratifiedCalibrator
    from helios_fusion.conformal.mondrian import MondrianConformalRegressor
    from helios_fusion.conformal.split import SplitConformalRegressor
    from helios_fusion.training.load_table_3_1 import TrainingEventFrame

logger = logging.getLogger(__name__)

#: Locked alpha for the kill-gate (90% intervals).
DEFAULT_ALPHA: float = 0.1


@dataclass(slots=True)
class TrainingArtifacts:
    """Bundle of fitted artifacts produced by :func:`run_full_training`.

    Attributes:
        frames: The seven loaded :class:`TrainingEventFrame`.
        bma_priors: ``{event_id: {model_id: weight}}``.
        calibrator: Fitted stratified isotonic calibrator.
        split_conformal: Fitted split conformal regressor.
        mondrian_conformal: Fitted Mondrian conformal regressor.
        data_gaps: Union of per-event ``data_gaps`` keyed by event_id.
    """

    frames: list[TrainingEventFrame]
    bma_priors: dict[str, dict[str, float]]
    calibrator: SeverityStratifiedCalibrator
    split_conformal: SplitConformalRegressor
    mondrian_conformal: MondrianConformalRegressor
    data_gaps: dict[str, dict[str, str]]


def run_full_training(
    *,
    use_real_data: bool = True,
    cadence_hours: float = 1.0,
    rng_seed: int | None = None,
    frames: Iterable[TrainingEventFrame] | None = None,
) -> TrainingArtifacts:
    """Run the four sequential fits and return all artifacts.

    Args:
        use_real_data: If ``True``, attempt live ISWA / SWPC pulls (with
            synthetic-proxy fallback for sources that don't cover the
            event). If ``False``, skip the network entirely.
        cadence_hours: Time-grid cadence for the per-event dataframes.
        rng_seed: Override the synthetic-proxy seed.
        frames: Optional pre-loaded list of training-event frames. If
            provided, the loader is NOT invoked. Useful for tests and for
            re-running fits without re-pulling data.

    Returns:
        A :class:`TrainingArtifacts` bundle.
    """
    if frames is None:
        frames_list = load_all_training_events(
            use_real_data=use_real_data,
            cadence_hours=cadence_hours,
            rng_seed=rng_seed,
        )
    else:
        frames_list = list(frames)

    bma_priors = fit_bma_priors(frames_list)
    calibrator = fit_stratified_calibrators(frames_list, bma_priors)
    split_cp = fit_split_conformal(frames_list, bma_priors, calibrator=calibrator)
    mondrian_cp = fit_mondrian_conformal(frames_list, bma_priors, calibrator=calibrator)

    return TrainingArtifacts(
        frames=frames_list,
        bma_priors=bma_priors,
        calibrator=calibrator,
        split_conformal=split_cp,
        mondrian_conformal=mondrian_cp,
        data_gaps={f.event.event_id: dict(f.data_gaps) for f in frames_list},
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _git_sha(repo_root: Path) -> str:
    """Return the short git SHA for ``repo_root``, or ``"unknown"``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _build_transformation_record(
    *,
    record_id: str,
    transformation_type: str,
    parameters: dict[str, Any],
    code_ref: str,
    input_refs: list[str],
    output_refs: list[str],
) -> dict[str, Any]:
    """Construct a :class:`HeliosTransformationRecord` payload.

    Returns the JSON-friendly ``model_dump(mode="json")`` representation so
    the caller can write it to disk.
    """
    # Local import keeps the test environment lazy for fast unit tests that
    # don't touch persistence.
    from helios_provenance.models import Agent, HeliosTransformationRecord

    now = datetime.now(UTC)
    record = HeliosTransformationRecord(
        id=record_id,
        created_at=now,
        agent=Agent(
            id="helios-fusion-engine/training",
            name="HELIOS fusion-engine training pipeline",
            type="software",
            version=_fusion_engine_version(),
        ),
        type=transformation_type,  # type: ignore[arg-type]
        parameters=parameters,
        code_ref=code_ref,
        input_refs=input_refs,
        output_refs=output_refs,
    )
    return record.model_dump(mode="json")


def _fusion_engine_version() -> str:
    """Best-effort lookup of the installed fusion-engine version."""
    try:
        from importlib.metadata import version

        return version("helios-fusion-engine")
    except Exception:
        return "unknown"


def persist_artifacts(
    artifacts: TrainingArtifacts,
    weights_dir: Path,
    *,
    repo_root: Path,
    osf_preregistration_url: str | None = None,
) -> dict[str, Path]:
    """Write all four trained artifacts to ``weights_dir`` with provenance.

    Produces six files:

    * ``bma_priors_table_3_1.npz`` + ``.provenance.json``
    * ``isotonic_calibrators_stratified.npz`` + ``.provenance.json``
    * ``conformal_residuals_split.npz`` + ``.provenance.json``
    * ``conformal_residuals_mondrian.npz`` + ``.provenance.json``
    * ``manifest.json``

    Args:
        artifacts: Bundle from :func:`run_full_training`.
        weights_dir: Destination directory (typically
            ``helios-fusion-internal/weights/``). Created if missing.
        repo_root: Path to the fusion-engine repo root; used to look up the
            git SHA at training time. Pass the worktree's root, not the
            primary checkout.
        osf_preregistration_url: Optional OSF URL to record in the
            manifest. ``None`` until the operator files pre-registration.

    Returns:
        Map of artifact-name -> written file path.
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    git_sha = _git_sha(repo_root)
    training_run_id = (
        f"helios-fusion-engine/training-run/{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    event_ids = [f.event.event_id for f in artifacts.frames]

    # 1. BMA priors -- one .npz with seven named float arrays + a JSON
    #    sidecar that maps array index to model_id per event. Keeping the
    #    npz purely numeric lets the consumer pass allow_pickle=False.
    bma_path = weights_dir / "bma_priors_table_3_1.npz"
    bma_payload: dict[str, np.ndarray] = {}
    bma_param_summary: dict[str, dict[str, float]] = {}
    bma_index: dict[str, list[str]] = {}
    for event_id, weights in artifacts.bma_priors.items():
        model_ids = sorted(weights.keys())
        bma_payload[event_id] = np.asarray([weights[m] for m in model_ids], dtype=np.float64)
        bma_index[event_id] = model_ids
        bma_param_summary[event_id] = {m: float(weights[m]) for m in model_ids}
    np.savez(bma_path, **bma_payload)  # type: ignore[arg-type]
    (weights_dir / "bma_priors_table_3_1.index.json").write_text(
        json.dumps(bma_index, indent=2, sort_keys=True) + "\n"
    )
    written["bma_priors"] = bma_path
    _write_provenance(
        bma_path.with_suffix(".npz.provenance.json"),
        _build_transformation_record(
            record_id=f"{training_run_id}/bma_priors",
            transformation_type="bma",
            parameters={
                "weight_policy": "hss_clipped",
                "binary_threshold": 0.5,
                "weights_per_event": bma_param_summary,
                "git_sha": git_sha,
            },
            code_ref=f"helios_fusion.training.fit_bma@{git_sha}",
            input_refs=event_ids,
            output_refs=[bma_path.name],
        ),
    )

    # 2. Isotonic calibrators. The calibrator state-dict is structured JSON;
    #    we write it as a sidecar .state.json (the .npz file is reserved for
    #    numeric arrays so consumers can open it safely without pickling).
    cal_path = weights_dir / "isotonic_calibrators_stratified.npz"
    cal_state = artifacts.calibrator.to_state_dict()
    _counts_map = artifacts.calibrator.sample_counts
    cal_counts = np.asarray(
        [_counts_map["quiet"], _counts_map["moderate"], _counts_map["extreme"]],
        dtype=np.int64,
    )
    np.savez(cal_path, sample_counts=cal_counts)
    (weights_dir / "isotonic_calibrators_stratified.state.json").write_text(
        json.dumps(cal_state, indent=2, sort_keys=True) + "\n"
    )
    written["isotonic_calibrators"] = cal_path
    _write_provenance(
        cal_path.with_suffix(".npz.provenance.json"),
        _build_transformation_record(
            record_id=f"{training_run_id}/isotonic_calibrators_stratified",
            transformation_type="calibration",
            parameters={
                "approach": "severity_stratified_isotonic",
                "strata": ["quiet", "moderate", "extreme"],
                "sample_counts": artifacts.calibrator.sample_counts,
                "git_sha": git_sha,
            },
            code_ref=f"helios_fusion.training.fit_isotonic@{git_sha}",
            input_refs=event_ids,
            output_refs=[cal_path.name],
        ),
    )

    # 3. Split-conformal residuals.
    split_path = weights_dir / "conformal_residuals_split.npz"
    split_state = artifacts.split_conformal.to_state_dict()
    np.savez(
        split_path,
        residuals=np.asarray(split_state["residuals"], dtype=np.float64),
    )
    (weights_dir / "conformal_residuals_split.schema.json").write_text(
        json.dumps({"schema_version": split_state["schema_version"]}, indent=2) + "\n"
    )
    written["conformal_split"] = split_path
    _write_provenance(
        split_path.with_suffix(".npz.provenance.json"),
        _build_transformation_record(
            record_id=f"{training_run_id}/conformal_residuals_split",
            transformation_type="conformal",
            parameters={
                "approach": "split_marginal",
                "n_calibration": artifacts.split_conformal.n_calibration,
                "default_alpha": DEFAULT_ALPHA,
                "git_sha": git_sha,
            },
            code_ref=f"helios_fusion.training.fit_conformal@{git_sha}",
            input_refs=event_ids,
            output_refs=[split_path.name],
        ),
    )

    # 4. Mondrian conformal residuals -- one numeric array per stratum;
    #    schema metadata lives in a sidecar .state.json.
    mondrian_path = weights_dir / "conformal_residuals_mondrian.npz"
    mondrian_state = artifacts.mondrian_conformal.to_state_dict()
    mondrian_npz: dict[str, np.ndarray] = {}
    for stratum in ("quiet", "moderate", "extreme"):
        sub = mondrian_state["sub"][stratum]
        mondrian_npz[f"residuals__{stratum}"] = np.asarray(sub["residuals"], dtype=np.float64)
    np.savez(mondrian_path, **mondrian_npz)  # type: ignore[arg-type]
    (weights_dir / "conformal_residuals_mondrian.state.json").write_text(
        json.dumps(mondrian_state, indent=2, sort_keys=True) + "\n"
    )
    written["conformal_mondrian"] = mondrian_path
    _write_provenance(
        mondrian_path.with_suffix(".npz.provenance.json"),
        _build_transformation_record(
            record_id=f"{training_run_id}/conformal_residuals_mondrian",
            transformation_type="conformal",
            parameters={
                "approach": "mondrian_per_stratum",
                "per_stratum_counts": artifacts.mondrian_conformal.per_stratum_counts(),
                "default_alpha": DEFAULT_ALPHA,
                "git_sha": git_sha,
            },
            code_ref=f"helios_fusion.training.fit_conformal@{git_sha}",
            input_refs=event_ids,
            output_refs=[mondrian_path.name],
        ),
    )

    # 5. Manifest.
    manifest = {
        "training_run_id": training_run_id,
        "training_run_date_utc": datetime.now(UTC).isoformat(),
        "fusion_engine_version": _fusion_engine_version(),
        "fusion_engine_git_sha": git_sha,
        "connectors_version": _package_version("helios-spaceweather-connectors"),
        "provenance_spec_version": _package_version("helios-provenance-spec"),
        "training_events": [
            {
                "event_id": f.event.event_id,
                "label": f.event.label,
                "onset_utc": f.event.onset.isoformat(),
                "secondary_onsets_utc": [o.isoformat() for o in f.event.secondary_onsets],
                "n_rows": len(f.records),
                "component_models": list(f.component_models),
                "data_gaps": dict(f.data_gaps),
                "synthetic_proxy_rows": int((f.records["source"] == "synthetic_proxy").sum())
                if not f.records.empty
                else 0,
                "iswa_rows": int((f.records["source"] == "iswa").sum())
                if not f.records.empty
                else 0,
            }
            for f in artifacts.frames
        ],
        "artifacts": {name: path.name for name, path in written.items()},
        "osf_preregistration_url": osf_preregistration_url,
    }
    manifest_path = weights_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    written["manifest"] = manifest_path

    logger.info("persisted %d artifacts to %s", len(written), weights_dir)
    return written


def _write_provenance(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON-friendly provenance record and validate it.

    Re-loads the payload through :class:`HeliosTransformationRecord` so a
    corrupt write blows up here instead of at runtime.
    """
    from helios_provenance.models import HeliosTransformationRecord

    try:
        HeliosTransformationRecord.model_validate(payload)
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"provenance record at {path} failed validation: {exc}") from exc
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _package_version(name: str) -> str:
    """Best-effort version lookup; ``"unknown"`` if not installed."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


__all__ = [
    "DEFAULT_ALPHA",
    "TrainingArtifacts",
    "persist_artifacts",
    "run_full_training",
]
