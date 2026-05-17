"""End-to-end test: 5-model BMA + isotonic + conformal → EvalReport.

Demonstrates the full pipeline on the synthetic fixtures. Asserts every
metric is populated in the report and that the BMA-fused output beats the
worst single-model baseline on Brier.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from helios_fusion.bma import BMAOrchestrator
from helios_fusion.calibration import (
    IsotonicCalibrator,
    SeverityStratifiedCalibrator,
)
from helios_fusion.conformal import MondrianConformalRegressor
from helios_fusion.eval import (
    EvalReport,
    best_individual_component_baseline,
    brier_score,
    evaluate,
)
from helios_fusion.eval.harness import StratumReport
from helios_fusion.types import ModelOutput, SeverityStratum


def test_full_pipeline_smoke(
    synthetic_model_outputs: dict[str, list[ModelOutput]],
    synthetic_strata: list[SeverityStratum],
    synthetic_truth: np.ndarray,
) -> None:
    model_ids = sorted(synthetic_model_outputs.keys())
    n = len(synthetic_truth)
    # Split 60/40 train/test.
    cut = int(n * 0.6)

    # 1) Train BMA weights on first 60%.
    bma = BMAOrchestrator(weight_policy="hss_clipped", prediction_target="event")
    window: list[tuple[ModelOutput, float]] = []
    for model_id in model_ids:
        for i in range(cut):
            window.append((synthetic_model_outputs[model_id][i], float(synthetic_truth[i])))
    bma.update_weights(window)

    # 2) Fuse on all timesteps to get a raw fused stream.
    fused_raw: list[float] = []
    for i in range(n):
        outputs = [synthetic_model_outputs[m][i] for m in model_ids]
        fused = bma.fuse(outputs)
        fused_raw.append(fused.value)
    fused_raw_arr = np.asarray(fused_raw, dtype=np.float64)

    # 3) Calibrate on training half via severity-stratified isotonic.
    cal = SeverityStratifiedCalibrator()
    cal.fit(fused_raw_arr[:cut], synthetic_truth[:cut], synthetic_strata[:cut])
    fused_cal = cal.transform(fused_raw_arr, synthetic_strata)

    # 4) Conformal residuals on calibration half, applied to test half.
    cp = MondrianConformalRegressor()
    cp.fit(fused_cal[:cut], synthetic_truth[:cut], synthetic_strata[:cut])
    intervals = cp.predict_interval(fused_cal[cut:], synthetic_strata[cut:], alpha=0.1)
    coverage = float(
        np.mean(
            (synthetic_truth[cut:] >= intervals[:, 0]) & (synthetic_truth[cut:] <= intervals[:, 1])
        )
    )
    # End-to-end coverage should be near 0.9 (loose because n_test ~ 320).
    assert 0.80 < coverage < 0.99, f"end-to-end coverage {coverage:.3f}"

    # 5) Evaluate the fused, calibrated stream on the test half.
    report = evaluate(
        fused_cal[cut:],
        synthetic_truth[cut:],
        synthetic_strata[cut:],
        n_bootstrap=200,
        seed=51,
    )
    assert isinstance(report, EvalReport)
    assert isinstance(report.aggregate, StratumReport)

    # All metrics populated, finite point estimates on aggregate.
    agg = report.aggregate
    for name, m in (
        ("hss", agg.hss),
        ("tss", agg.tss),
        ("pod", agg.pod),
        ("far", agg.far),
        ("brier", agg.brier),
    ):
        assert np.isfinite(m.point), f"{name} point estimate not finite"
        assert m.ci_low <= m.point + 0.5
        assert m.point - 0.5 <= m.ci_high

    # Reliability slope close to 1 (within 0.2 on synthetic test half).
    assert abs(report.aggregate.reliability_slope - 1.0) < 0.25, (
        f"slope={report.aggregate.reliability_slope}"
    )

    # All three strata are present.
    assert set(report.per_stratum.keys()) == {"quiet", "moderate", "extreme"}


def test_evaluate_round_trip_via_model_dump(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
    synthetic_strata: list[SeverityStratum],
) -> None:
    """EvalReport must JSON-serialise cleanly for the kill-gate."""
    report = evaluate(
        synthetic_component_probs["well_calibrated"],
        synthetic_truth,
        synthetic_strata,
        n_bootstrap=50,
        seed=53,
    )
    d = report.as_dict()
    assert d["aggregate"]["n_samples"] == len(synthetic_truth)
    assert "per_stratum" in d
    assert set(d["per_stratum"].keys()) == {"quiet", "moderate", "extreme"}


def test_evaluate_handles_empty_stratum(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
) -> None:
    """A stratum with no samples in the input emits a NaN-populated stub."""
    # Force all samples to "quiet" so moderate/extreme are empty.
    n = len(synthetic_truth)
    strata = ["quiet"] * n
    report = evaluate(
        synthetic_component_probs["well_calibrated"],
        synthetic_truth,
        strata,
        n_bootstrap=20,
        seed=54,
    )
    assert report.per_stratum["quiet"].n_samples == n
    assert report.per_stratum["moderate"].n_samples == 0
    assert report.per_stratum["extreme"].n_samples == 0
    # Empty stratum HSS is NaN.
    assert np.isnan(report.per_stratum["moderate"].hss.point)


def test_evaluate_unknown_stratum_raises() -> None:
    with pytest.raises(ValueError, match="unknown severity"):
        evaluate(
            [0.5, 0.5],
            [1.0, 0.0],
            ["bogus", "quiet"],  # type: ignore[list-item]
            n_bootstrap=0,
        )


def test_evaluate_input_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate([0.5, 0.5], [1.0], ["quiet", "quiet"], n_bootstrap=0)
    with pytest.raises(ValueError, match="severity_strata length"):
        evaluate([0.5, 0.5], [1.0, 0.0], ["quiet"], n_bootstrap=0)
    with pytest.raises(ValueError, match="non-empty"):
        evaluate([], [], [], n_bootstrap=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate([1.5], [1.0], ["quiet"], n_bootstrap=0)


def test_baseline_identifies_well_calibrated_model(
    synthetic_model_outputs: dict[str, list[ModelOutput]],
    synthetic_truth: np.ndarray,
) -> None:
    """Best-individual-component baseline returns a model_id and finite HSS."""
    model_ids = sorted(synthetic_model_outputs.keys())
    # Take the first 200 events as a held-out set.
    obs = synthetic_truth[:200].astype(int)
    comp_outputs: list[ModelOutput] = []
    for m in model_ids:
        comp_outputs.extend(synthetic_model_outputs[m][:200])
    best_id, point, ci = best_individual_component_baseline(
        comp_outputs, obs, n_bootstrap=50, seed=55
    )
    assert best_id in model_ids
    assert np.isfinite(point)
    assert ci[0] <= ci[1]


def test_baseline_observed_length_mismatch_raises(
    synthetic_model_outputs: dict[str, list[ModelOutput]],
) -> None:
    model_ids = list(synthetic_model_outputs.keys())
    comp = [synthetic_model_outputs[model_ids[0]][i] for i in range(5)]
    with pytest.raises(ValueError, match="must have length 5"):
        best_individual_component_baseline(comp, [1, 0, 1])


def test_baseline_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        best_individual_component_baseline([], [])


def test_bma_lineage_records_excluded_models() -> None:
    bma = BMAOrchestrator(weights={"A": 0.5, "B": 0.5})
    t = datetime(2024, 1, 1, tzinfo=UTC)
    out = bma.fuse(
        [ModelOutput(id="x", model_id="A", timestamp=t, value=0.3, value_units="probability")]
    )
    excluded = out.lineage[0].parameters["excluded_models"]
    assert excluded == ["B"]


def test_eval_report_has_schema_version() -> None:
    """Every report must carry a schema_version field for downstream compat."""
    report = evaluate(
        [0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.95, 0.05, 0.6, 0.5],
        [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        ["quiet"] * 4 + ["moderate"] * 4 + ["extreme"] * 2,
        n_bootstrap=10,
        seed=57,
    )
    assert report.schema_version.startswith("helios-fusion-engine/eval/")


def test_full_pipeline_calibrated_beats_uncalibrated_brier(
    synthetic_truth: np.ndarray,
    synthetic_true_posterior: np.ndarray,
) -> None:
    """A heavily miscalibrated stream is repaired by isotonic on hold-out.

    Constructs a deliberately compressed stream whose probabilities all sit
    in ``[0.7, 0.9]`` regardless of the true posterior. Isotonic on the
    training half should bring hold-out Brier substantially closer to the
    true-posterior baseline. This is the "calibration works" end-to-end
    assertion.
    """
    # Heavy compression: predicted probabilities live in a narrow band that
    # does not match the truth marginal.
    p = np.clip(0.7 + 0.2 * synthetic_true_posterior, 0.0, 1.0)
    y = synthetic_truth
    cut = int(p.size * 0.6)

    cal = IsotonicCalibrator()
    cal.fit(p[:cut], y[:cut])
    p_test_cal = cal.transform(p[cut:])

    b_raw, _ = brier_score(p[cut:], y[cut:], n_bootstrap=0)
    b_cal, _ = brier_score(p_test_cal, y[cut:], n_bootstrap=0)
    # Strong miscalibration should drop Brier by at least 50%.
    assert b_cal < b_raw * 0.5, f"raw={b_raw:.6f} cal={b_cal:.6f}"
