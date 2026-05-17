"""Calibration tests.

Covers isotonic vs. Platt vs. severity-stratified. Asserts the proposal's
Platt-rejection rationale on synthetic data with severity-dependent bias.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helios_fusion.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SeverityStratifiedCalibrator,
)
from helios_fusion.eval.metrics import brier_score, reliability_diagram
from helios_fusion.types import SeverityStratum


def test_isotonic_round_trip_state_dict() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(np.float64)
    cal = IsotonicCalibrator()
    cal.fit(p, y)
    state = cal.to_state_dict()
    revived = IsotonicCalibrator.from_state_dict(state)
    np.testing.assert_allclose(cal.transform(p), revived.transform(p), atol=1e-12)


def test_isotonic_fixes_biased_stream(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
) -> None:
    """Isotonic on the underconfident stream should produce slope near 1.0."""
    p = synthetic_component_probs["underconfident"]
    y = synthetic_truth
    # 60/40 split: 60% fit, 40% evaluate.
    cut = int(0.6 * p.size)
    cal = IsotonicCalibrator()
    cal.fit(p[:cut], y[:cut])
    p_eval_cal = cal.transform(p[cut:])
    _, _, slope = reliability_diagram(p_eval_cal, y[cut:], n_bins=10)
    # H2 OSF target is |slope - 1| <= 0.15; demand the same on synthetic data.
    assert abs(slope - 1.0) < 0.15, f"slope after isotonic was {slope}"


def test_isotonic_output_is_monotone(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
) -> None:
    p = synthetic_component_probs["overconfident"]
    y = synthetic_truth
    cal = IsotonicCalibrator()
    cal.fit(p, y)
    # Probe on a sorted grid; transform output must be non-decreasing.
    grid = np.linspace(0.0, 1.0, 101)
    out = cal.transform(grid)
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-12), f"non-monotone: min diff {diffs.min()}"


def test_isotonic_clips_to_unit_interval() -> None:
    cal = IsotonicCalibrator()
    cal.fit(np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.0, 0.0, 1.0, 1.0]))
    out = cal.transform(np.array([0.0, 0.5, 1.0]))
    assert (out >= 0).all() and (out <= 1).all()


def test_isotonic_unfitted_raises() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.transform(np.array([0.5]))
    with pytest.raises(RuntimeError, match="cannot serialise"):
        cal.to_state_dict()


def test_isotonic_invalid_probabilities_raises() -> None:
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cal.fit(np.array([1.5, 0.2]), np.array([1.0, 0.0]))


def test_isotonic_state_dict_bad_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported schema_version"):
        IsotonicCalibrator.from_state_dict({"schema_version": "garbage"})


def test_platt_round_trip() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(np.float64)
    cal = PlattCalibrator()
    cal.fit(p, y)
    state = cal.to_state_dict()
    revived = PlattCalibrator.from_state_dict(state)
    np.testing.assert_allclose(cal.transform(p), revived.transform(p), atol=1e-12)


def test_platt_unfitted_raises() -> None:
    cal = PlattCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))
    with pytest.raises(RuntimeError):
        cal.a


def test_platt_state_dict_bad_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        PlattCalibrator.from_state_dict({"schema_version": "garbage"})


def test_platt_worsens_calibration_at_extremes(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
) -> None:
    """Proposal-rejection rationale, in code.

    On a stream whose miscalibration is non-monotone in probability (the
    ``severity_biased_extreme`` stream is well-calibrated at quiet/moderate
    Kp but overconfident at extreme Kp), Platt's two-parameter sigmoid
    cannot fix the localised tail miscalibration without distorting the
    middle. We assert isotonic strictly beats Platt by Brier score on the
    held-out half of the stream.
    """
    p = synthetic_component_probs["severity_biased_extreme"]
    y = synthetic_truth
    cut = int(0.6 * p.size)

    iso = IsotonicCalibrator()
    iso.fit(p[:cut], y[:cut])
    iso_eval = iso.transform(p[cut:])

    platt = PlattCalibrator()
    platt.fit(p[:cut], y[:cut])
    platt_eval = platt.transform(p[cut:])

    brier_iso, _ = brier_score(iso_eval, y[cut:], n_bootstrap=0)
    brier_platt, _ = brier_score(platt_eval, y[cut:], n_bootstrap=0)
    assert brier_iso < brier_platt, (
        f"isotonic Brier {brier_iso:.4f} should beat Platt Brier "
        f"{brier_platt:.4f} on tail-miscalibrated synthetic data"
    )


def test_severity_stratified_beats_unstratified_on_stratum_specific_bias(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
    synthetic_strata: list[SeverityStratum],
) -> None:
    """Severity-stratified isotonic should outperform unstratified on
    stratum-specific bias.

    The ``severity_biased_quiet`` stream is overconfident *only* on quiet
    Kp samples. An unstratified isotonic calibrator averages the bias
    pattern across all strata, leaving residual miscalibration on quiet;
    a per-stratum calibrator can recover.
    """
    p = synthetic_component_probs["severity_biased_quiet"]
    y = synthetic_truth
    s = synthetic_strata
    cut = int(0.6 * p.size)

    unstrat = IsotonicCalibrator()
    unstrat.fit(p[:cut], y[:cut])
    unstrat_eval = unstrat.transform(p[cut:])

    strat = SeverityStratifiedCalibrator()
    strat.fit(p[:cut], y[:cut], s[:cut])
    strat_eval = strat.transform(p[cut:], s[cut:])

    # Focus on quiet stratum where the bias lives.
    quiet_mask = np.array([lbl == "quiet" for lbl in s[cut:]], dtype=bool)
    b_unstrat, _ = brier_score(unstrat_eval[quiet_mask], y[cut:][quiet_mask], n_bootstrap=0)
    b_strat, _ = brier_score(strat_eval[quiet_mask], y[cut:][quiet_mask], n_bootstrap=0)
    assert b_strat <= b_unstrat + 1e-6, (
        f"stratified Brier on quiet={b_strat:.4f} should beat unstratified={b_unstrat:.4f}"
    )


def test_severity_stratified_round_trip(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_truth: np.ndarray,
    synthetic_strata: list[SeverityStratum],
) -> None:
    p = synthetic_component_probs["well_calibrated"]
    y = synthetic_truth
    s = synthetic_strata
    cal = SeverityStratifiedCalibrator()
    cal.fit(p, y, s)
    state = cal.to_state_dict()
    revived = SeverityStratifiedCalibrator.from_state_dict(state)
    np.testing.assert_allclose(cal.transform(p, s), revived.transform(p, s), atol=1e-12)


def test_severity_stratified_rejects_unknown_label() -> None:
    cal = SeverityStratifiedCalibrator()
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.95, 0.05])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    s = ["quiet", "quiet", "moderate", "moderate", "extreme", "bogus"]
    with pytest.raises(ValueError, match="unknown severity"):
        cal.fit(p, y, s)  # type: ignore[arg-type]


def test_severity_stratified_unfitted_transform_raises(
    synthetic_strata: list[SeverityStratum],
) -> None:
    cal = SeverityStratifiedCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.transform(np.array([0.5]), ["quiet"])


def test_severity_stratified_too_few_samples_raises() -> None:
    cal = SeverityStratifiedCalibrator()
    # Only one extreme sample — should raise.
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    s: list[SeverityStratum] = ["quiet", "quiet", "moderate", "moderate", "extreme"]
    with pytest.raises(ValueError, match=r">= 2"):
        cal.fit(p, y, s)


def test_severity_stratified_state_dict_bad_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SeverityStratifiedCalibrator.from_state_dict({"schema_version": "garbage"})


@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=20, max_size=200))
@settings(max_examples=20, deadline=None)
def test_isotonic_monotone_property(probs: list[float]) -> None:
    arr = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(3)
    y = (rng.uniform(0, 1, arr.size) < arr).astype(np.float64)
    cal = IsotonicCalibrator()
    cal.fit(arr, y)
    grid = np.linspace(0, 1, 51)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-12)
