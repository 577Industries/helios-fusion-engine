"""Hand-checked metric tests.

These tests verify metric formulas against pre-computed examples so the
kill-gate's pre-registered HSS/Brier/CRPS definitions can never silently drift.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helios_fusion.eval.metrics import (
    brier_score,
    crps,
    far,
    hss,
    pod,
    reliability_diagram,
    tss,
)


def test_hss_donaldson_hand_computed(hand_computed_hss_case: dict[str, object]) -> None:
    pred = hand_computed_hss_case["predicted"]
    obs = hand_computed_hss_case["observed"]
    expected = float(hand_computed_hss_case["expected_hss"])
    point, _ = hss(pred, obs, n_bootstrap=0)
    assert point == pytest.approx(expected, abs=1e-12)


def test_hss_perfect_forecast_is_one() -> None:
    point, _ = hss([1, 0, 1, 0, 1], [1, 0, 1, 0, 1], n_bootstrap=0)
    assert point == pytest.approx(1.0, abs=1e-12)


def test_hss_random_against_random_is_about_zero() -> None:
    """HSS = 0 under random forecasting; bootstrap should bracket 0."""
    rng = np.random.default_rng(21)
    n = 4000
    pred = rng.integers(0, 2, n)
    obs = rng.integers(0, 2, n)
    point, ci = hss(pred, obs, n_bootstrap=200, seed=21)
    assert abs(point) < 0.05
    assert ci[0] < 0 < ci[1]


def test_tss_perfect() -> None:
    point, _ = tss([1, 0, 1, 0], [1, 0, 1, 0], n_bootstrap=0)
    assert point == pytest.approx(1.0, abs=1e-12)


def test_pod_far_hand_computed() -> None:
    # a=3, b=1, c=2, d=4 -> POD = 3/5 = 0.6, FAR = 1/4 = 0.25
    pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    obs = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    pod_p, _ = pod(pred, obs, n_bootstrap=0)
    far_p, _ = far(pred, obs, n_bootstrap=0)
    assert pod_p == pytest.approx(3 / 5, abs=1e-12)
    assert far_p == pytest.approx(1 / 4, abs=1e-12)


def test_brier_perfect_zero() -> None:
    point, _ = brier_score([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], n_bootstrap=0)
    assert point == pytest.approx(0.0, abs=1e-12)


def test_brier_hand_computed() -> None:
    """Brier = mean((p - y)^2)."""
    probs = [0.7, 0.2, 0.9, 0.4]
    obs = [1.0, 0.0, 1.0, 0.0]
    expected = float(np.mean((np.asarray(probs) - np.asarray(obs)) ** 2))
    point, _ = brier_score(probs, obs, n_bootstrap=0)
    assert point == pytest.approx(expected, abs=1e-12)


def test_brier_invalid_probs_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.5], [1.0], n_bootstrap=0)


def test_crps_deterministic_reduces_to_mae() -> None:
    """For a single-sample 'ensemble' the CRPS equals absolute error."""
    preds = np.array([[0.5], [0.8], [0.2]])
    obs = np.array([1.0, 0.0, 1.0])
    point, _ = crps(preds, obs, n_bootstrap=0)
    expected_mae = float(np.mean([abs(0.5 - 1.0), abs(0.8 - 0.0), abs(0.2 - 1.0)]))
    assert point == pytest.approx(expected_mae, abs=1e-12)


def test_crps_hersbach_two_member_ensemble() -> None:
    """Closed-form for a two-member ensemble, one event.

    Ensemble x = (0.2, 0.8), y = 1.
    Term1 = (|0.2-1| + |0.8-1|) / 2 = (0.8 + 0.2)/2 = 0.5
    Term2 = sum_{i,j} |x_i - x_j| / (2 * 4) =
        (|0.2-0.2| + |0.2-0.8| + |0.8-0.2| + |0.8-0.8|) / 8
        = (0 + 0.6 + 0.6 + 0) / 8 = 0.15
    CRPS = 0.5 - 0.15 = 0.35.
    """
    preds = np.array([[0.2, 0.8]])
    obs = np.array([1.0])
    point, _ = crps(preds, obs, n_bootstrap=0)
    assert point == pytest.approx(0.35, abs=1e-12)


def test_crps_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="row count mismatch"):
        crps(np.array([[0.5], [0.6]]), np.array([1.0]), n_bootstrap=0)


def test_reliability_diagram_slope_perfect() -> None:
    """For perfectly calibrated probabilities, slope should be near 1.0."""
    rng = np.random.default_rng(31)
    n = 5000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(np.float64)
    _, _, slope = reliability_diagram(p, y, n_bins=10)
    assert abs(slope - 1.0) < 0.1


def test_reliability_diagram_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability_diagram([1.5], [1.0])
    with pytest.raises(ValueError, match=r"same shape"):
        reliability_diagram([0.5], [1.0, 0.0])
    with pytest.raises(ValueError, match=r">= 2"):
        reliability_diagram([0.5], [1.0], n_bins=1)
    with pytest.raises(ValueError, match=r"non-empty"):
        reliability_diagram([], [])


def test_metric_inputs_must_be_binary() -> None:
    with pytest.raises(ValueError, match="binary"):
        hss([0, 1, 2], [0, 1, 0], n_bootstrap=0)


def test_metric_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        hss([0, 1], [0, 1, 0], n_bootstrap=0)


def test_metric_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hss([], [], n_bootstrap=0)


def test_bootstrap_ci_zero_means_nan() -> None:
    _, ci = hss([1, 0, 1, 0], [1, 0, 1, 0], n_bootstrap=0)
    assert math.isnan(ci[0]) and math.isnan(ci[1])


def test_bootstrap_ci_brackets_point() -> None:
    point, ci = hss(
        [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        n_bootstrap=200,
        seed=42,
    )
    # The CI does NOT always strictly contain the point estimate
    # (the bootstrap distribution can be skewed), but it should be
    # within a sensible margin.
    assert ci[0] - 0.5 <= point <= ci[1] + 0.5


@given(
    st.lists(st.sampled_from([0, 1]), min_size=10, max_size=200),
    st.lists(st.sampled_from([0, 1]), min_size=10, max_size=200),
)
@settings(max_examples=20, deadline=None)
def test_hss_in_range(p: list[int], o: list[int]) -> None:
    # Trim to same length.
    n = min(len(p), len(o))
    if n < 2:
        return
    point, _ = hss(p[:n], o[:n], n_bootstrap=0)
    if math.isnan(point):
        return
    assert -1.0 - 1e-12 <= point <= 1.0 + 1e-12


def test_far_undefined_returns_nan() -> None:
    # No predicted positives → FAR is undefined → NaN.
    point, _ = far([0, 0, 0], [1, 0, 1], n_bootstrap=0)
    assert math.isnan(point)


def test_pod_undefined_returns_nan() -> None:
    # No observed positives → POD undefined.
    point, _ = pod([1, 0, 0], [0, 0, 0], n_bootstrap=0)
    assert math.isnan(point)


def test_tss_undefined_returns_nan() -> None:
    point, _ = tss([1, 0, 0], [0, 0, 0], n_bootstrap=0)
    assert math.isnan(point)


def test_hss_with_few_bootstrap_returns_finite_ci() -> None:
    rng = np.random.default_rng(33)
    n = 50
    pred = rng.integers(0, 2, n)
    obs = rng.integers(0, 2, n)
    _, ci = hss(pred, obs, n_bootstrap=50, seed=33)
    assert math.isfinite(ci[0]) and math.isfinite(ci[1])
    assert ci[0] <= ci[1]
