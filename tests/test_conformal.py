"""Conformal prediction tests.

Verifies marginal coverage for split conformal and per-stratum coverage for
Mondrian conformal on synthetic data with known noise distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from helios_fusion.conformal import (
    MondrianConformalRegressor,
    SplitConformalRegressor,
)
from helios_fusion.types import SeverityStratum


def _coverage(intervals: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1])))


def test_split_conformal_achieves_marginal_coverage() -> None:
    """At alpha=0.1, empirical coverage on a held-out set is within 5pp of 0.9."""
    rng = np.random.default_rng(11)
    n = 4000
    y_true = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    predictions = y_true + noise

    # 60% calibration, 40% test.
    cut = int(n * 0.6)
    reg = SplitConformalRegressor()
    reg.fit(predictions[:cut], y_true[:cut])

    intervals = reg.predict_interval(predictions[cut:], alpha=0.1)
    coverage = _coverage(intervals, y_true[cut:])
    assert 0.85 < coverage < 0.95, f"coverage {coverage:.3f} far from 0.9"


def test_split_conformal_coverage_at_alpha_005() -> None:
    """At alpha=0.05, empirical coverage should be near 0.95."""
    rng = np.random.default_rng(12)
    n = 4000
    y_true = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    predictions = y_true + noise

    cut = int(n * 0.6)
    reg = SplitConformalRegressor()
    reg.fit(predictions[:cut], y_true[:cut])
    intervals = reg.predict_interval(predictions[cut:], alpha=0.05)
    coverage = _coverage(intervals, y_true[cut:])
    assert 0.92 < coverage < 0.98, f"coverage {coverage:.3f} far from 0.95"


def test_split_conformal_round_trip() -> None:
    rng = np.random.default_rng(13)
    p = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)
    reg = SplitConformalRegressor()
    reg.fit(p, y)
    state = reg.to_state_dict()
    revived = SplitConformalRegressor.from_state_dict(state)
    np.testing.assert_allclose(
        reg.predict_interval(p[:5]), revived.predict_interval(p[:5]), atol=1e-12
    )


def test_split_conformal_unfitted_raises() -> None:
    reg = SplitConformalRegressor()
    with pytest.raises(RuntimeError, match="must be fit"):
        reg.predict_interval(np.array([0.5]))
    with pytest.raises(RuntimeError, match="cannot serialise"):
        reg.to_state_dict()


def test_split_conformal_bad_alpha_raises() -> None:
    reg = SplitConformalRegressor()
    reg.fit(np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.5]))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        reg.predict_interval(np.array([0.0]), alpha=1.5)


def test_split_conformal_state_dict_bad_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SplitConformalRegressor.from_state_dict({"schema_version": "garbage"})


def test_mondrian_coverage_per_stratum() -> None:
    """Mondrian intervals achieve target coverage *within each* stratum."""
    rng = np.random.default_rng(14)
    n_per = 1500
    strata: list[SeverityStratum] = ["quiet"] * n_per + ["moderate"] * n_per + ["extreme"] * n_per

    # Noise level varies per stratum — Mondrian should fit per-stratum widths.
    y_quiet = rng.normal(0.0, 0.5, n_per)
    y_moderate = rng.normal(0.0, 1.0, n_per)
    y_extreme = rng.normal(0.0, 2.0, n_per)
    y = np.concatenate([y_quiet, y_moderate, y_extreme])

    # Predictions = truth + per-stratum noise of matching scale.
    p = y + np.concatenate(
        [
            rng.normal(0.0, 0.1, n_per),
            rng.normal(0.0, 0.2, n_per),
            rng.normal(0.0, 0.4, n_per),
        ]
    )

    # 60/40 split keeping strata interleaved.
    idx = np.arange(y.size)
    rng.shuffle(idx)
    cut = int(y.size * 0.6)
    cal_idx, test_idx = idx[:cut], idx[cut:]

    cal_p, cal_y = p[cal_idx], y[cal_idx]
    cal_s = [strata[i] for i in cal_idx]
    test_p, test_y = p[test_idx], y[test_idx]
    test_s = [strata[i] for i in test_idx]

    reg = MondrianConformalRegressor()
    reg.fit(cal_p, cal_y, cal_s)
    intervals = reg.predict_interval(test_p, test_s, alpha=0.1)

    # Aggregate coverage near 0.9.
    cov = _coverage(intervals, test_y)
    assert 0.85 < cov < 0.95, f"marginal coverage {cov:.3f} far from 0.9"

    # Per-stratum coverage individually within 5pp of 0.9.
    for s_label in ("quiet", "moderate", "extreme"):
        mask = np.array([lbl == s_label for lbl in test_s], dtype=bool)
        sub_cov = _coverage(intervals[mask], test_y[mask])
        assert 0.83 < sub_cov < 0.97, (
            f"per-stratum coverage for {s_label}={sub_cov:.3f} far from 0.9"
        )

    # Per-stratum widths should reflect the noise levels: quiet < moderate < extreme.
    widths = {}
    for s_label in ("quiet", "moderate", "extreme"):
        mask = np.array([lbl == s_label for lbl in test_s], dtype=bool)
        widths[s_label] = float(np.mean(intervals[mask, 1] - intervals[mask, 0]))
    assert widths["quiet"] < widths["moderate"] < widths["extreme"]


def test_mondrian_round_trip() -> None:
    rng = np.random.default_rng(15)
    p = rng.normal(0, 1, 60)
    y = rng.normal(0, 1, 60)
    strata: list[SeverityStratum] = ["quiet"] * 20 + ["moderate"] * 20 + ["extreme"] * 20
    reg = MondrianConformalRegressor()
    reg.fit(p, y, strata)
    state = reg.to_state_dict()
    revived = MondrianConformalRegressor.from_state_dict(state)
    np.testing.assert_allclose(
        reg.predict_interval(p[:10], strata[:10]),
        revived.predict_interval(p[:10], strata[:10]),
        atol=1e-12,
    )


def test_mondrian_unfitted_raises() -> None:
    reg = MondrianConformalRegressor()
    with pytest.raises(RuntimeError, match="must be fit"):
        reg.predict_interval(np.array([0.0]), ["quiet"])
    with pytest.raises(RuntimeError, match="cannot serialise"):
        reg.to_state_dict()


def test_mondrian_unknown_stratum_raises() -> None:
    reg = MondrianConformalRegressor()
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="unknown severity"):
        reg.fit(p, y, ["bogus", "quiet", "moderate"])  # type: ignore[list-item]


def test_mondrian_missing_stratum_in_fit_raises() -> None:
    reg = MondrianConformalRegressor()
    # No "extreme" samples — should raise.
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="no calibration samples"):
        reg.fit(p, y, ["quiet", "quiet", "moderate"])


def test_mondrian_state_dict_bad_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MondrianConformalRegressor.from_state_dict({"schema_version": "garbage"})
