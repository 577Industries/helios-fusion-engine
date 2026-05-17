"""CCMC-compatible verification metrics with bootstrap confidence intervals.

All metric definitions match the OSF pre-registration template verbatim.
See ``helios-program/orchestration/osf_preregistration.template.md`` §6.

Contingency-table notation
--------------------------

For a binary forecast / observation pair, the 2x2 contingency table is::

                            observed=1     observed=0
        predicted=1            a (hits)      b (false alarms)
        predicted=0            c (misses)    d (correct rejections)

with totals ``n = a + b + c + d``.

Metric formulas (locked):

* **HSS** (Donaldson 1975): :math:`\\text{HSS} = \\frac{2(ad - bc)}
  {(a + c)(c + d) + (a + b)(b + d)}`. Range: :math:`[-1, 1]`; perfect = 1.
* **TSS** (Hanssen-Kuipers): :math:`\\text{TSS} = \\frac{a}{a+c} - \\frac{b}{b+d}`.
* **POD**: :math:`\\text{POD} = \\frac{a}{a+c}` (sensitivity).
* **FAR**: :math:`\\text{FAR} = \\frac{b}{a+b}` (false alarm ratio; lower is better).

Probability metrics:

* **Brier**: :math:`\\frac{1}{n}\\sum (p_i - y_i)^2`.
* **CRPS** for an empirical predictive distribution given by quantiles or by
  a quantile-based interval. The framework provides a generic ensemble CRPS
  estimator (Hersbach decomposition for sorted samples).
* **Reliability slope**: linear regression of ``observed_freq`` on
  ``bin_centre`` across ``n_bins`` equal-frequency bins. Slope = 1 ⇒ perfect
  calibration; H2 in the OSF pre-registration tests :math:`|s - 1| \\le 0.15`
  per stratum.

Bootstrap CIs
-------------

All metric functions accept ``n_bootstrap`` (default 1000) and return
``(point_estimate, (lo, hi))`` for the 95% two-sided CI by default. Resampling
is over events (rows) with replacement, per the OSF pre-registration template.
``n_bootstrap=0`` skips resampling and returns ``(point, (nan, nan))``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_DEFAULT_BOOTSTRAP: int = 1000
_DEFAULT_CI_LEVEL: float = 0.95
_DEFAULT_BOOTSTRAP_SEED: int = 20260517  # YYYYMMDD; locked for reproducibility


# --------------------------------------------------------------------------- #
# Contingency-table helpers
# --------------------------------------------------------------------------- #
def _contingency(
    predicted: npt.NDArray[np.int_], observed: npt.NDArray[np.int_]
) -> tuple[int, int, int, int]:
    """Return ``(a, b, c, d)`` from binary arrays (any 0/1-like dtype)."""
    a = int(np.sum((predicted == 1) & (observed == 1)))
    b = int(np.sum((predicted == 1) & (observed == 0)))
    c = int(np.sum((predicted == 0) & (observed == 1)))
    d = int(np.sum((predicted == 0) & (observed == 0)))
    return a, b, c, d


def _to_binary_array(x: Sequence[int] | npt.ArrayLike) -> npt.NDArray[np.int_]:
    arr = np.asarray(x).astype(np.int_)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D array; got shape {arr.shape}")
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"binary arrays must contain only 0/1; got {sorted(unique)}")
    return arr


def _check_same_length(a: npt.NDArray[np.int_], b: npt.NDArray[np.int_]) -> None:
    if a.shape != b.shape:
        raise ValueError(f"predicted and observed must have same shape; got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("inputs must be non-empty")


# --------------------------------------------------------------------------- #
# Point-estimate functions (private; the public API wraps with bootstrap)
# --------------------------------------------------------------------------- #
def _hss_point(predicted: npt.NDArray[np.int_], observed: npt.NDArray[np.int_]) -> float:
    a, b, c, d = _contingency(predicted, observed)
    num = 2 * (a * d - b * c)
    den = (a + c) * (c + d) + (a + b) * (b + d)
    if den == 0:
        return float("nan")
    return num / den


def _tss_point(predicted: npt.NDArray[np.int_], observed: npt.NDArray[np.int_]) -> float:
    a, b, c, d = _contingency(predicted, observed)
    if (a + c) == 0 or (b + d) == 0:
        return float("nan")
    return a / (a + c) - b / (b + d)


def _pod_point(predicted: npt.NDArray[np.int_], observed: npt.NDArray[np.int_]) -> float:
    a, _b, c, _d = _contingency(predicted, observed)
    if (a + c) == 0:
        return float("nan")
    return a / (a + c)


def _far_point(predicted: npt.NDArray[np.int_], observed: npt.NDArray[np.int_]) -> float:
    a, b, _c, _d = _contingency(predicted, observed)
    if (a + b) == 0:
        return float("nan")
    return b / (a + b)


def _brier_point(probs: npt.NDArray[np.float64], observed: npt.NDArray[np.float64]) -> float:
    return float(np.mean((probs - observed) ** 2))


# --------------------------------------------------------------------------- #
# Bootstrap helper
# --------------------------------------------------------------------------- #
def _bootstrap_ci(
    fn: Callable[[npt.NDArray[np.int_]], float],
    n: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_level: float,
) -> tuple[float, float]:
    """Sample ``n_bootstrap`` resampled indices and report a two-sided CI.

    ``fn`` is invoked with a numpy index array and must return a float.
    """
    if n_bootstrap <= 0:
        return float("nan"), float("nan")
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = fn(idx)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - ci_level
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return lo, hi


def _make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(_DEFAULT_BOOTSTRAP_SEED if seed is None else seed)


# --------------------------------------------------------------------------- #
# Public metric API (point + bootstrap CI)
# --------------------------------------------------------------------------- #
def hss(
    predicted_binary: Sequence[int] | npt.ArrayLike,
    observed_binary: Sequence[int] | npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """Heidke Skill Score (Donaldson 1975).

    Args:
        predicted_binary: 1-D binary forecast array.
        observed_binary: 1-D binary observation array.
        n_bootstrap: Number of bootstrap resamples; ``0`` disables CI.
        ci_level: Two-sided CI level (default 0.95).
        seed: RNG seed for reproducibility; falls back to module default.

    Returns:
        Tuple ``(point_estimate, (ci_low, ci_high))``.
    """
    p = _to_binary_array(predicted_binary)
    o = _to_binary_array(observed_binary)
    _check_same_length(p, o)
    point = _hss_point(p, o)
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: _hss_point(p[idx], o[idx]),
        n=p.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


def tss(
    predicted_binary: Sequence[int] | npt.ArrayLike,
    observed_binary: Sequence[int] | npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """True Skill Statistic (Hanssen-Kuipers)."""
    p = _to_binary_array(predicted_binary)
    o = _to_binary_array(observed_binary)
    _check_same_length(p, o)
    point = _tss_point(p, o)
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: _tss_point(p[idx], o[idx]),
        n=p.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


def pod(
    predicted_binary: Sequence[int] | npt.ArrayLike,
    observed_binary: Sequence[int] | npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """Probability of Detection (sensitivity)."""
    p = _to_binary_array(predicted_binary)
    o = _to_binary_array(observed_binary)
    _check_same_length(p, o)
    point = _pod_point(p, o)
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: _pod_point(p[idx], o[idx]),
        n=p.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


def far(
    predicted_binary: Sequence[int] | npt.ArrayLike,
    observed_binary: Sequence[int] | npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """False Alarm Ratio (lower is better)."""
    p = _to_binary_array(predicted_binary)
    o = _to_binary_array(observed_binary)
    _check_same_length(p, o)
    point = _far_point(p, o)
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: _far_point(p[idx], o[idx]),
        n=p.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


def brier_score(
    probs: npt.ArrayLike,
    observed: npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """Mean squared error of probability forecasts.

    Args:
        probs: 1-D probability forecasts in ``[0, 1]``.
        observed: 1-D observations (0/1 or probability frequencies).

    Returns:
        ``(point, (lo, hi))`` per the bootstrap protocol.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"probs and observed must have same shape; got {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("probs/observed must be non-empty")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs must be in [0, 1]")
    point = _brier_point(p, y)
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: _brier_point(p[idx], y[idx]),
        n=p.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


def crps(
    predicted_quantiles: npt.ArrayLike,
    observed: npt.ArrayLike,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
    ci_level: float = _DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, tuple[float, float]]:
    """Continuous Ranked Probability Score (ensemble form).

    Estimates CRPS from an ensemble representation of the predictive
    distribution. The ensemble form (Hersbach 2000; Gneiting & Raftery 2007)
    is

    .. math::

        \\text{CRPS} = \\frac{1}{M} \\sum_{m=1}^{M} |x_m - y|
            - \\frac{1}{2 M^2} \\sum_{m=1}^{M} \\sum_{m'=1}^{M} |x_m - x_{m'}|

    where :math:`\\{x_m\\}` is an ensemble of ``M`` samples (or quantiles)
    representing the predictive distribution for a single forecast instance.
    The score is averaged across forecast instances.

    Args:
        predicted_quantiles: 2-D array of shape ``(n_instances, M)``: each
            row is the ensemble for one forecast. For a deterministic
            forecast, ``M=1`` is permitted (the score reduces to MAE).
        observed: 1-D array of length ``n_instances``.
        n_bootstrap: Bootstrap resamples; ``0`` disables CI.
        ci_level: CI level.
        seed: RNG seed.

    Returns:
        ``(point_estimate, (lo, hi))``.
    """
    q = np.asarray(predicted_quantiles, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    y = np.asarray(observed, dtype=np.float64)
    if y.ndim != 1 or q.ndim != 2:
        raise ValueError(
            f"predicted_quantiles must be 2-D and observed 1-D; got shapes {q.shape} and {y.shape}"
        )
    if q.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: q has {q.shape[0]} rows, y has {y.shape[0]}")

    def _per_instance_crps(ensemble: npt.NDArray[np.float64], obs: float) -> float:
        m = ensemble.size
        term1 = float(np.mean(np.abs(ensemble - obs)))
        if m == 1:
            return term1
        # vectorised pairwise term
        diffs = np.abs(ensemble[:, None] - ensemble[None, :])
        term2 = float(diffs.sum()) / (2 * m * m)
        return term1 - term2

    per_instance = np.array(
        [_per_instance_crps(q[i], y[i]) for i in range(q.shape[0])],
        dtype=np.float64,
    )
    point = float(per_instance.mean())
    rng = _make_rng(seed)
    ci = _bootstrap_ci(
        lambda idx: float(per_instance[idx].mean()),
        n=per_instance.size,
        n_bootstrap=n_bootstrap,
        rng=rng,
        ci_level=ci_level,
    )
    return point, ci


# --------------------------------------------------------------------------- #
# Reliability diagram
# --------------------------------------------------------------------------- #
def reliability_diagram(
    probs: npt.ArrayLike,
    observed: npt.ArrayLike,
    n_bins: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Reliability diagram (equal-width bins) and fitted slope.

    Bins are equal-width in ``[0, 1]``. Empty bins are dropped before fitting
    the slope. The slope is the OLS slope of observed frequency on bin
    centre, weighted by bin sample count (Bröcker 2008 §2).

    Args:
        probs: 1-D probabilities in ``[0, 1]``.
        observed: 1-D observations (0/1 or frequencies).
        n_bins: Number of equal-width bins. Default 10.

    Returns:
        Tuple ``(bin_centres, observed_freq_per_bin, slope)``.
        Bins with zero samples are dropped from the first two return values.

    Raises:
        ValueError: On shape mismatch or empty input.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"probs and observed must have same shape; got {p.shape} vs {y.shape}")
    if p.size == 0:
        raise ValueError("probs/observed must be non-empty")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs must be in [0, 1]")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1-based bins for values inside; clamp upper edge.
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    centres = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(n_bins)], dtype=np.float64)
    freqs = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = bins == i
        if mask.any():
            freqs[i] = float(y[mask].mean())
            counts[i] = int(mask.sum())

    # Drop empty bins for the returned arrays AND for the slope fit.
    keep = counts > 0
    centres_kept = centres[keep]
    freqs_kept = freqs[keep]
    counts_kept = counts[keep].astype(np.float64)

    slope: float
    if centres_kept.size < 2:
        slope = float("nan")
    else:
        # Weighted linear regression: slope of freq on centre, weights = counts.
        w = counts_kept
        x = centres_kept
        z = freqs_kept
        x_mean = float(np.sum(w * x) / np.sum(w))
        z_mean = float(np.sum(w * z) / np.sum(w))
        num = float(np.sum(w * (x - x_mean) * (z - z_mean)))
        den = float(np.sum(w * (x - x_mean) ** 2))
        slope = num / den if den > 0 else float("nan")
        if math.isnan(slope) or math.isinf(slope):
            slope = float("nan")

    return centres_kept, freqs_kept, slope
