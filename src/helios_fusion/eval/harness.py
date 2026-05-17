"""High-level evaluation harness.

Composes :mod:`helios_fusion.eval.metrics` into a per-stratum AND aggregate
:class:`EvalReport`. The kill-gate runner consumes the same
:class:`EvalReport` and applies the pre-registered decision rules.

The report fields and structure are pinned to the OSF pre-registration
template (``helios-program/orchestration/osf_preregistration.template.md``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from helios_fusion.eval.metrics import (
    brier_score,
    crps,
    far,
    hss,
    pod,
    reliability_diagram,
    tss,
)
from helios_fusion.types import SeverityStratum

if TYPE_CHECKING:
    import numpy.typing as npt
    from pydantic import BaseModel

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_STRATA_TUPLE: tuple[SeverityStratum, ...] = get_args(SeverityStratum)
_DEFAULT_BINARY_THRESHOLD: float = 0.5


class MetricReport(BaseModel):
    """Container for one metric's point estimate and CI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    n_samples: int


class StratumReport(BaseModel):
    """Per-stratum metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stratum: SeverityStratum | None = None  # None ⇒ aggregate across strata
    n_samples: int
    hss: MetricReport
    tss: MetricReport
    pod: MetricReport
    far: MetricReport
    brier: MetricReport
    crps: MetricReport | None = None
    reliability_slope: float = float("nan")
    reliability_bin_centres: list[float] = Field(default_factory=list)
    reliability_bin_freqs: list[float] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full evaluation report consumed by the kill-gate.

    ``per_stratum`` carries one entry per Kp severity stratum (``quiet``,
    ``moderate``, ``extreme``); ``aggregate`` carries the same metrics
    computed across the full set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "helios-fusion-engine/eval/0.1.0"
    binary_threshold: float
    n_bootstrap: int
    aggregate: StratumReport
    per_stratum: dict[SeverityStratum, StratumReport]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable plain dict."""
        return self.model_dump(mode="json")


def _build_metric_report(
    point: float,
    ci: tuple[float, float],
    n_samples: int,
    ci_level: float = 0.95,
) -> MetricReport:
    return MetricReport(
        point=float(point),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        ci_level=ci_level,
        n_samples=n_samples,
    )


def _compute_stratum(
    probs: np.ndarray,
    observed: np.ndarray,
    binary_threshold: float,
    n_bootstrap: int,
    stratum: SeverityStratum | None,
    n_bins: int,
    seed: int | None,
) -> StratumReport:
    pred_bin = (probs >= binary_threshold).astype(np.int_)
    obs_bin = (observed >= binary_threshold).astype(np.int_)
    n = int(probs.size)

    hss_p, hss_ci = hss(pred_bin, obs_bin, n_bootstrap=n_bootstrap, seed=seed)
    tss_p, tss_ci = tss(pred_bin, obs_bin, n_bootstrap=n_bootstrap, seed=seed)
    pod_p, pod_ci = pod(pred_bin, obs_bin, n_bootstrap=n_bootstrap, seed=seed)
    far_p, far_ci = far(pred_bin, obs_bin, n_bootstrap=n_bootstrap, seed=seed)
    brier_p, brier_ci = brier_score(probs, observed, n_bootstrap=n_bootstrap, seed=seed)
    # Compute CRPS only when the (synthetic or real) inputs justify it; for
    # binary outcomes the CRPS over a deterministic forecast reduces to MAE
    # and is reported for completeness.
    crps_p, crps_ci = crps(probs, observed, n_bootstrap=n_bootstrap, seed=seed)

    if n >= n_bins:
        centres, freqs, slope = reliability_diagram(probs, observed, n_bins=n_bins)
    else:
        centres = np.array([], dtype=np.float64)
        freqs = np.array([], dtype=np.float64)
        slope = float("nan")

    return StratumReport(
        stratum=stratum,
        n_samples=n,
        hss=_build_metric_report(hss_p, hss_ci, n),
        tss=_build_metric_report(tss_p, tss_ci, n),
        pod=_build_metric_report(pod_p, pod_ci, n),
        far=_build_metric_report(far_p, far_ci, n),
        brier=_build_metric_report(brier_p, brier_ci, n),
        crps=_build_metric_report(crps_p, crps_ci, n),
        reliability_slope=slope,
        reliability_bin_centres=[float(x) for x in centres],
        reliability_bin_freqs=[float(x) for x in freqs],
    )


def evaluate(
    fused_probs: Sequence[float] | npt.ArrayLike,
    ground_truth: Sequence[float] | npt.ArrayLike,
    severity_strata: Sequence[SeverityStratum],
    *,
    binary_threshold: float = _DEFAULT_BINARY_THRESHOLD,
    n_bootstrap: int = 1000,
    n_bins: int = 10,
    seed: int | None = None,
) -> EvalReport:
    """Compute per-stratum AND aggregate metrics for one fused output run.

    Args:
        fused_probs: 1-D fused probability forecasts in ``[0, 1]``.
        ground_truth: 1-D observed values (0/1 or probability frequencies).
        severity_strata: Per-sample Kp stratum label.
        binary_threshold: Threshold for binarising probabilities into the
            contingency table.
        n_bootstrap: Number of bootstrap resamples for CIs (1000 per OSF
            template). Pass ``0`` to skip CIs (fast path for tests).
        n_bins: Number of reliability-diagram bins.
        seed: RNG seed; falls back to module default for reproducibility.

    Returns:
        :class:`EvalReport` with ``aggregate`` and ``per_stratum`` filled in.

    Raises:
        ValueError: On length mismatches, unknown stratum labels, or empty
            input.
    """
    probs = np.asarray(fused_probs, dtype=np.float64)
    obs = np.asarray(ground_truth, dtype=np.float64)
    strata = list(severity_strata)

    if probs.shape != obs.shape:
        raise ValueError(
            f"fused_probs and ground_truth must have same shape; got {probs.shape} vs {obs.shape}"
        )
    if probs.shape[0] != len(strata):
        raise ValueError(
            f"severity_strata length {len(strata)} does not match input length {probs.shape[0]}"
        )
    if probs.size == 0:
        raise ValueError("evaluate requires non-empty inputs")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("fused_probs must be in [0, 1]")
    unknown = {s for s in strata if s not in _STRATA_TUPLE}
    if unknown:
        raise ValueError(f"unknown severity stratum label(s): {sorted(unknown)}")

    aggregate = _compute_stratum(probs, obs, binary_threshold, n_bootstrap, None, n_bins, seed)

    per_stratum: dict[SeverityStratum, StratumReport] = {}
    grouped_indices: dict[SeverityStratum, list[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        grouped_indices[s].append(i)

    for stratum in _STRATA_TUPLE:
        idxs = grouped_indices.get(stratum, [])
        if not idxs:
            # Emit a placeholder with NaN metrics so the report shape is
            # stable across stratum populations.
            per_stratum[stratum] = StratumReport(
                stratum=stratum,
                n_samples=0,
                hss=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                tss=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                pod=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                far=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                brier=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                crps=_build_metric_report(float("nan"), (float("nan"), float("nan")), 0),
                reliability_slope=float("nan"),
                reliability_bin_centres=[],
                reliability_bin_freqs=[],
            )
            continue
        idx_arr = np.array(idxs, dtype=np.int_)
        per_stratum[stratum] = _compute_stratum(
            probs[idx_arr],
            obs[idx_arr],
            binary_threshold,
            n_bootstrap,
            stratum,
            n_bins,
            seed,
        )

    report = EvalReport(
        binary_threshold=binary_threshold,
        n_bootstrap=n_bootstrap,
        aggregate=aggregate,
        per_stratum=per_stratum,
    )
    logger.info(
        "Evaluation complete: n=%d, aggregate HSS=%.4f, slope=%.4f",
        probs.size,
        aggregate.hss.point,
        aggregate.reliability_slope,
    )
    return report
