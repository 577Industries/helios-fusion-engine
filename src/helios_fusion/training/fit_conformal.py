"""Fit the split + Mondrian conformal calibration residual sets.

Both fits use the BMA-fused-then-isotonic-calibrated probability stream
across the seven training events as the calibration set:

* :class:`SplitConformalRegressor` -- marginal residual quantile (one
  array of absolute residuals).
* :class:`MondrianConformalRegressor` -- per-stratum residual quantiles
  (three sub-arrays, one per Kp severity stratum).

The locked alpha for the kill-gate is ``0.1`` (90% intervals); this module
returns the fitted regressors with their full residual sets so any alpha
in (0, 1) can be requested downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from helios_fusion.conformal.mondrian import MondrianConformalRegressor
from helios_fusion.conformal.split import SplitConformalRegressor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from helios_fusion.calibration.stratified import SeverityStratifiedCalibrator
    from helios_fusion.training.load_table_3_1 import TrainingEventFrame

logger = logging.getLogger(__name__)


def _build_calibration_set(
    frames: Iterable[TrainingEventFrame],
    bma_priors: dict[str, dict[str, float]],
    calibrator: SeverityStratifiedCalibrator | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build pooled (fused_probs, observed, strata) across training events.

    If ``calibrator`` is provided, the fused probabilities are passed
    through the per-stratum isotonic calibration; otherwise they're returned
    raw (BMA-fused only). The locked pipeline uses the calibrated path.
    """
    fused_list: list[float] = []
    obs_list: list[float] = []
    strata_list: list[str] = []

    for frame in frames:
        event_id = frame.event.event_id
        if event_id not in bma_priors:
            raise ValueError(f"missing BMA prior for event {event_id!r}")
        weights = bma_priors[event_id]
        for _ts, group in frame.records.groupby("timestamp"):
            present = {row["model_id"]: row["probability"] for _, row in group.iterrows()}
            usable = {m: weights[m] for m in present if m in weights}
            if not usable:
                continue
            wsum = sum(usable.values())
            if wsum == 0:
                continue
            fused = sum(usable[m] / wsum * present[m] for m in usable)
            fused_list.append(float(fused))
            obs_list.append(float(group["observed"].iloc[0]))
            strata_list.append(str(group["severity_stratum"].iloc[0]))

    fused_arr = np.asarray(fused_list, dtype=np.float64)
    obs_arr = np.asarray(obs_list, dtype=np.float64)
    if calibrator is not None:
        fused_arr = calibrator.transform(fused_arr, strata_list)  # type: ignore[arg-type]
    return fused_arr, obs_arr, strata_list


def fit_split_conformal(
    frames: Iterable[TrainingEventFrame],
    bma_priors: dict[str, dict[str, float]],
    *,
    calibrator: SeverityStratifiedCalibrator | None = None,
) -> SplitConformalRegressor:
    """Fit a marginal :class:`SplitConformalRegressor` on pooled samples.

    Args:
        frames: Training-event bundles.
        bma_priors: ``{event_id: {model_id: weight}}``.
        calibrator: Optional fitted stratified isotonic calibrator. The
            locked pipeline ALWAYS passes the calibrator (the calibrated
            probability stream is what downstream consumers see). The
            argument is optional to allow ablation tests.

    Returns:
        A fitted :class:`SplitConformalRegressor`.
    """
    fused, obs, _strata = _build_calibration_set(frames, bma_priors, calibrator)
    if fused.size == 0:
        raise ValueError("no samples available to fit split conformal regressor")
    regressor = SplitConformalRegressor()
    regressor.fit(fused, obs)
    logger.info("split conformal fit; n_cal=%d", regressor.n_calibration)
    return regressor


def fit_mondrian_conformal(
    frames: Iterable[TrainingEventFrame],
    bma_priors: dict[str, dict[str, float]],
    *,
    calibrator: SeverityStratifiedCalibrator | None = None,
) -> MondrianConformalRegressor:
    """Fit a per-stratum :class:`MondrianConformalRegressor`.

    Args:
        frames: Training-event bundles.
        bma_priors: ``{event_id: {model_id: weight}}``.
        calibrator: Optional fitted stratified isotonic calibrator (see
            :func:`fit_split_conformal` for the locked-pipeline note).

    Returns:
        A fitted :class:`MondrianConformalRegressor`.
    """
    fused, obs, strata = _build_calibration_set(frames, bma_priors, calibrator)
    if fused.size == 0:
        raise ValueError("no samples available to fit Mondrian conformal regressor")
    regressor = MondrianConformalRegressor()
    regressor.fit(fused, obs, strata)  # type: ignore[arg-type]
    logger.info("Mondrian conformal fit; counts=%s", regressor.per_stratum_counts())
    return regressor


__all__ = ["fit_mondrian_conformal", "fit_split_conformal"]
