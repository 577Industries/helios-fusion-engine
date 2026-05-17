"""Fit the severity-stratified isotonic calibrator on Table 3-1 samples.

This module pools BMA-fused probabilities across the seven training events
and fits one :class:`~helios_fusion.calibration.SeverityStratifiedCalibrator`
across the three Kp severity strata (``quiet`` / ``moderate`` / ``extreme``).

The proposal-locked pattern (§2 Obj. 2) is:

1. Compute per-event BMA fused probabilities.
2. Pool the (probability, observed, stratum) tuples across all training
   events.
3. Fit one isotonic regressor per stratum on that pool.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from helios_fusion.calibration.stratified import SeverityStratifiedCalibrator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from helios_fusion.training.load_table_3_1 import TrainingEventFrame

logger = logging.getLogger(__name__)


def fit_stratified_calibrators(
    frames: Iterable[TrainingEventFrame],
    bma_priors: dict[str, dict[str, float]],
) -> SeverityStratifiedCalibrator:
    """Fit per-stratum isotonic calibrators on pooled training-event samples.

    For each event we:

    1. Group rows by timestamp.
    2. Compute the BMA-fused probability at each timestamp using the
       event's fitted weight vector from ``bma_priors``.
    3. Record (fused_prob, observed, stratum) per timestamp.

    Then we pool across all events and fit the
    :class:`SeverityStratifiedCalibrator`.

    Args:
        frames: Training-event bundles from
            :func:`load_table_3_1` / :func:`load_all_training_events`.
        bma_priors: Map of ``event_id -> {model_id: weight}``, e.g. from
            :func:`helios_fusion.training.fit_bma_priors`.

    Returns:
        A fitted :class:`SeverityStratifiedCalibrator`.

    Raises:
        ValueError: If any stratum receives fewer than 2 pooled samples
            (isotonic regression requires >= 2 per stratum).
    """
    fused_probs: list[float] = []
    observed: list[float] = []
    strata: list[str] = []

    for frame in frames:
        event_id = frame.event.event_id
        if event_id not in bma_priors:
            raise ValueError(f"missing BMA prior for event {event_id!r}")
        weights = bma_priors[event_id]
        df = frame.records

        for _ts, group in df.groupby("timestamp"):
            # Weighted average over the models present at this timestamp.
            present = {row["model_id"]: row["probability"] for _, row in group.iterrows()}
            usable = {m: weights[m] for m in present if m in weights}
            if not usable:
                continue
            wsum = sum(usable.values())
            if wsum == 0:
                continue
            fused = sum(usable[m] / wsum * present[m] for m in usable)
            # Observed/stratum are constant across rows at this timestamp.
            obs_val = float(group["observed"].iloc[0])
            stratum_val = str(group["severity_stratum"].iloc[0])
            fused_probs.append(float(fused))
            observed.append(obs_val)
            strata.append(stratum_val)

    if not fused_probs:
        raise ValueError("no pooled samples generated; cannot fit calibrator")

    calibrator = SeverityStratifiedCalibrator()
    # SeverityStratifiedCalibrator.fit enforces >= 2 samples per stratum.
    calibrator.fit(
        np.asarray(fused_probs, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        strata,  # type: ignore[arg-type]
    )
    logger.info(
        "Stratified calibrator fit; n_pooled=%d; per-stratum counts=%s",
        len(fused_probs),
        calibrator.sample_counts,
    )
    return calibrator


__all__ = ["fit_stratified_calibrators"]
