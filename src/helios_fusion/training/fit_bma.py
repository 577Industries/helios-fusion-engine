"""Fit per-event BMA weight vectors.

Uses the canonical
:func:`helios_fusion.bma.weights.compute_skill_weights` formulation (the
proposal-default ``hss_clipped`` policy). The "rolling 90-day window"
described in proposal §2 Obj. 2 is applied here per training event: each
event window (+/-5 days) is treated as the verification window for that
event's BMA-weight assignment.

The output is a mapping ``{event_id: {model_id: weight}}`` where each
inner mapping sums to 1.0.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from helios_fusion.bma.weights import compute_skill_weights
from helios_fusion.eval.metrics import hss

if TYPE_CHECKING:
    from collections.abc import Iterable

    from helios_fusion.training.load_table_3_1 import TrainingEventFrame

logger = logging.getLogger(__name__)

#: Binary threshold for HSS computation. Locked at 0.5 (proposal §2 Obj. 2).
_BINARY_THRESHOLD: float = 0.5


def fit_bma_priors(
    frames: Iterable[TrainingEventFrame],
    *,
    binary_threshold: float = _BINARY_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Fit BMA weights per event.

    For each event frame, compute per-model HSS over the event window,
    clip to [0, 1], mix in an epsilon floor, and renormalise via
    :func:`compute_skill_weights` with the locked ``hss_clipped`` policy.

    Args:
        frames: Iterable of :class:`TrainingEventFrame` from
            :func:`load_table_3_1` / :func:`load_all_training_events`.
        binary_threshold: Threshold used to binarise probabilities and
            observations for the HSS computation. Default 0.5 (locked).

    Returns:
        ``{event_id: {model_id: weight}}`` -- one normalised weight vector
        per event. Each inner mapping sums to 1.0.

    Raises:
        ValueError: If any event frame has zero rows or no usable models.
    """
    out: dict[str, dict[str, float]] = {}
    for frame in frames:
        df = frame.records
        if df.empty:
            raise ValueError(f"event {frame.event.event_id!r} has no rows")

        skill_by_model: dict[str, float] = {}
        for model_id in frame.component_models:
            sub = df[df["model_id"] == model_id]
            if sub.empty:
                logger.warning(
                    "event=%s model=%s has no rows; skipping",
                    frame.event.event_id,
                    model_id,
                )
                continue
            preds = (sub["probability"].to_numpy() >= binary_threshold).astype(int).tolist()
            obs = (sub["observed"].to_numpy() >= binary_threshold).astype(int).tolist()
            if len(set(obs)) < 2 or len(set(preds)) < 2:
                # HSS is degenerate; fall back to 0 (epsilon floor will
                # rescue the model from zero weight).
                skill_by_model[model_id] = 0.0
                continue
            point, _ = hss(preds, obs, n_bootstrap=0)
            skill_by_model[model_id] = 0.0 if math.isnan(point) else point

        if not skill_by_model:
            raise ValueError(
                f"event {frame.event.event_id!r} produced no usable model skill scores"
            )

        weights = compute_skill_weights(skill_by_model, policy="hss_clipped")
        out[frame.event.event_id] = weights
        logger.info(
            "BMA weights fit for event=%s; top model %s @ %.4f",
            frame.event.event_id,
            max(weights, key=lambda k: weights[k]),
            max(weights.values()),
        )
    return out


__all__ = ["fit_bma_priors"]
