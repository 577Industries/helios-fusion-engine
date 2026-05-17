"""Best-individual-component-model baseline.

The kill-gate comparator (proposal §2 Obj. 3 success criterion, OSF
pre-registration §6) is the *best* single component model's HSS on the
hold-out, where "best" is identified by HSS on the hold-out using the
*same* binarisation threshold the fused engine uses. The fused HSS must
exceed this baseline by :math:`\\ge 15\\%` (relative).

This module isolates that comparator from the fusion machinery so the
kill-gate runner can ask for it directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from helios_fusion.eval.metrics import hss

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helios_fusion.types import ModelOutput

logger = logging.getLogger(__name__)


def best_individual_component_baseline(
    component_outputs: Sequence[ModelOutput],
    observed_binary: Sequence[int] | npt.ArrayLike,
    binary_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[str, float, tuple[float, float]]:
    """Identify the best single-component model by HSS on the held-out set.

    The inputs are grouped by ``model_id``. For each model, the predicted
    probabilities are binarised at ``binary_threshold`` and the HSS computed
    against ``observed_binary`` (restricted to the indices for which that
    model produced an output). The model with the highest HSS point estimate
    is returned, along with its HSS and bootstrap CI.

    Args:
        component_outputs: All upstream-model outputs over the held-out set.
            Multiple entries per model_id are expected — one per held-out
            event.
        observed_binary: Observed binary outcome per held-out event, in the
            same order as the events.
        binary_threshold: Threshold for binarising predicted probabilities.
        n_bootstrap: Bootstrap resamples for the CI on the winning model.
        seed: RNG seed.

    Returns:
        Tuple ``(best_model_id, hss_point, (ci_lo, ci_hi))``.

    Raises:
        ValueError: If ``component_outputs`` is empty, if any model has no
            observations corresponding to its outputs, or if the events
            cannot be aligned.

    Note:
        ``observed_binary`` is assumed indexed by event order matching the
        sorted unique event-timestamps in ``component_outputs``. Callers
        that need a different alignment should pre-stratify their inputs.
    """
    if not component_outputs:
        raise ValueError("component_outputs must not be empty")

    # Group component outputs by model_id, keeping event ordering by timestamp.
    by_model: dict[str, list[ModelOutput]] = defaultdict(list)
    for rec in component_outputs:
        by_model[rec.model_id].append(rec)

    # Build the canonical event order from sorted unique timestamps.
    all_timestamps = sorted({rec.timestamp for rec in component_outputs})
    event_index = {t: i for i, t in enumerate(all_timestamps)}

    obs = np.asarray(observed_binary, dtype=np.int_)
    if obs.shape != (len(all_timestamps),):
        raise ValueError(
            f"observed_binary must have length {len(all_timestamps)} "
            f"(number of unique event timestamps); got {obs.shape}"
        )

    best_id: str | None = None
    best_point: float = -float("inf")
    best_ci: tuple[float, float] = (float("nan"), float("nan"))

    for model_id, records in by_model.items():
        # Order records by event index.
        records_sorted = sorted(records, key=lambda r: event_index[r.timestamp])
        idx = np.array([event_index[r.timestamp] for r in records_sorted], dtype=np.int_)
        preds = np.array(
            [int(r.value >= binary_threshold) for r in records_sorted],
            dtype=np.int_,
        )
        obs_subset = obs[idx]
        point, ci = hss(preds, obs_subset, n_bootstrap=n_bootstrap, seed=seed)
        if not np.isnan(point) and point > best_point:
            best_id = model_id
            best_point = point
            best_ci = ci

    if best_id is None:
        raise ValueError("no component model produced a finite HSS; check input alignment")
    logger.info(
        "Best individual component model: %s (HSS=%.4f, 95%% CI=[%.4f, %.4f])",
        best_id,
        best_point,
        best_ci[0],
        best_ci[1],
    )
    return best_id, best_point, best_ci
