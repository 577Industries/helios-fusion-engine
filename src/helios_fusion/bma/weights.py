"""Pure functions for computing BMA weights from skill scores.

These helpers are deliberately stateless so they can be unit-tested in
isolation from the orchestrator. The orchestrator composes them with the
verification-window bookkeeping.

Skill-to-weight policies
------------------------

``"linear"``  — weights proportional to ``max(skill, 0)``. Simple, exposes
intuitive equal-weight degenerate case when all skills tie.

``"softmax"`` — weights ``exp(beta * skill)`` normalised. ``beta`` controls
sharpness. Always strictly positive, even when skill is negative.

``"hss_clipped"`` — like ``linear`` but with the floor at ``0`` and a small
``epsilon`` mixed in so unobserved models don't drop to zero immediately.
This is the proposal-default policy (§2 Obj. 2: "dynamic weights conditioned
on rolling 90-day verification skill per model"). HSS in
``[-1, 1]`` is clipped to ``[0, 1]`` then mixed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

WeightPolicy = Literal["linear", "softmax", "hss_clipped"]
"""Available weight-computation policies."""


def renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Renormalise a dict of weights so values sum to 1.

    Args:
        weights: Mapping of model_id to non-negative weight.

    Returns:
        A new mapping with the same keys, values summing to 1.0.

    Raises:
        ValueError: If the input is empty, contains negatives, or sums to
            zero (no model can be weighted).
    """
    if not weights:
        raise ValueError("weights mapping must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError("all weights must be non-negative")

    total = float(sum(weights.values()))
    if total == 0.0:
        raise ValueError("weights sum to zero; cannot renormalise (every model has zero weight)")
    return {k: float(v / total) for k, v in weights.items()}


def compute_skill_weights(
    skill_by_model: dict[str, float],
    policy: WeightPolicy = "hss_clipped",
    softmax_beta: float = 4.0,
    epsilon: float = 1e-3,
) -> dict[str, float]:
    """Map skill scores to BMA weights.

    The default policy is ``"hss_clipped"`` — clip negative HSS values to
    zero, mix in ``epsilon`` so models never receive exactly zero weight
    (preserving the option for a recovering model to gain weight back), then
    renormalise.

    Args:
        skill_by_model: Mapping of model_id to skill score. For HSS, the
            score is expected in ``[-1, 1]``; the policies still work on any
            real-valued skill.
        policy: One of :data:`WeightPolicy`.
        softmax_beta: Sharpness parameter for ``"softmax"``. Ignored for
            other policies.
        epsilon: Floor mixed into ``"hss_clipped"`` and ``"linear"``.

    Returns:
        A mapping of model_id to weight with values summing to 1.0.

    Raises:
        ValueError: If ``skill_by_model`` is empty or ``policy`` is unknown.
    """
    if not skill_by_model:
        raise ValueError("skill_by_model must not be empty")

    model_ids = list(skill_by_model.keys())
    scores: npt.NDArray[np.float64] = np.asarray(
        [skill_by_model[m] for m in model_ids], dtype=np.float64
    )

    raw: npt.NDArray[np.float64]
    if policy == "linear":
        raw = np.clip(scores, a_min=0.0, a_max=None) + epsilon
    elif policy == "softmax":
        # subtract max for numerical stability
        raw = np.exp(softmax_beta * (scores - scores.max()))
    elif policy == "hss_clipped":
        raw = np.clip(scores, a_min=0.0, a_max=1.0) + epsilon
    else:  # pragma: no cover - guarded by Literal at call sites
        raise ValueError(f"unknown weight policy: {policy!r}")

    total = float(raw.sum())
    if total == 0.0:  # pragma: no cover - epsilon prevents this in practice
        raise ValueError("computed weights sum to zero; check skill scores or epsilon")
    normalised = raw / total
    return {m: float(w) for m, w in zip(model_ids, normalised, strict=True)}
