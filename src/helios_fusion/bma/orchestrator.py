"""Bayesian Model Averaging orchestrator.

Implements the BMA orchestration described in NASA SBIR proposal §2 Obj. 2:

    Bayesian Model Averaging (BMA) with dynamic weights conditioned on
    rolling 90-day verification skill per model.

The orchestrator does *not* hold trained weights — callers either supply them
explicitly at construction time or call :meth:`BMAOrchestrator.update_weights`
with a recent verification window. This separation is intentional: the public
framework (this repo) never ships with weights, which keeps the calibration
parameters and BMA priors in the private ``helios-fusion-internal`` companion.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import TYPE_CHECKING

from helios_fusion.bma.weights import (
    WeightPolicy,
    compute_skill_weights,
    renormalize_weights,
)
from helios_fusion.eval.metrics import hss
from helios_fusion.types import FusedOutput, LineageStep, ModelOutput

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class BMAOrchestrator:
    """Bayesian Model Averaging across upstream component models.

    The orchestrator stores per-model weights, applies them to a list of
    :class:`~helios_fusion.types.ModelOutput` instances (one per component
    model at the same timestamp), and emits a :class:`FusedOutput` with full
    lineage.

    Missing models are handled by exclusion: if a component model is absent
    from the input list for a given event, its weight is excluded and the
    remaining weights are renormalised. The exclusion is recorded in
    ``LineageStep.parameters['excluded_models']``.

    Args:
        weights: Optional initial weights mapping ``model_id`` to weight.
            If ``None``, weights must be set via :meth:`update_weights`
            before :meth:`fuse` can be called. Values do not need to sum to
            1; they will be renormalised at construction.
        weight_policy: Policy used by :meth:`update_weights`. Default
            ``"hss_clipped"`` matches the proposal's HSS-skill formulation.
        prediction_target: Human-readable name written to the emitted
            :class:`FusedOutput.prediction_target` field.
        value_units: Units string written to :class:`FusedOutput.value_units`.

    Raises:
        ValueError: If ``weights`` is provided and contains negative entries
            or sums to zero.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        weight_policy: WeightPolicy = "hss_clipped",
        prediction_target: str = "fused_value",
        value_units: str = "probability",
    ) -> None:
        self._weight_policy: WeightPolicy = weight_policy
        self._prediction_target = prediction_target
        self._value_units = value_units
        self._weights: dict[str, float] | None
        if weights is None:
            self._weights = None
        else:
            self._weights = renormalize_weights(weights)

    @property
    def weights(self) -> dict[str, float] | None:
        """Current normalised weights mapping; ``None`` if not yet set."""
        return None if self._weights is None else dict(self._weights)

    @property
    def weight_policy(self) -> WeightPolicy:
        """Policy used when :meth:`update_weights` computes skill weights."""
        return self._weight_policy

    def update_weights(
        self,
        verification_window: Sequence[tuple[ModelOutput, float]],
        *,
        binary_threshold: float = 0.5,
    ) -> None:
        """Recompute weights from a verification window.

        For each model, the rolling-window HSS is computed by binarising the
        model's probability outputs at ``binary_threshold`` and comparing to
        the observed truth value (also binarised at ``binary_threshold``).
        The resulting per-model HSS feeds :func:`compute_skill_weights`.

        Args:
            verification_window: Sequence of ``(ModelOutput, observed)``
                pairs. Each pair contributes one (predicted, observed) sample
                under that model's ``model_id``. Callers are expected to
                provide ~90 days of samples for production use (per
                proposal §2 Obj. 2), but no minimum is enforced here.
            binary_threshold: Threshold used to binarise both the predicted
                probability and the observed value. Default ``0.5``.

        Raises:
            ValueError: If ``verification_window`` is empty.
        """
        if not verification_window:
            raise ValueError("verification_window must not be empty")

        by_model: dict[str, list[tuple[float, float]]] = {}
        for record, observed in verification_window:
            by_model.setdefault(record.model_id, []).append((float(record.value), float(observed)))

        skill_by_model: dict[str, float] = {}
        for model_id, samples in by_model.items():
            preds = [p for p, _ in samples]
            obs = [o for _, o in samples]
            pred_bin = [int(p >= binary_threshold) for p in preds]
            obs_bin = [int(o >= binary_threshold) for o in obs]
            if len(set(obs_bin)) < 2 or len(set(pred_bin)) < 2:
                # HSS undefined on degenerate samples; fall back to 0 so the
                # weight clipping policy hands the model the epsilon floor.
                logger.warning(
                    "skill for model %s is degenerate "
                    "(constant predictions or constant observations); "
                    "falling back to skill=0.",
                    model_id,
                )
                skill_by_model[model_id] = 0.0
                continue
            point, _ = hss(pred_bin, obs_bin, n_bootstrap=0)
            # `hss` may return NaN on a perfectly degenerate confusion matrix.
            skill_by_model[model_id] = 0.0 if math.isnan(point) else point

        self._weights = compute_skill_weights(skill_by_model, policy=self._weight_policy)
        logger.info(
            "BMA weights updated from %d samples across %d model(s).",
            len(verification_window),
            len(by_model),
        )

    def fuse(self, outputs: Sequence[ModelOutput]) -> FusedOutput:
        """Fuse a list of upstream model outputs into one FusedOutput.

        Args:
            outputs: List of :class:`ModelOutput` to fuse. All entries must
                share a timestamp.

        Returns:
            A :class:`FusedOutput` carrying the weighted-average ``value``,
            the timestamp inherited from the inputs, and a single
            :class:`LineageStep` of type ``"bma_fuse"``.

        Raises:
            ValueError: If ``outputs`` is empty, if timestamps disagree, if
                weights have not been set, or if every input model is
                missing from the configured weights.
        """
        if not outputs:
            raise ValueError("outputs must not be empty")
        if self._weights is None:
            raise ValueError(
                "BMA weights not set; call update_weights or provide weights at construction time"
            )

        # Validate timestamp homogeneity.
        timestamps = {o.timestamp for o in outputs}
        if len(timestamps) != 1:
            raise ValueError(
                f"all outputs must share a timestamp; got {sorted(t.isoformat() for t in timestamps)}"
            )
        timestamp = next(iter(timestamps))

        # Identify which configured models are present in this event.
        present_ids = {o.model_id for o in outputs}
        configured_ids = set(self._weights.keys())
        usable = present_ids & configured_ids
        excluded = sorted(configured_ids - present_ids)

        if not usable:
            raise ValueError(
                "none of the supplied outputs match the configured weights; "
                f"outputs={sorted(present_ids)} configured={sorted(configured_ids)}"
            )

        # Renormalise weights over the usable subset (missing-model exclusion).
        subset_weights = renormalize_weights({m: self._weights[m] for m in usable})

        # Weighted average over the usable outputs. If a configured model
        # appears multiple times in `outputs` (operator error), keep the
        # last entry and log a warning.
        values_by_model: dict[str, ModelOutput] = {}
        for o in outputs:
            if o.model_id in usable:
                if o.model_id in values_by_model:
                    logger.warning(
                        "duplicate ModelOutput for model_id=%s at timestamp=%s; using last entry.",
                        o.model_id,
                        timestamp.isoformat(),
                    )
                values_by_model[o.model_id] = o

        fused_value = float(
            sum(subset_weights[m] * values_by_model[m].value for m in subset_weights)
        )

        fused_id = f"fused-{uuid.uuid4()}"
        step = LineageStep(
            transformation_id=f"bma-{uuid.uuid4()}",
            transformation_type="bma_fuse",
            input_ids=[values_by_model[m].id for m in subset_weights],
            output_ids=[fused_id],
            weight=None,
            parameters={
                "weight_policy": self._weight_policy,
                "weights_used": subset_weights,
                "excluded_models": excluded,
            },
        )

        return FusedOutput(
            id=fused_id,
            prediction_target=self._prediction_target,
            timestamp=timestamp,
            value=fused_value,
            value_units=self._value_units,
            conformal_interval=None,
            lineage=[step],
        )
