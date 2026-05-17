"""Sprint C-Training: fit BMA priors, isotonic calibrators, and conformal residuals.

This subpackage produces the **trained artifacts** the kill-gate hold-out
evaluation will consume. The artifacts themselves live in the private
``helios-fusion-internal/weights/`` companion repo; this subpackage contains
only the reproducible code paths that produce them.

Public entry points
-------------------

* :data:`TRAINING_EVENTS` -- the seven Table 3-1 training events.
* :func:`load_table_3_1` -- pull Scoreboard A/B/C data for a single event.
* :func:`load_all_training_events` -- pull all seven events in parallel.
* :func:`fit_bma_priors` -- fit per-event BMA weight vectors using the
  rolling-90-day skill formulation already in
  :func:`helios_fusion.bma.weights.compute_skill_weights`.
* :func:`fit_stratified_calibrators` -- fit the
  :class:`~helios_fusion.calibration.SeverityStratifiedCalibrator` on the
  training events' pooled samples.
* :func:`fit_split_conformal` / :func:`fit_mondrian_conformal` -- fit the
  split-conformal and Mondrian-conformal calibration residual sets.
* :func:`run_full_training` -- end-to-end orchestrator that runs the four
  fits in sequence and returns a :class:`TrainingArtifacts` bundle.

Critical-constraint reminders (see
``helios-program/specs/2026-05-17-Sprint-C-Training-spec.md``):

* The seven events are locked. Hyperparameters are locked. Do NOT retune.
* This subpackage does NOT execute the hold-out evaluation. That's gated on
  OSF pre-registration filing per the kill-gate discipline.
"""

from __future__ import annotations

from helios_fusion.training.fit_bma import fit_bma_priors
from helios_fusion.training.fit_conformal import fit_mondrian_conformal, fit_split_conformal
from helios_fusion.training.fit_isotonic import fit_stratified_calibrators
from helios_fusion.training.load_table_3_1 import (
    DEFAULT_COMPONENT_MODELS,
    DEFAULT_COMPONENT_MODELS_LEGACY,
    EMPIRICAL_ISWA_COVERAGE,
    TRAINING_EVENTS,
    TrainingEvent,
    TrainingEventFrame,
    load_all_training_events,
    load_table_3_1,
)
from helios_fusion.training.pipeline import TrainingArtifacts, run_full_training
from helios_fusion.training.swpc_sep_archive import event_truth_labels, fetch_sep_event_list

__all__ = [
    "DEFAULT_COMPONENT_MODELS",
    "DEFAULT_COMPONENT_MODELS_LEGACY",
    "EMPIRICAL_ISWA_COVERAGE",
    "TRAINING_EVENTS",
    "TrainingArtifacts",
    "TrainingEvent",
    "TrainingEventFrame",
    "event_truth_labels",
    "fetch_sep_event_list",
    "fit_bma_priors",
    "fit_mondrian_conformal",
    "fit_split_conformal",
    "fit_stratified_calibrators",
    "load_all_training_events",
    "load_table_3_1",
    "run_full_training",
]
