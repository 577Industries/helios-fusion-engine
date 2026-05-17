"""Severity-stratified isotonic calibration.

Holds one :class:`~helios_fusion.calibration.IsotonicCalibrator` per Kp
severity stratum so reliability calibration can specialise across
``quiet`` / ``moderate`` / ``extreme`` regimes.

This is the proposal-default calibrator for the kill-gate run. See
proposal §2 Obj. 2:

    Severity-stratified validation across quiet, moderate, and extreme
    conditions (Kp-stratified bins) to prevent calibration collapse on the
    events that matter most.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, get_args

import numpy as np
import numpy.typing as npt

from helios_fusion.calibration.isotonic import IsotonicCalibrator
from helios_fusion.types import SeverityStratum

logger = logging.getLogger(__name__)

_STATE_SCHEMA: str = "helios-fusion-engine/calibration/stratified/0.1.0"
_STRATA_TUPLE: tuple[SeverityStratum, ...] = get_args(SeverityStratum)


class SeverityStratifiedCalibrator:
    """Per-stratum isotonic calibrator.

    Three independent :class:`IsotonicCalibrator` instances are fitted on
    the partition of the calibration set induced by the per-sample stratum
    labels. At transform time the same labels select which sub-calibrator
    handles each input.

    Attributes:
        fitted: True iff every sub-calibrator has been fitted.

    Note:
        Stratum ``"extreme"`` typically has fewer samples than ``"quiet"``.
        The :meth:`fit` method does NOT enforce a minimum-samples-per-stratum
        check — callers should monitor the per-stratum sample counts via the
        :attr:`sample_counts` property and decide whether the calibrator is
        trustworthy for their workload.
    """

    def __init__(self) -> None:
        self._calibrators: dict[SeverityStratum, IsotonicCalibrator] = {
            s: IsotonicCalibrator() for s in _STRATA_TUPLE
        }
        self._sample_counts: dict[SeverityStratum, int] = {s: 0 for s in _STRATA_TUPLE}
        self._fitted_strata: set[SeverityStratum] = set()

    @property
    def fitted(self) -> bool:
        """Whether all strata have at least been fit-attempted."""
        return self._fitted_strata == set(_STRATA_TUPLE)

    @property
    def sample_counts(self) -> dict[SeverityStratum, int]:
        """Number of samples used to fit each stratum's calibrator."""
        return dict(self._sample_counts)

    def fit(
        self,
        probs: npt.ArrayLike,
        observed: npt.ArrayLike,
        severity_strata: Sequence[SeverityStratum],
    ) -> None:
        """Fit one isotonic calibrator per stratum.

        Args:
            probs: 1-D array of probabilities in ``[0, 1]``.
            observed: 1-D array of observed outcomes (0/1 or frequencies),
                same length as ``probs``.
            severity_strata: Sequence of stratum labels, same length as
                ``probs``. Each label must be one of :data:`SeverityStratum`.

        Raises:
            ValueError: On shape mismatch, unknown stratum label, or a
                stratum with fewer than 2 samples (isotonic regression
                requires at least 2 points).
        """
        p = np.asarray(probs, dtype=np.float64)
        y = np.asarray(observed, dtype=np.float64)
        s = list(severity_strata)
        if p.shape != y.shape:
            raise ValueError(
                f"probs and observed must have the same shape; got {p.shape} vs {y.shape}"
            )
        if p.shape[0] != len(s):
            raise ValueError(
                f"severity_strata length {len(s)} does not match probs/observed length {p.shape[0]}"
            )

        unknown = {label for label in s if label not in _STRATA_TUPLE}
        if unknown:
            raise ValueError(f"unknown severity stratum label(s): {sorted(unknown)}")

        for stratum in _STRATA_TUPLE:
            mask = np.array([lbl == stratum for lbl in s], dtype=bool)
            count = int(mask.sum())
            self._sample_counts[stratum] = count
            if count < 2:
                raise ValueError(
                    f"stratum {stratum!r} has only {count} sample(s); "
                    "need >= 2 for isotonic regression"
                )
            self._calibrators[stratum].fit(p[mask], y[mask])
            self._fitted_strata.add(stratum)

        logger.info(
            "SeverityStratifiedCalibrator fit; counts=%s",
            self._sample_counts,
        )

    def transform(
        self,
        probs: npt.ArrayLike,
        severity_strata: Sequence[SeverityStratum],
    ) -> npt.NDArray[np.float64]:
        """Apply the per-stratum calibrator selected by each sample's label.

        Args:
            probs: 1-D array of probabilities in ``[0, 1]``.
            severity_strata: Stratum label per sample (same length as
                ``probs``).

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If the calibrator is not fully fitted.
            ValueError: On shape mismatch or unknown stratum label.
        """
        if not self.fitted:
            raise RuntimeError("SeverityStratifiedCalibrator must be fit before transform")
        p = np.asarray(probs, dtype=np.float64)
        s = list(severity_strata)
        if p.shape[0] != len(s):
            raise ValueError(
                f"severity_strata length {len(s)} does not match probs length {p.shape[0]}"
            )

        out = np.empty_like(p, dtype=np.float64)
        for stratum in _STRATA_TUPLE:
            mask = np.array([lbl == stratum for lbl in s], dtype=bool)
            if not mask.any():
                continue
            out[mask] = self._calibrators[stratum].transform(p[mask])

        # Sanity check for any sample that didn't match a known stratum.
        bad = [lbl for lbl in s if lbl not in _STRATA_TUPLE]
        if bad:
            raise ValueError(f"unknown severity stratum label(s): {sorted(set(bad))}")
        return out

    def to_state_dict(self) -> dict[str, Any]:
        """Serialise per-stratum sub-calibrator state to a JSON-friendly dict.

        Raises:
            RuntimeError: If the calibrator is not fully fitted.
        """
        if not self.fitted:
            raise RuntimeError("cannot serialise unfitted SeverityStratifiedCalibrator")
        return {
            "schema_version": _STATE_SCHEMA,
            "fitted": True,
            "sample_counts": dict(self._sample_counts),
            "calibrators": {
                stratum: self._calibrators[stratum].to_state_dict() for stratum in _STRATA_TUPLE
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> SeverityStratifiedCalibrator:
        """Rehydrate from a state dict."""
        version = state.get("schema_version")
        if version != _STATE_SCHEMA:
            raise ValueError(f"unsupported schema_version {version!r}; expected {_STATE_SCHEMA!r}")
        instance = cls()
        for stratum in _STRATA_TUPLE:
            sub = state["calibrators"][stratum]
            instance._calibrators[stratum] = IsotonicCalibrator.from_state_dict(sub)
            instance._fitted_strata.add(stratum)
        counts = state.get("sample_counts", {})
        for stratum in _STRATA_TUPLE:
            instance._sample_counts[stratum] = int(counts.get(stratum, 0))
        return instance
