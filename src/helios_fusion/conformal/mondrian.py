"""Mondrian (per-stratum) split conformal regression.

Implements the Mondrian variant of split conformal prediction (Vovk 2003)
stratified by Kp severity. Each stratum carries its own calibration
residual set and quantile, so coverage holds *per stratum* rather than only
marginally.

Per the proposal §2 Obj. 2:

    Severity-stratified validation across quiet, moderate, and extreme
    conditions (Kp-stratified bins) to prevent calibration collapse on the
    events that matter most.

Mondrian conformal is the canonical mechanism for stratified coverage in
the literature.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, get_args

import numpy as np
import numpy.typing as npt

from helios_fusion.conformal.split import SplitConformalRegressor
from helios_fusion.types import SeverityStratum

logger = logging.getLogger(__name__)

_STATE_SCHEMA: str = "helios-fusion-engine/conformal/mondrian/0.1.0"
_STRATA_TUPLE: tuple[SeverityStratum, ...] = get_args(SeverityStratum)


class MondrianConformalRegressor:
    """Per-stratum split conformal regressor.

    One :class:`SplitConformalRegressor` per stratum; calibration and
    interval generation route through the per-sample stratum label.

    Attributes:
        fitted: True iff every stratum has at least 1 calibration sample
            and has been fit.
    """

    def __init__(self) -> None:
        self._sub: dict[SeverityStratum, SplitConformalRegressor] = {
            s: SplitConformalRegressor() for s in _STRATA_TUPLE
        }
        self._fitted_strata: set[SeverityStratum] = set()

    @property
    def fitted(self) -> bool:
        """Whether all strata have been fitted."""
        return self._fitted_strata == set(_STRATA_TUPLE)

    def per_stratum_counts(self) -> dict[SeverityStratum, int]:
        """Number of calibration residuals per stratum."""
        return {s: self._sub[s].n_calibration for s in _STRATA_TUPLE}

    def fit(
        self,
        predictions: npt.ArrayLike,
        observed: npt.ArrayLike,
        severity_strata: Sequence[SeverityStratum],
    ) -> None:
        """Fit a separate residual quantile per stratum.

        Args:
            predictions: 1-D point predictions.
            observed: 1-D observed values, same length as ``predictions``.
            severity_strata: Stratum label per sample (same length as
                ``predictions``).

        Raises:
            ValueError: On shape mismatch, unknown stratum label, or a
                stratum with no calibration samples.
        """
        p = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(observed, dtype=np.float64)
        s = list(severity_strata)
        if p.shape != y.shape:
            raise ValueError(
                f"predictions and observed must have the same shape; got {p.shape} vs {y.shape}"
            )
        if p.shape[0] != len(s):
            raise ValueError(
                f"severity_strata length {len(s)} does not match "
                f"predictions/observed length {p.shape[0]}"
            )
        unknown = {label for label in s if label not in _STRATA_TUPLE}
        if unknown:
            raise ValueError(f"unknown severity stratum label(s): {sorted(unknown)}")

        for stratum in _STRATA_TUPLE:
            mask = np.array([lbl == stratum for lbl in s], dtype=bool)
            if not mask.any():
                raise ValueError(
                    f"stratum {stratum!r} has no calibration samples; "
                    "Mondrian conformal requires >= 1 sample per stratum"
                )
            self._sub[stratum].fit(p[mask], y[mask])
            self._fitted_strata.add(stratum)

        logger.info(
            "MondrianConformalRegressor fit; counts=%s",
            self.per_stratum_counts(),
        )

    def predict_interval(
        self,
        predictions: npt.ArrayLike,
        severity_strata: Sequence[SeverityStratum],
        alpha: float = 0.1,
    ) -> npt.NDArray[np.float64]:
        """Return per-sample prediction intervals using the matching stratum.

        Args:
            predictions: 1-D point predictions.
            severity_strata: Stratum label per sample (same length).
            alpha: Miscoverage level (same semantics as
                :meth:`SplitConformalRegressor.predict_interval`).

        Returns:
            Array of shape ``(n, 2)`` with columns ``[lower, upper]``.

        Raises:
            RuntimeError: If the regressor is not fitted.
            ValueError: On shape mismatch, unknown stratum, or invalid alpha.
        """
        if not self.fitted:
            raise RuntimeError("MondrianConformalRegressor must be fit before predict")
        p = np.asarray(predictions, dtype=np.float64)
        s = list(severity_strata)
        if p.shape[0] != len(s):
            raise ValueError(
                f"severity_strata length {len(s)} does not match predictions length {p.shape[0]}"
            )

        out = np.empty((p.size, 2), dtype=np.float64)
        for stratum in _STRATA_TUPLE:
            mask = np.array([lbl == stratum for lbl in s], dtype=bool)
            if not mask.any():
                continue
            sub_intervals = self._sub[stratum].predict_interval(p[mask], alpha)
            out[mask] = sub_intervals

        bad = [lbl for lbl in s if lbl not in _STRATA_TUPLE]
        if bad:
            raise ValueError(f"unknown severity stratum label(s): {sorted(set(bad))}")
        return out

    def to_state_dict(self) -> dict[str, Any]:
        """Serialise per-stratum sub-regressor state.

        Raises:
            RuntimeError: If the regressor is not fully fitted.
        """
        if not self.fitted:
            raise RuntimeError("cannot serialise unfitted MondrianConformalRegressor")
        return {
            "schema_version": _STATE_SCHEMA,
            "sub": {stratum: self._sub[stratum].to_state_dict() for stratum in _STRATA_TUPLE},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MondrianConformalRegressor:
        """Rehydrate from a state dict."""
        version = state.get("schema_version")
        if version != _STATE_SCHEMA:
            raise ValueError(f"unsupported schema_version {version!r}; expected {_STATE_SCHEMA!r}")
        instance = cls()
        for stratum in _STRATA_TUPLE:
            instance._sub[stratum] = SplitConformalRegressor.from_state_dict(state["sub"][stratum])
            instance._fitted_strata.add(stratum)
        return instance
