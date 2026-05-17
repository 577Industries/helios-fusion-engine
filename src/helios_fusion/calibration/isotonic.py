"""Isotonic-regression reliability calibrator.

Implements the proposal-default reliability calibrator (NASA SBIR §2 Obj. 2):

    Isotonic regression reliability calibration so predicted probabilities
    match observed event frequencies (Platt scaling considered and rejected
    for known miscalibration at extremes).

This is a thin, typed wrapper around
:class:`sklearn.isotonic.IsotonicRegression` with explicit state-dict
serialisation. Generic binary-object persistence is intentionally NOT part
of the public API; the state dict is JSON-friendly and the recommended
on-disk path is JSON or ``joblib`` with version pinning.

State-dict shape
----------------

The shape is stable across patch versions of this package; a
``schema_version`` field carries the contract version.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

_STATE_SCHEMA: str = "helios-fusion-engine/calibration/isotonic/0.1.0"


class IsotonicCalibrator:
    """Isotonic-regression probability calibrator.

    The calibrator learns a monotone non-decreasing map from predicted
    probability to calibrated probability. The mapping is fitted on a
    held-out reliability set and is *not* a function of any other input —
    in particular, do not fit on the training set used to fit the upstream
    model, or on the same data used to compute conformal residuals
    downstream.

    The wrapped scikit-learn estimator is configured with
    ``out_of_bounds="clip"`` so calibrated outputs are always in ``[0, 1]``
    even at extrapolation. ``y_min=0`` and ``y_max=1`` are explicit because
    the wrapper is probability-typed.

    Attributes:
        fitted: True iff :meth:`fit` has been called successfully.
    """

    def __init__(self) -> None:
        self._regressor = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    def fit(
        self,
        probs: npt.ArrayLike,
        observed: npt.ArrayLike,
    ) -> None:
        """Fit the isotonic regression on a calibration set.

        Args:
            probs: 1-D array of predicted probabilities in ``[0, 1]``.
            observed: 1-D array of observed outcomes (0/1 or probability
                frequencies). Must be the same length as ``probs``.

        Raises:
            ValueError: If ``probs`` and ``observed`` have mismatched shapes
                or empty input, or probabilities fall outside ``[0, 1]``.
        """
        p = np.asarray(probs, dtype=np.float64)
        y = np.asarray(observed, dtype=np.float64)
        if p.shape != y.shape:
            raise ValueError(
                f"probs and observed must have the same shape; got {p.shape} vs {y.shape}"
            )
        if p.size == 0:
            raise ValueError("probs/observed must be non-empty")
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("probs must be in [0, 1]; found values outside the range")
        if np.any(np.isnan(p)) or np.any(np.isnan(y)):
            raise ValueError("probs and observed must not contain NaN")

        self._regressor.fit(p, y)
        self._fitted = True
        logger.info(
            "IsotonicCalibrator fit on %d samples; knots=%d",
            p.size,
            len(self._regressor.X_thresholds_),
        )

    def transform(self, probs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the fitted calibration to new probabilities.

        Args:
            probs: 1-D array of probabilities in ``[0, 1]``.

        Returns:
            Calibrated probabilities clipped to ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If ``probs`` contains values outside ``[0, 1]`` or
                NaN.
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        p = np.asarray(probs, dtype=np.float64)
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("probs must be in [0, 1]; found values outside the range")
        if np.any(np.isnan(p)):
            raise ValueError("probs must not contain NaN")
        result = self._regressor.predict(p)
        return np.asarray(result, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Serialisation (explicit state dict; no opaque binary blobs)
    # ------------------------------------------------------------------ #
    def to_state_dict(self) -> dict[str, Any]:
        """Serialise the calibrator state to a JSON-friendly dict.

        Returns:
            A dict containing ``schema_version``, ``fitted``, and (if
            fitted) the isotonic knot positions and values as plain lists.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("cannot serialise unfitted IsotonicCalibrator")
        return {
            "schema_version": _STATE_SCHEMA,
            "fitted": True,
            "x_thresholds": self._regressor.X_thresholds_.tolist(),
            "y_thresholds": self._regressor.y_thresholds_.tolist(),
            "y_min": float(self._regressor.y_min),
            "y_max": float(self._regressor.y_max),
            "increasing": True,
            "out_of_bounds": self._regressor.out_of_bounds,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> IsotonicCalibrator:
        """Rehydrate a calibrator from a state dict.

        Args:
            state: A dict previously produced by :meth:`to_state_dict`.

        Returns:
            A fitted :class:`IsotonicCalibrator`.

        Raises:
            ValueError: If the schema version is unrecognised.
        """
        version = state.get("schema_version")
        if version != _STATE_SCHEMA:
            raise ValueError(f"unsupported schema_version {version!r}; expected {_STATE_SCHEMA!r}")
        instance = cls()
        x = np.asarray(state["x_thresholds"], dtype=np.float64)
        y = np.asarray(state["y_thresholds"], dtype=np.float64)
        instance._regressor.X_thresholds_ = x
        instance._regressor.y_thresholds_ = y
        instance._regressor.X_min_ = float(x.min())
        instance._regressor.X_max_ = float(x.max())
        # scikit-learn uses an internal f_ interpolator built lazily;
        # rebuild it explicitly so .predict works without a prior .fit.
        instance._regressor._build_f(x, y)
        instance._fitted = True
        return instance
