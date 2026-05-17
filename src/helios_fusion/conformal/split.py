"""Split conformal regression for continuous outputs.

Implements marginal split conformal prediction (Vovk et al. 2005) using
absolute-residual scores. Provides a finite-sample-valid two-sided
prediction interval at any user-chosen coverage level ``1 - alpha``.

Quantile rule
-------------

Given a calibration set of size ``n`` with absolute residuals
``r_1, ..., r_n``, the split-conformal half-width at level ``1 - alpha`` is

.. math::

    q = \\text{quantile}\\left(\\{r_1, \\ldots, r_n\\},\\,
        \\frac{\\lceil (n + 1)(1 - \\alpha) \\rceil}{n}\\right)

with ``method="higher"`` for the quantile estimator. This is the standard
finite-sample-corrected split-conformal rule. The interval for a new
prediction :math:`\\hat y` is :math:`[\\hat y - q,\\ \\hat y + q]`.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_STATE_SCHEMA: str = "helios-fusion-engine/conformal/split/0.1.0"


class SplitConformalRegressor:
    """Marginal split conformal regressor.

    Fit on a calibration set ``(predictions, observed)`` to learn the
    finite-sample residual quantile. Predict prediction intervals on new
    point predictions at any requested coverage level.

    The class is stateless except for the calibration residual array; it
    does NOT itself produce point predictions. Callers feed it predictions
    from any upstream model (typically the fused BMA output).

    Attributes:
        fitted: True iff :meth:`fit` has been called.
        n_calibration: Number of calibration residuals retained.
    """

    def __init__(self) -> None:
        self._residuals: npt.NDArray[np.float64] | None = None

    @property
    def fitted(self) -> bool:
        """Whether the regressor has been fitted."""
        return self._residuals is not None

    @property
    def n_calibration(self) -> int:
        """Number of calibration residuals."""
        if self._residuals is None:
            return 0
        return int(self._residuals.size)

    def fit(
        self,
        predictions: npt.ArrayLike,
        observed: npt.ArrayLike,
    ) -> None:
        """Compute and store absolute residuals.

        Args:
            predictions: 1-D point predictions on the calibration set.
            observed: 1-D observed values, same length as ``predictions``.

        Raises:
            ValueError: On shape mismatch or empty input.
        """
        p = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(observed, dtype=np.float64)
        if p.shape != y.shape:
            raise ValueError(
                f"predictions and observed must have the same shape; got {p.shape} vs {y.shape}"
            )
        if p.size == 0:
            raise ValueError("predictions/observed must be non-empty")
        if np.any(np.isnan(p)) or np.any(np.isnan(y)):
            raise ValueError("predictions and observed must not contain NaN")
        self._residuals = np.abs(p - y).astype(np.float64)
        logger.info(
            "SplitConformalRegressor fit; n_calibration=%d",
            self._residuals.size,
        )

    def _quantile(self, alpha: float) -> float:
        if self._residuals is None:
            raise RuntimeError("SplitConformalRegressor must be fit first")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        n = self._residuals.size
        # Finite-sample-corrected quantile fraction; clip to 1.0 if requested
        # coverage cannot be reached with the calibration size (n small).
        frac = math.ceil((n + 1) * (1.0 - alpha)) / n
        if frac >= 1.0:
            return float(self._residuals.max())
        return float(np.quantile(self._residuals, frac, method="higher"))

    def predict_interval(
        self,
        predictions: npt.ArrayLike,
        alpha: float = 0.1,
    ) -> npt.NDArray[np.float64]:
        """Return prediction intervals for new point predictions.

        Args:
            predictions: 1-D point predictions.
            alpha: Miscoverage level. The returned interval targets coverage
                ``1 - alpha``. Default ``0.1`` (90% intervals).

        Returns:
            Array of shape ``(n, 2)`` with columns ``[lower, upper]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If ``alpha`` is outside ``(0, 1)``.
        """
        q = self._quantile(alpha)
        p = np.asarray(predictions, dtype=np.float64)
        out = np.empty((p.size, 2), dtype=np.float64)
        out[:, 0] = p - q
        out[:, 1] = p + q
        return out

    def to_state_dict(self) -> dict[str, Any]:
        """Serialise residuals to a JSON-friendly dict.

        Raises:
            RuntimeError: If the regressor is not fitted.
        """
        if self._residuals is None:
            raise RuntimeError("cannot serialise unfitted SplitConformalRegressor")
        return {
            "schema_version": _STATE_SCHEMA,
            "residuals": self._residuals.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> SplitConformalRegressor:
        """Rehydrate from a state dict."""
        version = state.get("schema_version")
        if version != _STATE_SCHEMA:
            raise ValueError(f"unsupported schema_version {version!r}; expected {_STATE_SCHEMA!r}")
        instance = cls()
        instance._residuals = np.asarray(state["residuals"], dtype=np.float64)
        return instance
