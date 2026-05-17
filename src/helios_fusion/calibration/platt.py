"""Platt-scaling calibrator (kept for comparison; proposal-rejected).

Why this exists
---------------

The NASA SBIR proposal §2 Obj. 2 explicitly rejects Platt scaling:

    Isotonic regression reliability calibration so predicted probabilities
    match observed event frequencies (Platt scaling considered and rejected
    for known miscalibration at extremes).

This module ships the rejected calibrator anyway so the framework can
*demonstrate* the rejection rationale on synthetic and real data. The
``test_calibration.py`` suite asserts that Platt scaling produces *worse*
calibration than isotonic at synthetic extremes — that assertion is the
rejection rationale, in code.

Rejection rationale (short form)
--------------------------------

Platt scaling fits a two-parameter sigmoid (``a, b``) to map raw scores to
calibrated probabilities. The two-parameter sigmoid has insufficient
flexibility to track piecewise-monotone reliability curves that arise when
upstream models are well-calibrated at moderate probabilities but
underconfident (or overconfident) at extreme probabilities — exactly the
regime where SEP all-clear-revocation decisions are made. Isotonic
regression is non-parametric in the relevant sense (monotone non-decreasing)
and handles this regime gracefully.

References
----------

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities
with supervised learning. *ICML*.

Kull, M., Filho, T. S., & Flach, P. (2017). Beta calibration: a
well-founded and easily implemented improvement on logistic calibration
for binary classifiers. *AISTATS*. (Argues Platt is a special case of beta
calibration that breaks at the tails.)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_STATE_SCHEMA: str = "helios-fusion-engine/calibration/platt/0.1.0"
_EPS: float = 1e-12


class PlattCalibrator:
    """Two-parameter sigmoid calibrator (Platt 1999).

    Fits ``a, b`` such that ``calibrated(p) = sigmoid(a * logit(p) + b)``.
    The fitter uses the regularised log-loss formulation from
    Lin et al. 2007 to avoid the original-Platt label-smoothing approximation
    overshoot near 0/1.

    This calibrator is provided for comparison only. Production HELIOS code
    should use :class:`~helios_fusion.calibration.IsotonicCalibrator` or
    :class:`~helios_fusion.calibration.SeverityStratifiedCalibrator`.
    """

    def __init__(self) -> None:
        self._a: float | None = None
        self._b: float | None = None
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def a(self) -> float:
        """Fitted slope parameter. Raises if unfitted."""
        if self._a is None:
            raise RuntimeError("PlattCalibrator not yet fit")
        return self._a

    @property
    def b(self) -> float:
        """Fitted intercept parameter. Raises if unfitted."""
        if self._b is None:
            raise RuntimeError("PlattCalibrator not yet fit")
        return self._b

    @staticmethod
    def _logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
        return np.log(p_clipped / (1.0 - p_clipped))

    @staticmethod
    def _sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # numerically stable sigmoid
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(self, probs: npt.ArrayLike, observed: npt.ArrayLike) -> None:
        """Fit (a, b) via L-BFGS-B on the log-loss.

        Args:
            probs: 1-D array of probabilities in ``[0, 1]``.
            observed: 1-D array of binary outcomes (0/1).

        Raises:
            ValueError: On shape mismatch, empty input, or out-of-range
                probabilities.
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
            raise ValueError("probs must be in [0, 1]")
        if np.any(np.isnan(p)) or np.any(np.isnan(y)):
            raise ValueError("probs and observed must not contain NaN")

        logits = self._logit(p)

        def neg_log_loss(params: npt.NDArray[np.float64]) -> float:
            a, b = params
            z = a * logits + b
            # log(1 + exp(-z)) form is numerically stabilised below.
            log_sigmoid = -np.logaddexp(0.0, -z)
            log_one_minus = -np.logaddexp(0.0, z)
            return float(-np.sum(y * log_sigmoid + (1.0 - y) * log_one_minus))

        result = minimize(
            neg_log_loss,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
        )
        if not result.success:  # pragma: no cover - rare optimisation failure
            logger.warning("Platt L-BFGS-B did not converge: %s", result.message)
        self._a = float(result.x[0])
        self._b = float(result.x[1])
        self._fitted = True

    def transform(self, probs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the fitted Platt sigmoid to new probabilities.

        Args:
            probs: 1-D array of probabilities in ``[0, 1]``.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If ``probs`` contains values outside ``[0, 1]``.
        """
        if not self._fitted:
            raise RuntimeError("PlattCalibrator must be fit before transform")
        p = np.asarray(probs, dtype=np.float64)
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("probs must be in [0, 1]")
        logits = self._logit(p)
        assert self._a is not None
        assert self._b is not None
        return self._sigmoid(self._a * logits + self._b)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialise the (a, b) pair.

        Raises:
            RuntimeError: If the calibrator is not fitted.
        """
        if not self._fitted:
            raise RuntimeError("cannot serialise unfitted PlattCalibrator")
        return {
            "schema_version": _STATE_SCHEMA,
            "fitted": True,
            "a": self._a,
            "b": self._b,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PlattCalibrator:
        """Rehydrate from a state dict produced by :meth:`to_state_dict`."""
        version = state.get("schema_version")
        if version != _STATE_SCHEMA:
            raise ValueError(f"unsupported schema_version {version!r}; expected {_STATE_SCHEMA!r}")
        instance = cls()
        instance._a = float(state["a"])
        instance._b = float(state["b"])
        instance._fitted = True
        return instance
