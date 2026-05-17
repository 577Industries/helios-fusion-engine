"""Conformal prediction wrappers for continuous outputs.

Public surface:

* :class:`SplitConformalRegressor` — marginal split conformal residual
  intervals (Vovk et al. 2005).
* :class:`MondrianConformalRegressor` — Mondrian (per-stratum) conformal
  intervals (Vovk 2003), used for severity-stratified validation.

Design note (deviation rationale)
---------------------------------

The implementation is from-scratch, NOT a thin wrapper over ``mapie`` or
``crepes``. Reasons:

1. **Mondrian-per-Kp-stratum semantics**: neither ``mapie`` nor ``crepes``
   exposes Kp-bin Mondrian taxonomy as a first-class object; both require
   a sklearn estimator wrapper. The HELIOS framework operates on already-
   fused point estimates, not on a fittable estimator, so the sklearn
   contract is awkward.
2. **Determinism over generality**: the split-conformal residual quantile
   is a single ``np.quantile`` call with ``method="higher"`` to match the
   :math:`\\lceil (n+1)(1-\\alpha) \\rceil / n` finite-sample correction.
   Wrapping ``mapie`` to reach that exact quantile rule costs more code
   than just implementing it directly.
3. **Dependency surface**: ``mapie`` 0.7 pulls in scikit-learn and a
   transitive dependency on ``llvmlite`` via ``numba`` in some optional
   paths. Keeping conformal from-scratch shrinks the surface.

The from-scratch implementations are tested for empirical coverage against
the requested :math:`1 - \\alpha` level on synthetic data with known
distributions; see ``tests/test_conformal.py``.
"""

from __future__ import annotations

from helios_fusion.conformal.mondrian import MondrianConformalRegressor
from helios_fusion.conformal.split import SplitConformalRegressor

__all__ = [
    "MondrianConformalRegressor",
    "SplitConformalRegressor",
]
