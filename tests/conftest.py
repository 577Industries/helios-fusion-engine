"""Synthetic-data fixtures for the framework test suite.

The fixtures produce five "component model" probability streams with known
biases plus a synthetic ground truth and Kp series. They are deterministic
(seeded) so tests can assert exact numbers.

Component-model biases (locked across the suite):

* ``well_calibrated``   — true posterior + small noise
* ``underconfident``    — true posterior pulled toward 0.5
* ``overconfident``     — true posterior pushed away from 0.5
* ``severity_biased_lo`` — well-calibrated on quiet/moderate; overconfident on extreme
* ``severity_biased_hi`` — well-calibrated on moderate/extreme; overconfident on quiet
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from helios_fusion.stratification import assign_severity_stratum
from helios_fusion.types import ModelOutput, SeverityStratum

if TYPE_CHECKING:
    import numpy.typing as npt

_SYNTH_SEED: int = 20260517  # YYYYMMDD; locked for reproducibility


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic RNG for the whole test session."""
    return np.random.default_rng(_SYNTH_SEED)


@pytest.fixture(scope="session")
def synthetic_kp_series(rng: np.random.Generator) -> np.ndarray:
    """Realistic Kp severity distribution.

    Marginal mass approx: 70% quiet, 25% moderate, 5% extreme. We sample
    from a Beta-scaled mixture so the synthetic Kp series carries enough
    extreme-stratum samples for Mondrian conformal tests (>= 1 per stratum)
    while reflecting climatological skew.
    """
    n = 800
    # Component 1: quiet — Beta(2, 5) scaled to [0, 3]
    quiet = rng.beta(2.0, 5.0, size=int(n * 0.70)) * 3.0
    # Component 2: moderate — Beta(2, 2) scaled to [3.5, 6]
    moderate = 3.5 + rng.beta(2.0, 2.0, size=int(n * 0.25)) * 2.5
    # Component 3: extreme — Beta(2, 2) scaled to [7, 9]
    extreme = 7.0 + rng.beta(2.0, 2.0, size=int(n * 0.05)) * 2.0
    arr = np.concatenate([quiet, moderate, extreme])
    rng.shuffle(arr)
    return arr[:n]


@pytest.fixture(scope="session")
def synthetic_strata(synthetic_kp_series: np.ndarray) -> list[SeverityStratum]:
    """Stratum labels derived from :data:`synthetic_kp_series`."""
    return [assign_severity_stratum(float(k)) for k in synthetic_kp_series]


@pytest.fixture(scope="session")
def synthetic_truth(synthetic_kp_series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Binary ground truth driven by a logistic of Kp.

    The "event" is more likely at high Kp, matching the proposal §2 Obj. 3
    SEP all-clear-revocation threshold structure where extreme-Kp conditions
    drive event rate. Specifically:

        P(event) = sigmoid(0.9 * (Kp - 4.5))

    so quiet ⇒ ~10% event rate, moderate ⇒ ~50%, extreme ⇒ ~95%.
    """
    p = 1.0 / (1.0 + np.exp(-0.9 * (synthetic_kp_series - 4.5)))
    u = rng.random(p.shape)
    return (u < p).astype(np.float64)


@pytest.fixture(scope="session")
def synthetic_true_posterior(synthetic_kp_series: np.ndarray) -> np.ndarray:
    """The Bayes-optimal probability used to generate the truth."""
    return 1.0 / (1.0 + np.exp(-0.9 * (synthetic_kp_series - 4.5)))


def _well_calibrated(p_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(p_true + rng.normal(0.0, 0.02, p_true.shape), 0.0, 1.0)


def _underconfident(p_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Pull every probability toward 0.5 by 35%, then small noise.
    p = 0.5 + 0.65 * (p_true - 0.5)
    return np.clip(p + rng.normal(0.0, 0.02, p.shape), 0.0, 1.0)


def _overconfident(p_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Push probabilities away from 0.5; clip at 0.02/0.98 to avoid logit overflow.
    p = 0.5 + 1.55 * (p_true - 0.5)
    return np.clip(p + rng.normal(0.0, 0.02, p.shape), 0.02, 0.98)


def _severity_biased(
    p_true: np.ndarray,
    strata: list[SeverityStratum],
    rng: np.random.Generator,
    bias_in_stratum: SeverityStratum,
) -> np.ndarray:
    """Apply a strong directional bias only when stratum matches.

    The bias pattern is asymmetric in p: at low p_true the affected stratum
    overcalls (predicts high p), at high p_true it undercalls. This
    isotonic-irrecoverable pattern is the canonical motivation for a
    severity-stratified calibrator — an unstratified isotonic regression
    that pools across strata averages this with the well-calibrated mass and
    leaves residual miscalibration on the biased stratum.
    """
    base = _well_calibrated(p_true, rng)
    # Strong asymmetric bias on the chosen stratum: shift up at low p, down at high p.
    # This produces a NON-monotone observed-vs-predicted reliability curve
    # in the biased stratum, which a single global isotonic regression
    # cannot perfectly fit when other strata dominate the calibration set.
    biased = np.clip(0.4 + 0.3 * np.sin(np.pi * p_true) + 0.2 * p_true, 0.0, 1.0)
    biased = biased + rng.normal(0.0, 0.02, biased.shape)
    biased = np.clip(biased, 0.02, 0.98)
    out = base.copy()
    for i, s in enumerate(strata):
        if s == bias_in_stratum:
            out[i] = biased[i]
    return out


@pytest.fixture(scope="session")
def synthetic_component_probs(
    synthetic_true_posterior: np.ndarray,
    synthetic_strata: list[SeverityStratum],
) -> dict[str, np.ndarray]:
    """Five synthetic component-model probability streams.

    Returns a dict keyed by ``model_id``.
    """
    # New RNG so this fixture is independent of any callers that consumed
    # the session ``rng`` before us.
    local_rng = np.random.default_rng(_SYNTH_SEED ^ 0x1234)
    p_true = synthetic_true_posterior
    return {
        "well_calibrated": _well_calibrated(p_true, local_rng),
        "underconfident": _underconfident(p_true, local_rng),
        "overconfident": _overconfident(p_true, local_rng),
        "severity_biased_extreme": _severity_biased(p_true, synthetic_strata, local_rng, "extreme"),
        "severity_biased_quiet": _severity_biased(p_true, synthetic_strata, local_rng, "quiet"),
    }


def _make_model_outputs(
    model_id: str,
    probs: npt.NDArray[np.float64],
    strata: list[SeverityStratum],
) -> list[ModelOutput]:
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        ModelOutput(
            id=f"{model_id}-{i}",
            model_id=model_id,
            timestamp=base_time + timedelta(hours=i),
            value=float(p),
            value_units="probability",
            severity_stratum=strata[i],
        )
        for i, p in enumerate(probs)
    ]


@pytest.fixture(scope="session")
def synthetic_model_outputs(
    synthetic_component_probs: dict[str, np.ndarray],
    synthetic_strata: list[SeverityStratum],
) -> dict[str, list[ModelOutput]]:
    """Five streams as :class:`ModelOutput` lists for orchestrator-level tests."""
    return {
        model_id: _make_model_outputs(model_id, probs, synthetic_strata)
        for model_id, probs in synthetic_component_probs.items()
    }


@pytest.fixture(scope="session")
def hand_computed_hss_case() -> dict[str, object]:
    """Hand-computed HSS example for Donaldson-1975 verification.

    Contingency table::
        a=12 hits, b=4 false alarms, c=2 misses, d=22 correct rejections
        n = 40, expected HSS = 0.764705882... (within 1e-9).
    """
    predicted = [1] * 12 + [1] * 4 + [0] * 2 + [0] * 22
    observed = [1] * 12 + [0] * 4 + [1] * 2 + [0] * 22
    a, b, c, d = 12, 4, 2, 22
    num = 2 * (a * d - b * c)
    den = (a + c) * (c + d) + (a + b) * (b + d)
    return {"predicted": predicted, "observed": observed, "expected_hss": num / den}
