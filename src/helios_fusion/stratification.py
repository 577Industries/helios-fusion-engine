"""Severity stratification utilities.

The Kp bin edges are pre-registered in the OSF template (`§7. Severity strata`)
and are NOT a tuning parameter. Changing this assignment after pre-registration
invalidates the kill-gate analysis.

Definitions (locked):
    * ``quiet``    — Kp 0.0 to 3.0 inclusive
    * ``moderate`` — Kp 4.0 to 6.0 inclusive
    * ``extreme``  — Kp 7.0 to 9.0 inclusive

Kp is reported on the 28-step quasi-logarithmic scale 0o, 0+, 1-, 1o, 1+, ...,
9o; commonly digitised as a float in [0, 9] with thirds (e.g. 5.33 ≡ "5+").
The bin edges below treat the boundary value as belonging to the *lower*
stratum (i.e. ``Kp = 3.0`` is ``quiet``, ``Kp = 4.0`` is ``moderate``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, get_args

from helios_fusion.types import ModelOutput, SeverityStratum

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_STRATA_TUPLE: tuple[SeverityStratum, ...] = get_args(SeverityStratum)

# Boundary policy: Kp == cutoff belongs to the *lower* stratum.
_QUIET_MAX: float = 3.0
_MODERATE_MAX: float = 6.0
_KP_FLOOR: float = 0.0
_KP_CEILING: float = 9.0


def assign_severity_stratum(kp: float) -> SeverityStratum:
    """Return the severity stratum for a Kp value.

    Args:
        kp: Kp index value, expected in ``[0, 9]``.

    Returns:
        The locked stratum label.

    Raises:
        ValueError: If ``kp`` is NaN or outside ``[0, 9]``.

    Examples:
        >>> assign_severity_stratum(0.0)
        'quiet'
        >>> assign_severity_stratum(3.0)
        'quiet'
        >>> assign_severity_stratum(3.33)
        'moderate'
        >>> assign_severity_stratum(7.0)
        'extreme'
    """
    if kp != kp:  # NaN check without numpy import
        raise ValueError("kp must not be NaN")
    if not (_KP_FLOOR <= kp <= _KP_CEILING):
        raise ValueError(f"kp must be in [{_KP_FLOOR}, {_KP_CEILING}]; got {kp}")

    if kp <= _QUIET_MAX:
        return "quiet"
    if kp <= _MODERATE_MAX:
        return "moderate"
    return "extreme"


def stratify_by_severity(
    records: Iterable[ModelOutput],
) -> dict[SeverityStratum, list[ModelOutput]]:
    """Group records by their pre-assigned severity stratum.

    Records that do not carry a ``severity_stratum`` are dropped with a
    warning. Use :func:`assign_severity_stratum` on the upstream Kp value to
    populate the field before calling this function.

    Args:
        records: Iterable of :class:`~helios_fusion.types.ModelOutput`.

    Returns:
        A dict mapping each stratum label to the list of records in it.
        All three strata are always present in the returned dict; absent
        strata map to an empty list.
    """
    out: dict[SeverityStratum, list[ModelOutput]] = defaultdict(list)
    # Pre-populate so every stratum is present in the output.
    for stratum in _STRATA_TUPLE:
        out[stratum] = []

    dropped = 0
    for rec in records:
        stratum_opt = rec.severity_stratum
        if stratum_opt is None:
            dropped += 1
            continue
        out[stratum_opt].append(rec)

    if dropped:
        logger.warning(
            "stratify_by_severity dropped %d record(s) with no severity_stratum",
            dropped,
        )
    return dict(out)
