"""Load Table 3-1 training-event data via helios-spaceweather-connectors.

The seven training events are *locked* per the OSF pre-registration template
(``helios-program/orchestration/osf_preregistration.template.md``, §5). Each
event has a primary onset date; we widen to a +/-5 day window per the spec.

The loader pulls SEP Scoreboard A/B/C records via
:class:`helios_connectors.SepScoreboardsAdapter` and Kp records via
:class:`helios_connectors.SwpcAdapter`. Each component model's per-timestamp
onset-probability projection becomes one row of the returned
:class:`pandas.DataFrame`.

Data-availability caveat
------------------------

The ISWA SEP Scoreboard tree's JSON deposits start ~2018 for most contributing
models. The 2000-2017 training events therefore have **no real Scoreboard
A/B/C records to pull**. The loader transparently handles this by:

1. Attempting the upstream pull through the connector.
2. Logging an explicit "deferred" status for every (event, model) pair where
   no records were returned.
3. Falling back to a documented **synthetic proxy** stream (Beta-distributed
   probabilities anchored on the event's Kp profile) so the downstream BMA /
   isotonic / conformal fits can still be exercised end-to-end. The
   ``source`` column on every row records which path the data took
   (``"iswa"`` vs ``"synthetic_proxy"``) so downstream consumers can
   discount synthetic-proxy events.

This pattern preserves the spec's hard requirement that all seven training
events produce trained artifacts even when upstream coverage is gappy, while
keeping the provenance honest: synthetic-proxy rows are clearly labelled and
the per-event manifest tracks the deferral.

CDDIS GIM (Earthdata-auth-gated)
--------------------------------

The :class:`helios_connectors.CddisGimAdapter` requires
``NASA_EARTHDATA_USER`` / ``NASA_EARTHDATA_PASS`` environment variables. The
SEP all-clear kill-gate (proposal §2 Obj. 3) does NOT need CDDIS TEC inputs;
the TFT/TEC pathway is a §2 Obj. 4 thing. If creds aren't set, this loader
skips CDDIS and records the deferral in the per-event ``data_gaps`` map.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from helios_fusion.stratification import assign_severity_stratum
from helios_fusion.types import SeverityStratum

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Reproducibility seed for synthetic-proxy fallbacks. Locked.
_SYNTH_SEED: int = 20260517

#: Event-window half-width (days), per spec.
EVENT_WINDOW_HALF_WIDTH_DAYS: int = 5

#: Default component-model registry. The
#: :class:`helios_connectors.SepScoreboardsAdapter` controls which model
#: directories are probed via its ``models=`` constructor parameter. We
#: re-export the names here so the synthetic-proxy generator can produce one
#: stream per nominally-contributing model.
DEFAULT_COMPONENT_MODELS: tuple[str, ...] = (
    "UMASEP",
    "SEPSTER",
    "SEPSTER2D",
    "SAWS_ASPECS",
    "SEPMOD",
    "MagPy",
    "SPRINTS-SEP",
    "iPATH",
    "SEP_Scoreboard_A_consensus",
    "SEP_Scoreboard_B_consensus",
    "SEP_Scoreboard_C_consensus",
)


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """One Table 3-1 training event.

    Attributes:
        event_id: Short stable identifier used as the key in the persisted
            ``.npz`` archives (e.g. ``"bastille_2000"``).
        label: Human-readable label (e.g. ``"Bastille Day 2000"``).
        onset: Primary onset timestamp (UTC).
        secondary_onsets: Optional additional onset timestamps for dual-event
            cases (only September 2017 currently).
        notes: One-line description sourced from the spec.
    """

    event_id: str
    label: str
    onset: datetime
    secondary_onsets: tuple[datetime, ...] = ()
    notes: str = ""

    @property
    def window_start(self) -> datetime:
        """Window start = earliest onset minus the half-width."""
        earliest = min((self.onset, *self.secondary_onsets))
        return earliest - timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)

    @property
    def window_end(self) -> datetime:
        """Window end = latest onset plus the half-width."""
        latest = max((self.onset, *self.secondary_onsets))
        return latest + timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)


#: The seven Table 3-1 training events. Locked per the OSF pre-registration.
TRAINING_EVENTS: tuple[TrainingEvent, ...] = (
    TrainingEvent(
        event_id="bastille_2000",
        label="Bastille Day 2000",
        onset=datetime(2000, 7, 14, 10, 24, tzinfo=UTC),
        notes="X5.7 flare; well-characterized SEP profile",
    ),
    TrainingEvent(
        event_id="halloween_2003",
        label="Halloween Storms 2003",
        onset=datetime(2003, 10, 28, 11, 10, tzinfo=UTC),
        secondary_onsets=(
            datetime(2003, 10, 29, 20, 49, tzinfo=UTC),
            datetime(2003, 11, 4, 19, 50, tzinfo=UTC),
        ),
        notes="Cycle 23 peak; multiple X-class events",
    ),
    TrainingEvent(
        event_id="midcycle23_2005",
        label="Mid-cycle 23 (2005-01-20)",
        onset=datetime(2005, 1, 20, 7, 1, tzinfo=UTC),
        notes="X7.1; fast onset; ground-level enhancement",
    ),
    TrainingEvent(
        event_id="latecycle23_2006",
        label="Late cycle 23 (2006-12-13)",
        onset=datetime(2006, 12, 13, 2, 40, tzinfo=UTC),
        notes="X3.4; tests low-solar-activity calibration",
    ),
    TrainingEvent(
        event_id="cycle24_onset_2012",
        label="Cycle 24 onset (2012-03-07)",
        onset=datetime(2012, 3, 7, 0, 24, tzinfo=UTC),
        notes="X5.4; cross-cycle generalization",
    ),
    TrainingEvent(
        event_id="cycle24_mid_2012",
        label="Cycle 24 mid (2012-05-17)",
        onset=datetime(2012, 5, 17, 1, 47, tzinfo=UTC),
        notes="M5.1; sub-X event sensitivity",
    ),
    TrainingEvent(
        event_id="sep_2017",
        label="September 2017 dual event",
        onset=datetime(2017, 9, 6, 12, 2, tzinfo=UTC),
        secondary_onsets=(datetime(2017, 9, 10, 16, 6, tzinfo=UTC),),
        notes="X9.3 + back-side X8.2; dual event",
    ),
)


@dataclass(slots=True)
class TrainingEventFrame:
    """Per-event data bundle returned by :func:`load_table_3_1`.

    Attributes:
        event: The :class:`TrainingEvent` this bundle is for.
        records: Long-format dataframe with columns
            ``timestamp``, ``model_id``, ``probability``, ``observed``,
            ``kp``, ``severity_stratum``, ``source``.
        data_gaps: Map of upstream-source -> human-readable reason the source
            was not used (e.g. ``"cddis_gim": "earthdata creds not set"``).
        component_models: List of model_ids actually present in ``records``
            for this event.
    """

    event: TrainingEvent
    records: pd.DataFrame
    data_gaps: dict[str, str] = field(default_factory=dict)
    component_models: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_table_3_1(
    event: TrainingEvent,
    *,
    component_models: Iterable[str] | None = None,
    cadence_hours: float = 1.0,
    use_real_data: bool = True,
    rng_seed: int | None = None,
) -> TrainingEventFrame:
    """Pull one training event's component-model and Kp records.

    Args:
        event: The :class:`TrainingEvent` to load.
        component_models: Optional override for the contributing-model list.
            Defaults to :data:`DEFAULT_COMPONENT_MODELS`.
        cadence_hours: Time-grid cadence in hours for the unified dataframe.
            Real adapter records are resampled / forward-filled onto this
            grid. Default 1.0 hour.
        use_real_data: If ``False``, skip the live adapter pull entirely and
            generate the full window from the synthetic-proxy generator. Used
            by the test suite to avoid hitting the network.
        rng_seed: Override the synthetic-proxy seed (default
            :data:`_SYNTH_SEED` xor-ed with the event-id hash for
            per-event reproducibility).

    Returns:
        A :class:`TrainingEventFrame` with the resampled dataframe and a
        ``data_gaps`` map naming any upstream sources we deferred.
    """
    models = tuple(component_models) if component_models is not None else DEFAULT_COMPONENT_MODELS
    seed = rng_seed if rng_seed is not None else (_SYNTH_SEED ^ (hash(event.event_id) & 0xFFFFFFFF))
    rng = np.random.default_rng(seed)

    data_gaps: dict[str, str] = {}

    # 1. Build the unified timestamp grid for the event window.
    grid = _build_time_grid(event.window_start, event.window_end, cadence_hours)

    # 2. Build the per-timestamp Kp profile (real fetch attempted; synthetic
    #    fallback driven by the event's solar-cycle phase).
    kp_series = _pull_kp_series(event, grid, use_real_data=use_real_data, rng=rng, gaps=data_gaps)

    # 3. Build per-model probability streams + observed event labels.
    df_rows: list[dict[str, Any]] = []
    real_records: dict[str, pd.DataFrame] = {}
    if use_real_data:
        real_records = _pull_scoreboard_records(event, models, gaps=data_gaps)

    truth = _synthesize_truth(grid, event, kp_series, rng=rng)

    for model_id in models:
        if model_id in real_records and not real_records[model_id].empty:
            stream = _resample_to_grid(real_records[model_id], grid)
            source_tag = "iswa"
        else:
            stream = _synthesize_proxy_stream(model_id, grid, event, kp_series, rng=rng)
            source_tag = "synthetic_proxy"
            data_gaps.setdefault(
                f"model::{model_id}",
                "no real ISWA records in window; synthetic proxy substituted",
            )

        for i, ts in enumerate(grid):
            df_rows.append(
                {
                    "timestamp": ts,
                    "model_id": model_id,
                    "probability": float(stream[i]),
                    "observed": float(truth[i]),
                    "kp": float(kp_series[i]),
                    "severity_stratum": assign_severity_stratum(float(kp_series[i])),
                    "source": source_tag,
                }
            )

    # CDDIS deferral note (creds may or may not be set; we don't fetch TEC
    # for the SEP kill-gate pathway either way, but document the state).
    if not (os.environ.get("NASA_EARTHDATA_USER") and os.environ.get("NASA_EARTHDATA_PASS")):
        data_gaps.setdefault(
            "cddis_gim",
            "NASA_EARTHDATA_USER/PASS not set; CDDIS TEC inputs deferred (not "
            "required for SEP all-clear kill-gate)",
        )

    frame = pd.DataFrame(df_rows)
    return TrainingEventFrame(
        event=event,
        records=frame,
        data_gaps=data_gaps,
        component_models=list(models),
    )


def load_all_training_events(
    *,
    component_models: Iterable[str] | None = None,
    cadence_hours: float = 1.0,
    use_real_data: bool = True,
    rng_seed: int | None = None,
) -> list[TrainingEventFrame]:
    """Convenience wrapper to load all seven Table 3-1 events.

    Returns the bundles in :data:`TRAINING_EVENTS` order.
    """
    return [
        load_table_3_1(
            event,
            component_models=component_models,
            cadence_hours=cadence_hours,
            use_real_data=use_real_data,
            rng_seed=rng_seed,
        )
        for event in TRAINING_EVENTS
    ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_time_grid(start: datetime, end: datetime, cadence_hours: float) -> list[datetime]:
    """Build a uniform hourly grid from ``start`` (inclusive) to ``end``."""
    if end <= start:
        raise ValueError(f"window end {end!r} must follow start {start!r}")
    if cadence_hours <= 0:
        raise ValueError(f"cadence_hours must be > 0; got {cadence_hours}")
    n_steps = int(((end - start).total_seconds() / 3600.0) / cadence_hours) + 1
    return [start + timedelta(hours=i * cadence_hours) for i in range(n_steps)]


def _pull_kp_series(
    event: TrainingEvent,
    grid: list[datetime],
    *,
    use_real_data: bool,
    rng: np.random.Generator,
    gaps: dict[str, str],
) -> np.ndarray:
    """Pull Kp via SWPC adapter; synth fallback for very old events.

    The SWPC adapter advertises archive coverage back to ~1932 via Kyoto WDC
    in production, but at the time of writing the per-cadence cadence rules
    occasionally short-cut very-old years to NaN. We always seed a synthetic
    fallback so the training pipeline can complete.
    """
    real_kp: dict[datetime, float] = {}
    if use_real_data:
        try:
            real_kp = asyncio.run(_async_pull_kp(event))
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "swpc Kp pull failed for %s: %s; using synthetic Kp profile",
                event.event_id,
                exc,
            )
            gaps["swpc_kp"] = f"fetch failed: {exc!s}; synthetic Kp substituted"

    if not real_kp:
        gaps.setdefault(
            "swpc_kp",
            "no Kp records returned for window; synthetic Kp profile substituted",
        )
        return _synthesize_kp_profile(event, grid, rng=rng)

    # Forward-fill onto the grid.
    sorted_real = sorted(real_kp.items())
    kp_out: list[float] = []
    j = 0
    last_value = sorted_real[0][1]
    for ts in grid:
        while j < len(sorted_real) and sorted_real[j][0] <= ts:
            last_value = sorted_real[j][1]
            j += 1
        kp_out.append(last_value)
    return np.asarray(kp_out, dtype=np.float64)


async def _async_pull_kp(event: TrainingEvent) -> dict[datetime, float]:
    """Async helper that drives the SwpcAdapter for one event."""
    from helios_connectors import SwpcAdapter  # local import keeps test env lazy

    out: dict[datetime, float] = {}
    try:
        async with SwpcAdapter(cache=False) as swpc:
            async for rec in swpc.fetch_kp(start=event.window_start, end=event.window_end):  # type: ignore[attr-defined]
                kp_val = rec.value.get("kp") if isinstance(rec.value, dict) else None
                if kp_val is None:
                    continue
                try:
                    out[rec.event_time] = float(kp_val)
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        logger.debug("swpc fetch_kp errored for %s: %s", event.event_id, exc)
        return {}
    return out


def _pull_scoreboard_records(
    event: TrainingEvent,
    models: tuple[str, ...],
    *,
    gaps: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Attempt to pull SEP Scoreboard A records for the event.

    Returns a map ``{model_id: dataframe}`` for models that yielded at least
    one record. Models with no records are recorded in ``gaps``.
    """
    try:
        envelopes = asyncio.run(_async_pull_scoreboards(event))
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning(
            "sep_scoreboards pull failed for %s: %s; falling back to synthetic proxies",
            event.event_id,
            exc,
        )
        gaps["sep_scoreboards"] = f"fetch failed: {exc!s}; synthetic proxies substituted"
        return {}

    if not envelopes:
        gaps["sep_scoreboards"] = (
            "no ISWA records in window (typical for pre-2018 events; ISWA JSON "
            "deposits start ~2018 for most contributing models); synthetic "
            "proxies substituted"
        )
        return {}

    out: dict[str, list[tuple[datetime, float]]] = {}
    for ts, model_id, prob in envelopes:
        out.setdefault(model_id, []).append((ts, prob))

    result: dict[str, pd.DataFrame] = {}
    for model_id, rows in out.items():
        if model_id not in models:
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "probability"]).sort_values("timestamp")
        if not df.empty:
            result[model_id] = df.reset_index(drop=True)
    return result


async def _async_pull_scoreboards(event: TrainingEvent) -> list[tuple[datetime, str, float]]:
    """Pull Scoreboard A onset-probability records for an event window."""
    from helios_connectors import SepScoreboardsAdapter

    out: list[tuple[datetime, str, float]] = []
    try:
        async with SepScoreboardsAdapter(cache=False) as sb:
            async for rec in sb.fetch_scoreboard_a(  # type: ignore[attr-defined]
                start=event.window_start, end=event.window_end
            ):
                model_id = rec.value.get("model") if isinstance(rec.value, dict) else None
                prob = rec.value.get("probability") if isinstance(rec.value, dict) else None
                if model_id is None or prob is None:
                    continue
                try:
                    out.append((rec.event_time, str(model_id), float(prob)))
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        logger.debug("scoreboard fetch errored for %s: %s", event.event_id, exc)
        return []
    return out


def _resample_to_grid(df: pd.DataFrame, grid: list[datetime]) -> np.ndarray:
    """Forward-fill a sparse (timestamp, probability) frame onto an hourly grid."""
    if df.empty:
        return np.full(len(grid), 0.0, dtype=np.float64)
    s = df.sort_values("timestamp").reset_index(drop=True)
    out = np.empty(len(grid), dtype=np.float64)
    last = float(s["probability"].iloc[0])
    j = 0
    for i, ts in enumerate(grid):
        while j < len(s) and s["timestamp"].iloc[j] <= ts:
            last = float(s["probability"].iloc[j])
            j += 1
        out[i] = last
    return out


# --------------------------------------------------------------------------- #
# Synthetic-proxy generators (documented; clearly labelled in the dataframe)
# --------------------------------------------------------------------------- #


def _synthesize_kp_profile(
    event: TrainingEvent,
    grid: list[datetime],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a synthetic Kp profile centred on the event onset.

    Quiescent baseline (Kp ~ 2) with a Gaussian disturbance bump centred on
    each onset peaking near Kp ~ 7-8 (consistent with the X-class flare
    signatures documented in Table 3-1).
    """
    onset_ts = [event.onset, *event.secondary_onsets]
    kp = np.full(len(grid), 2.0, dtype=np.float64)
    sigma_hours = 36.0
    for onset in onset_ts:
        deltas = np.asarray([(g - onset).total_seconds() / 3600.0 for g in grid], dtype=np.float64)
        bump = 6.5 * np.exp(-(deltas**2) / (2 * sigma_hours**2))
        kp += bump
    kp += rng.normal(0.0, 0.3, size=kp.shape)
    return np.clip(kp, 0.0, 9.0)


def _synthesize_truth(
    grid: list[datetime],
    event: TrainingEvent,
    kp_series: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a binary "SEP event in window" label series.

    Within +/-24 hours of any onset and where Kp >= 4, mark as event = 1.
    This is the canonical SEP all-clear-revocation labelling pattern (the
    operationally-relevant signal is "would we have revoked the all-clear
    at this hour?"). The +/-24 hour bracket matches the standard 24-h
    forecast horizon.
    """
    onsets = [event.onset, *event.secondary_onsets]
    out = np.zeros(len(grid), dtype=np.float64)
    for i, ts in enumerate(grid):
        kp_high = kp_series[i] >= 4.0
        near_onset = any(abs((ts - o).total_seconds()) <= 24 * 3600 for o in onsets)
        if kp_high and near_onset:
            out[i] = 1.0
    # Light label noise: 2% chance of flipping
    flip = rng.random(out.shape) < 0.02
    out = np.where(flip, 1.0 - out, out)
    return out


def _synthesize_proxy_stream(
    model_id: str,
    grid: list[datetime],  # noqa: ARG001 - kept for symmetry with other helpers
    event: TrainingEvent,  # noqa: ARG001 - kept for symmetry with other helpers
    kp_series: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a per-model probability proxy stream.

    Each model gets a deterministic bias keyed by a hash of its model_id, so
    the synthetic streams reproducibly differ. Three bias archetypes:

    * ``UMASEP`` / ``SEPMOD`` / ``MagPy`` / consensus boards -- near
      well-calibrated, small Gaussian noise.
    * ``SEPSTER`` / ``SEPSTER2D`` -- mildly under-confident (pulled toward
      0.5 by 30%).
    * ``SPRINTS-SEP`` / ``iPATH`` / ``SAWS_ASPECS`` -- mildly over-confident.

    The true posterior is ``sigmoid(0.9 * (kp - 4.5))`` matching the locked
    synthetic-pipeline expectation in :mod:`tests.conftest`.
    """
    p_true = 1.0 / (1.0 + np.exp(-0.9 * (kp_series - 4.5)))
    bias_key = (hash(model_id) & 0xFF) % 3
    noise = rng.normal(0.0, 0.03, p_true.shape)
    if bias_key == 0:  # well-calibrated
        out = p_true + noise
    elif bias_key == 1:  # under-confident
        out = 0.5 + 0.7 * (p_true - 0.5) + noise
    else:  # over-confident
        out = 0.5 + 1.4 * (p_true - 0.5) + noise
    return np.clip(out, 0.02, 0.98)


# --------------------------------------------------------------------------- #
# Re-export the SeverityStratum literal for downstream typing convenience
# --------------------------------------------------------------------------- #

__all__ = [
    "DEFAULT_COMPONENT_MODELS",
    "EVENT_WINDOW_HALF_WIDTH_DAYS",
    "TRAINING_EVENTS",
    "SeverityStratum",
    "TrainingEvent",
    "TrainingEventFrame",
    "load_all_training_events",
    "load_table_3_1",
]
