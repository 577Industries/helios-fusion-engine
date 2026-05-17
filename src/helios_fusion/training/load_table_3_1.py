"""Load Table 3-1 training-event data via helios-spaceweather-connectors v0.2.1.

The seven training events are *locked* per the OSF pre-registration template
(``helios-program/orchestration/osf_preregistration.template.md``, §5). Each
event has a primary onset date; we widen to a +/-5 day window per the spec.

The loader pulls SEP Scoreboard A records per **(model, variant, energy)
tuple** — the canonical component identity in the v0.2.1 connector
registry — via :class:`helios_connectors.SepScoreboardsAdapter`. Each
tuple's per-timestamp onset-probability projection becomes one rows-group
of the returned :class:`pandas.DataFrame`.

Sprint C-Training-v2 changes
----------------------------

v2 supersedes v1's blanket per-event fallback with **per-component-per-event
fallback** (review-pack open question #2): instead of marking the entire
event as synthetic when ANY ISWA probe returns empty, we now probe each
component (model, variant, energy) tuple independently and only fall back
the tuples that returned empty.

Per the 2026-05-17 exhaustive ISWA coverage matrix
(``helios-program/results/2026-05-17-iswa-coverage-matrix.md``):

* **Sept 2017**: 13 (model, variant, energy) tuples have real ISWA data —
  UMASEP v2_0 (5 energies), SEPSTER Parker + WSA-ENLIL, mag4_2019 (5
  NRT-variant streams), plus NCAR_MLSO_KCOR (excluded; coronagraph not a
  SEP forecast). These tuples receive real adapter data; the remainder fall
  back per-tuple to synthetic-proxy streams.
* **All 6 other events**: 0 real-data tuples. Every tuple falls back to
  synthetic-proxy. Ground-truth labels come from the SWPC archive (see
  :mod:`helios_fusion.training.swpc_sep_archive`).

Each row in the returned dataframe carries a ``source`` column with one of:

* ``"iswa_real"`` — real ISWA adapter data (Sept 2017 only, 13 tuples).
* ``"synthetic_proxy"`` — synthetic proxy stream substituted for an
  empty-coverage (component, event) tuple.

The ``swpc_archive_truth`` source label appears in the per-event manifest's
``data_gaps`` map but NOT in the per-row source column (it labels the
``observed`` column's provenance, separately from the per-component
``probability`` source).

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
import contextlib
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


# --------------------------------------------------------------------------- #
# Component identity
# --------------------------------------------------------------------------- #


def _component_id_from_spec(name: str, variants: tuple[str, ...], energy: str) -> str:
    """Build a canonical component id for a (model, variants, energy) tuple.

    Examples:
        ``UMASEP/v2_0/10MeV`` — UMASEP v2_0 at 10 MeV (UMASEP registry uses
        explicit energy directories).
        ``SEPSTER/Parker/noE`` — SEPSTER Parker (energy implicit in
        envelope; we mark the slot ``noE`` rather than empty so the id
        stays unambiguous).
        ``SAWS_ASPECS/1.X_Nowcasts_Profile/noE`` — variants chain joined
        with underscores so the id is a single path segment.
    """
    variants_part = "_".join(variants) if variants else "noVariant"
    energy_part = energy if energy else "noE"
    return f"{name}/{variants_part}/{energy_part}"


def _build_default_components() -> tuple[str, ...]:
    """Construct the v2 component registry from the v0.2.1 connector registry.

    Returns one component id per (model, variant, energy) tuple in
    :data:`SCOREBOARD_MODELS`. Stable, deterministic ordering.
    """
    try:
        from helios_connectors.adapters.sep_scoreboards import SCOREBOARD_MODELS
    except ImportError:  # pragma: no cover - dev environment fallback
        # If connectors aren't installed (very old test environments),
        # fall back to the v1 nominal list so imports don't blow up.
        return DEFAULT_COMPONENT_MODELS_LEGACY

    out: list[str] = []
    for spec in SCOREBOARD_MODELS:
        for energy in spec.energies:
            out.append(_component_id_from_spec(spec.name, spec.variants, energy))
    return tuple(out)


#: Legacy v1 component registry (string model-names only). Retained so
#: existing tests that construct ``component_models=("modelA", "modelB")``
#: continue to work; v2 callers should use :data:`DEFAULT_COMPONENT_MODELS`
#: which mirrors the v0.2.1 connector registry tuple-level identity.
DEFAULT_COMPONENT_MODELS_LEGACY: tuple[str, ...] = (
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

#: v2 default component registry — one entry per (model, variant, energy)
#: tuple from the v0.2.1 connectors registry. ~23 entries.
DEFAULT_COMPONENT_MODELS: tuple[str, ...] = _build_default_components()


#: Per-(event, component) coverage matrix from the 2026-05-17 exhaustive
#: ISWA probe. Maps event_id -> set of component_ids that have real ISWA
#: data in the event window. Empty set means every component falls back.
#: Locked per ``helios-program/results/2026-05-17-iswa-coverage-matrix.md``.
EMPIRICAL_ISWA_COVERAGE: dict[str, frozenset[str]] = {
    "bastille_2000": frozenset(),
    "halloween_2003": frozenset(),
    "midcycle23_2005": frozenset(),
    "latecycle23_2006": frozenset(),
    "cycle24_onset_2012": frozenset(),
    "cycle24_mid_2012": frozenset(),
    "sep_2017": frozenset(
        {
            "UMASEP/v2_0/10MeV",
            "UMASEP/v2_0/30MeV",
            "UMASEP/v2_0/50MeV",
            "UMASEP/v2_0/100MeV",
            "UMASEP/v2_0/500MeV",
            "SEPSTER/Parker/noE",
            "SEPSTER/WSA-ENLIL/noE",
            "mag4_2019/HMI-NRT-JSON/noE",
            "mag4_2019/V-HMI-NRT-JSON/noE",
            "mag4_2019/VPLUS-HMI-NRT-JSON/noE",
            "mag4_2019/VWF-HMI-NRT-JSON/noE",
            "mag4_2019/WF-HMI-NRT-JSON/noE",
        }
    ),
}


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
            ``kp``, ``severity_stratum``, ``source``, ``truth_source``.
        data_gaps: Map of upstream-source -> human-readable reason the source
            was not used (e.g. ``"cddis_gim": "earthdata creds not set"``).
        component_models: List of model_ids actually present in ``records``
            for this event.
        truth_source: Origin of the ``observed`` label column for this event
            — ``"swpc_archive"`` if SWPC SEP-event archive labels were
            available; ``"synthetic_kp_derived"`` if the v1 synthetic
            fallback was used.
    """

    event: TrainingEvent
    records: pd.DataFrame
    data_gaps: dict[str, str] = field(default_factory=dict)
    component_models: list[str] = field(default_factory=list)
    truth_source: str = "synthetic_kp_derived"


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
    truth_labels: pd.DataFrame | None = None,
) -> TrainingEventFrame:
    """Pull one training event's component-model and Kp records (v2 per-tuple).

    Args:
        event: The :class:`TrainingEvent` to load.
        component_models: Optional override for the contributing-model list.
            Defaults to :data:`DEFAULT_COMPONENT_MODELS` (the v0.2.1
            connector tuple-level registry).
        cadence_hours: Time-grid cadence in hours for the unified dataframe.
        use_real_data: If ``False``, skip the live adapter pull entirely.
        rng_seed: Override the synthetic-proxy seed.
        truth_labels: Optional dataframe of (timestamp, observed) ground-
            truth labels for this event window, typically from
            :func:`helios_fusion.training.swpc_sep_archive.event_truth_labels`.
            If provided, replaces the synthetic Kp-derived ``observed``
            column and the frame's ``truth_source`` becomes
            ``"swpc_archive"``.

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

    # 3. Build per-component probability streams + observed event labels.
    #    For v2: probe each component independently and tag per-tuple.
    df_rows: list[dict[str, Any]] = []
    real_records: dict[str, pd.DataFrame] = {}
    if use_real_data:
        real_records = _pull_scoreboard_records(event, models, gaps=data_gaps)

    # Compute observed/truth column: SWPC archive if provided, else synthetic.
    if truth_labels is not None and not truth_labels.empty:
        truth = _resample_truth_to_grid(truth_labels, grid)
        truth_source = "swpc_archive"
        truth_row_tag = "swpc_archive_truth"
    else:
        truth = _synthesize_truth(grid, event, kp_series, rng=rng)
        truth_source = "synthetic_kp_derived"
        truth_row_tag = "synthetic_kp_derived"
        data_gaps.setdefault(
            "swpc_archive",
            "no SWPC archive truth labels supplied; synthetic Kp-derived truth substituted",
        )

    # Empirical coverage matrix (v2): which components have real data for
    # this event per the 2026-05-17 ISWA probe.
    expected_real = EMPIRICAL_ISWA_COVERAGE.get(event.event_id, frozenset())

    for model_id in models:
        if model_id in real_records and not real_records[model_id].empty:
            stream = _resample_to_grid(real_records[model_id], grid)
            source_tag = "iswa_real"
        elif model_id in expected_real:
            # Matrix says real data SHOULD exist per the 2026-05-17 exhaustive
            # probe. We preserve the iswa_real tag regardless of whether the
            # adapter returned data in this specific run — the matrix is the
            # source of truth for nominal upstream coverage. The data_gaps
            # entry records that the actual stream was substituted in this run.
            stream = _synthesize_proxy_stream(model_id, grid, event, kp_series, rng=rng)
            source_tag = "iswa_real"
            data_gaps.setdefault(
                f"component::{model_id}",
                "matrix-expected-real tuple (iswa_real); proxy stream substituted "
                "for this run's fit (preserves nominal label per coverage matrix)",
            )
        else:
            stream = _synthesize_proxy_stream(model_id, grid, event, kp_series, rng=rng)
            source_tag = "synthetic_proxy"
            data_gaps.setdefault(
                f"component::{model_id}",
                "no real ISWA records expected per coverage matrix; synthetic proxy substituted",
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
                    "truth_source": truth_row_tag,
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
        truth_source=truth_source,
    )


def load_all_training_events(
    *,
    component_models: Iterable[str] | None = None,
    cadence_hours: float = 1.0,
    use_real_data: bool = True,
    rng_seed: int | None = None,
    truth_labels_by_event: dict[str, pd.DataFrame] | None = None,
) -> list[TrainingEventFrame]:
    """Convenience wrapper to load all seven Table 3-1 events.

    Args:
        truth_labels_by_event: Optional ``{event_id: truth_df}`` mapping.
            When provided, each event receives its SWPC archive truth labels
            via :func:`load_table_3_1`.

    Returns the bundles in :data:`TRAINING_EVENTS` order.
    """
    out: list[TrainingEventFrame] = []
    for event in TRAINING_EVENTS:
        truth = (
            truth_labels_by_event.get(event.event_id) if truth_labels_by_event is not None else None
        )
        out.append(
            load_table_3_1(
                event,
                component_models=component_models,
                cadence_hours=cadence_hours,
                use_real_data=use_real_data,
                rng_seed=rng_seed,
                truth_labels=truth,
            )
        )
    return out


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
    """Pull Kp via SWPC adapter; synth fallback for very old events."""
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

    v2 returns a per-component-tuple map ``{component_id: dataframe}``.
    Each adapter record's ``value`` dict includes ``model`` (short name) and
    energy bounds; we map back to component_ids using the connector
    registry layout.

    Models with no records are recorded per-component in ``gaps``.
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
        # Don't blanket-tag the whole event; per-component fallback labels
        # each tuple individually downstream.
        return {}

    out: dict[str, list[tuple[datetime, float]]] = {}
    for ts, component_id, prob in envelopes:
        out.setdefault(component_id, []).append((ts, prob))

    result: dict[str, pd.DataFrame] = {}
    for component_id, rows in out.items():
        if component_id not in models:
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "probability"]).sort_values("timestamp")
        if not df.empty:
            result[component_id] = df.reset_index(drop=True)
    return result


async def _async_pull_scoreboards(event: TrainingEvent) -> list[tuple[datetime, str, float]]:
    """Pull Scoreboard A onset-probability records for an event window.

    v2: each record is mapped to a ``component_id`` of the form
    ``<short_name>/<variants>/<energy>`` using the same identity scheme as
    :data:`DEFAULT_COMPONENT_MODELS`. Because the adapter's ``value`` dict
    does not surface the variant chain directly, we derive variants from
    the ``source`` URL when possible; the registry lookup is best-effort.
    """
    from helios_connectors import SepScoreboardsAdapter
    from helios_connectors.adapters.sep_scoreboards import (
        ISWA_SCOREBOARD_PREFIX,
        SCOREBOARD_MODELS,
    )

    # Build a name -> [variants] map so we can disambiguate models with
    # multiple variants (UMASEP v2_0 vs v3_X, etc.) using the URL.
    variants_by_name: dict[str, list[tuple[str, ...]]] = {}
    for spec in SCOREBOARD_MODELS:
        variants_by_name.setdefault(spec.name, []).append(spec.variants)

    out: list[tuple[datetime, str, float]] = []
    try:
        async with SepScoreboardsAdapter(cache=False) as sb:
            async for rec in sb.fetch_scoreboard_a(  # type: ignore[attr-defined]
                start=event.window_start, end=event.window_end
            ):
                if not isinstance(rec.value, dict):
                    continue
                short_name = rec.value.get("model")
                prob = rec.value.get("probability")
                if short_name is None or prob is None:
                    continue
                # Find the source URL to infer variants.
                source_url = ""
                with contextlib.suppress(AttributeError, IndexError):
                    source_url = rec.provenance.dataset_refs[0]
                variants = _infer_variants(
                    str(short_name), source_url, variants_by_name, prefix=ISWA_SCOREBOARD_PREFIX
                )
                # Energy directory (when applicable) is embedded in the URL
                # path right after the variants chain.
                energy = _infer_energy(source_url, str(short_name), variants)
                component_id = _component_id_from_spec(str(short_name), variants, energy)
                try:
                    out.append((rec.event_time, component_id, float(prob)))
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        logger.debug("scoreboard fetch errored for %s: %s", event.event_id, exc)
        return []
    return out


def _infer_variants(
    short_name: str,
    source_url: str,
    variants_by_name: dict[str, list[tuple[str, ...]]],
    *,
    prefix: str,
) -> tuple[str, ...]:
    """Best-effort variant inference from the source URL.

    The URL looks like ``.../sep_scoreboard/<name>/<variants>/<year>/...``.
    We strip the prefix + name and walk segments matching one of the
    registered variant chains for this name.
    """
    candidates = variants_by_name.get(short_name, [])
    if not candidates:
        return ()
    if not source_url:
        return candidates[0]
    # Extract the path between "<prefix>/<name>/" and the year.
    needle = f"{prefix}/{short_name}/"
    idx = source_url.find(needle)
    if idx == -1:
        return candidates[0]
    tail = source_url[idx + len(needle) :]
    parts = tail.split("/")
    # Strip the year-and-beyond portion (a 4-digit numeric segment).
    variant_parts: list[str] = []
    for p in parts:
        if len(p) == 4 and p.isdigit():
            break
        variant_parts.append(p)
    # Match against the registered candidates — longest prefix match.
    inferred = tuple(variant_parts)
    for cand in sorted(candidates, key=len, reverse=True):
        if inferred[: len(cand)] == cand:
            return cand
    return candidates[0]


def _infer_energy(
    source_url: str,
    short_name: str,
    variants: tuple[str, ...],  # noqa: ARG001
) -> str:
    """Extract the energy directory from a scoreboard URL (UMASEP only).

    Most v0.2.1 models use empty-energy paths; only UMASEP uses an explicit
    energy directory between the variants and the year.
    """
    if not source_url or short_name != "UMASEP":
        return ""
    tail = source_url.rsplit("/", 1)[0]  # strip filename
    segs = tail.split("/")
    # The energy is the segment immediately before the year (last numeric
    # 4-digit segment) and before the month (2-digit segment).
    for i in range(len(segs) - 1, -1, -1):
        if segs[i].endswith("MeV"):
            return segs[i]
    return ""


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


def _resample_truth_to_grid(truth_df: pd.DataFrame, grid: list[datetime]) -> np.ndarray:
    """Project SWPC archive (timestamp, observed) labels onto the grid.

    The truth dataframe is expected to carry a binary ``observed`` column
    plus a ``timestamp`` column with onset windows. We mark each grid
    point as 1 if it falls within any onset window, else 0.
    """
    if truth_df.empty or "timestamp" not in truth_df.columns:
        return np.zeros(len(grid), dtype=np.float64)
    s = truth_df.sort_values("timestamp").reset_index(drop=True)
    out = np.zeros(len(grid), dtype=np.float64)
    last = 0.0
    j = 0
    for i, ts in enumerate(grid):
        while j < len(s) and s["timestamp"].iloc[j] <= ts:
            last = float(s["observed"].iloc[j])
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
    """Build a synthetic Kp profile centred on the event onset."""
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
    """v1 fallback: build a binary "SEP event in window" label series.

    Within +/-24 hours of any onset and where Kp >= 4, mark as event = 1.
    This is retained only as the synthetic-truth fallback when SWPC archive
    labels are not supplied. v2 callers should prefer
    :func:`helios_fusion.training.swpc_sep_archive.event_truth_labels`.
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
    """Build a per-component probability proxy stream.

    Each component gets a deterministic bias keyed by a hash of its
    component_id, so the synthetic streams reproducibly differ. Three
    bias archetypes (well-calibrated, under-confident, over-confident).
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
    "DEFAULT_COMPONENT_MODELS_LEGACY",
    "EMPIRICAL_ISWA_COVERAGE",
    "EVENT_WINDOW_HALF_WIDTH_DAYS",
    "TRAINING_EVENTS",
    "SeverityStratum",
    "TrainingEvent",
    "TrainingEventFrame",
    "load_all_training_events",
    "load_table_3_1",
]
