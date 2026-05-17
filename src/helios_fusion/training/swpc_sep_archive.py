"""Ingest the NOAA SESC Solar Proton Events archive as ground-truth labels.

Per the Sprint C-Training-v2 coverage matrix Path B decision: the historical
ISWA SEP Scoreboards JSON tree only deposits data from January 2017 onward,
so 6 of the 7 Table 3-1 training events have zero real ISWA upstream
records. The NOAA Space Environment Services Center "Solar Proton Events
Affecting the Earth Environment, 1976-present" archive provides the
methodologically-stronger ground-truth signal: **observed** SEP onset
times + peak proton flux for every documented event from 1976 forward.

This module parses that archive once, caches it locally, and exposes a
``event_truth_labels`` helper that returns the per-grid-timestamp binary
``observed`` column for one Table 3-1 event window.

Primary source URL:
    https://umbra.nascom.nasa.gov/SEP/seps.html
Backup (NCEI/NGDC mirror):
    https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/
    solar-energetic-particles/sgd-particles/

Archive HTML structure
----------------------

The page is a single ``<table>`` with year-header rows (``<strong>YYYY
</strong>``) preceding the event rows for that year. Each event row has the
shape::

    <tr><td>Jul 14/1045<td>Jul 15/1230<td align = right>    24,000<td>
        <td>Halo/14 1054
        <td>Jul 14/1024   <td>X5/3B <td>N22/W07<td align = right>9077

Columns are: Start (Day/UT), Maximum (Day/UT), Proton Flux pfu @ >10 MeV,
[year-header column], CME, Flare Maximum (Day/UT), Importance, Location,
NOAA SEC Region. We extract the first three columns + the year from the
preceding ``<strong>YYYY</strong>`` row.

Methodology note
----------------

The ``observed`` label is binary: 1 when a grid timestamp falls within the
[Start, Maximum + 24h] window of any NOAA-documented SEP event; 0
otherwise. This matches the v1 "+/-24h around onset" labelling convention
but anchors on **observed** rather than synthesized onsets.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

#: Primary archive URL.
ARCHIVE_URL: str = "https://umbra.nascom.nasa.gov/SEP/seps.html"

#: Backup mirror (rarely needed; SESC list is stable).
BACKUP_URL: str = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
    "solar-energetic-particles/sgd-particles/"
)

#: Default cache directory under the user's $HOME.
_DEFAULT_CACHE: Path = Path.home() / ".cache" / "helios-fusion-engine" / "swpc_sep_archive"

#: Truth-label half-window (hours after Maximum) per the labelling
#: convention. Locked.
TRUTH_WINDOW_HOURS_POST_MAX: int = 24

#: Regex for the year-header row.
_YEAR_RE = re.compile(r"<strong>\s*(\d{4})\s*</strong>")

#: Regex for an event-row's leading ``<td>{Mon} {dd}/{HHMM}<td>{Mon} {dd}/{HHMM}<td...>{flux}``
#: pattern. We capture (start_day, start_hhmm, max_day, max_hhmm, flux_str).
#: Pattern tolerates ``&nbsp;`` between Mon and day (seen in 2017+ rows of
#: the umbra archive) and inline ``<sup>...</sup>`` footnote refs (seen
#: in some 2005+ rows) between the start cell and the max cell.
_GAP = r"(?:\s|&nbsp;)+"
# ``_AFTER_HHMM`` allows optional inline footnote markup (e.g.
# ``<sup><a href="#NOTE_1">[1]</a></sup>``) before the closing-and-next-td
# sequence. We match anything non-``<td>`` up to the next ``<td>``.
_EVENT_RE = re.compile(
    r"<tr>\s*<td>"
    rf"(?P<start_mon>[A-Z][a-z]{{2}}){_GAP}(?P<start_day>\d{{1,2}})/(?P<start_hhmm>\d{{3,4}})"
    r"[^<]*(?:<sup>[^<]*<a[^>]*>[^<]*</a>[^<]*</sup>)?\s*<td>"
    rf"(?P<max_mon>[A-Z][a-z]{{2}}){_GAP}(?P<max_day>\d{{1,2}})/(?P<max_hhmm>\d{{3,4}})"
    r"\s*<td[^>]*>\s*(?P<flux>[\d,]+|\*+)",
    re.IGNORECASE,
)

#: Month-name -> month-number map.
_MONTH_MAP: dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def fetch_sep_event_list(
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    html: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch and parse the NOAA SESC SEP-event archive.

    Args:
        cache_dir: Directory to cache the raw HTML. Defaults to
            ``~/.cache/helios-fusion-engine/swpc_sep_archive/``.
        force_refresh: If ``True``, re-download even if the cache hit is
            fresh.
        html: Optional pre-fetched HTML string. When provided, the network
            fetch is skipped entirely and the cache is bypassed. Used by
            tests with the recorded fixture.
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with columns ``start_utc``, ``max_utc``, ``peak_flux_pfu``,
        ``year``, and ``event_label`` (a short human string). Empty / unparseable
        events are dropped.
    """
    if html is None:
        html = _load_archive_html(cache_dir=cache_dir, force_refresh=force_refresh, timeout=timeout)

    rows = _parse_archive_html(html)
    if not rows:
        logger.warning("swpc_sep_archive: parsed 0 events from archive HTML")
        return pd.DataFrame(
            columns=["start_utc", "max_utc", "peak_flux_pfu", "year", "event_label"]
        )

    df = pd.DataFrame(rows)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["max_utc"] = pd.to_datetime(df["max_utc"], utc=True)
    return df


def event_truth_labels(
    event_id: str,
    window: tuple[datetime, datetime],
    *,
    archive_df: pd.DataFrame | None = None,
    cadence_hours: float = 1.0,
    cache_dir: Path | None = None,
    html: str | None = None,
) -> pd.DataFrame:
    """Build a per-timestamp binary truth-label dataframe for one event.

    Args:
        event_id: Table 3-1 event id (used only for the returned label
            column for traceability).
        window: ``(start_utc, end_utc)`` window — typically the +/-5d
            window from the :class:`TrainingEvent`.
        archive_df: Optional pre-fetched archive frame from
            :func:`fetch_sep_event_list`. When omitted, the function
            fetches (and caches) the archive on its own.
        cadence_hours: Grid cadence; the returned frame has one row per
            grid timestamp.
        cache_dir / html: Forwarded to :func:`fetch_sep_event_list` when
            archive_df is None.

    Returns:
        DataFrame with columns ``timestamp`` (tz-aware UTC), ``observed``
        (0 or 1), and ``event_id``. Sorted by timestamp.
    """
    start, end = window
    start_utc = start if start.tzinfo else start.replace(tzinfo=UTC)
    end_utc = end if end.tzinfo else end.replace(tzinfo=UTC)

    if archive_df is None:
        archive_df = fetch_sep_event_list(cache_dir=cache_dir, html=html)

    # Filter archive events that overlap the window.
    overlapping = archive_df[
        (archive_df["max_utc"] + pd.Timedelta(hours=TRUTH_WINDOW_HOURS_POST_MAX) >= start_utc)
        & (archive_df["start_utc"] <= end_utc)
    ]

    # Build the grid.
    n_steps = int((end_utc - start_utc).total_seconds() / 3600.0 / cadence_hours) + 1
    grid = [start_utc + timedelta(hours=i * cadence_hours) for i in range(n_steps)]

    observed = [0] * len(grid)
    for _, row in overlapping.iterrows():
        ev_start = row["start_utc"].to_pydatetime()
        ev_end = row["max_utc"].to_pydatetime() + timedelta(hours=TRUTH_WINDOW_HOURS_POST_MAX)
        for i, ts in enumerate(grid):
            if ev_start <= ts <= ev_end:
                observed[i] = 1

    return pd.DataFrame(
        {
            "timestamp": grid,
            "observed": observed,
            "event_id": [event_id] * len(grid),
        }
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_archive_html(
    *,
    cache_dir: Path | None,
    force_refresh: bool,
    timeout: float,
) -> str:
    """Load the archive HTML from cache or by HTTP fetch."""
    target_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_file = target_dir / "seps.html"

    if cache_file.exists() and not force_refresh:
        text = cache_file.read_text(encoding="utf-8")
        if text.strip():
            return text

    text = _http_get_archive(timeout=timeout)

    cache_file.write_text(text, encoding="utf-8")
    return text


def _parse_archive_html(html: str) -> list[dict[str, object]]:
    """Parse the SESC archive HTML into a list of event-row dicts.

    Iterates the HTML once, tracking the current year header. Each event
    row produces one dict with keys ``start_utc`` (datetime), ``max_utc``
    (datetime), ``peak_flux_pfu`` (float or NaN), ``year`` (int), and
    ``event_label`` (string).
    """
    rows: list[dict[str, object]] = []
    cursor = 0
    current_year: int | None = None

    while cursor < len(html):
        year_match = _YEAR_RE.search(html, cursor)
        event_match = _EVENT_RE.search(html, cursor)

        if year_match and (event_match is None or year_match.start() < event_match.start()):
            current_year = int(year_match.group(1))
            cursor = year_match.end()
            continue

        if event_match is None:
            break

        cursor = event_match.end()
        if current_year is None:
            continue

        try:
            start_dt = _parse_dayut(
                year=current_year,
                mon=event_match.group("start_mon"),
                day=int(event_match.group("start_day")),
                hhmm=event_match.group("start_hhmm"),
            )
            # Max can roll to the next month or year; resolve by carrying
            # forward from start_dt.
            max_dt = _parse_dayut(
                year=current_year,
                mon=event_match.group("max_mon"),
                day=int(event_match.group("max_day")),
                hhmm=event_match.group("max_hhmm"),
            )
            if max_dt < start_dt:
                # Year rollover (Dec/Jan boundary).
                max_dt = max_dt.replace(year=current_year + 1)
        except (KeyError, ValueError):
            continue

        flux_raw = event_match.group("flux")
        try:
            flux_val = (
                float(flux_raw.replace(",", ""))
                if flux_raw and "*" not in flux_raw
                else float("nan")
            )
        except ValueError:
            flux_val = float("nan")

        rows.append(
            {
                "start_utc": start_dt,
                "max_utc": max_dt,
                "peak_flux_pfu": flux_val,
                "year": current_year,
                "event_label": f"{event_match.group('start_mon')} {event_match.group('start_day')} {current_year}",
            }
        )

    return rows


def _http_get_archive(*, timeout: float) -> str:
    """Fetch the SESC archive HTML, trying primary then backup.

    Both URLs are inline-literal ``https://`` strings against vetted hosts;
    no caller-controlled value reaches ``urllib.request.urlopen``. We use
    ``httpx`` (already a transitive dependency via helios-connectors) for
    the fetch to avoid urllib's ``file://`` scheme support entirely.
    """
    try:
        import httpx  # local import: only needed in the live-fetch path
    except ImportError as exc:  # pragma: no cover - dev env always has httpx
        raise RuntimeError(
            "swpc_sep_archive: httpx is required for live archive fetches; "
            "install via `pip install httpx`"
        ) from exc

    headers = {"User-Agent": "helios-fusion-engine/0.1.2 (research)"}
    try:
        resp = httpx.get(
            "https://umbra.nascom.nasa.gov/SEP/seps.html",
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("swpc_sep_archive: primary fetch failed: %s", exc)

    try:
        resp = httpx.get(
            "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
            "solar-features/solar-energetic-particles/sgd-particles/",
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except (httpx.HTTPError, OSError) as exc:
        raise RuntimeError(
            f"swpc_sep_archive: both primary and backup fetches failed: {exc!s}"
        ) from exc


def _parse_dayut(*, year: int, mon: str, day: int, hhmm: str) -> datetime:
    """Parse a ``Mon dd/HHMM`` cell into a tz-aware UTC datetime.

    The archive's HHMM is either 3 or 4 digits (e.g. ``930`` for 09:30
    or ``1335`` for 13:35).
    """
    month = _MONTH_MAP[mon[:3].title()]
    hhmm_padded = hhmm.rjust(4, "0")
    hour = int(hhmm_padded[:2])
    minute = int(hhmm_padded[2:])
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


__all__ = [
    "ARCHIVE_URL",
    "BACKUP_URL",
    "TRUTH_WINDOW_HOURS_POST_MAX",
    "event_truth_labels",
    "fetch_sep_event_list",
]
