"""Tests for the SWPC SESC SEP-event archive ingestion (Sprint C-Training-v2).

Recorded fixtures avoid hitting the network; the trimmed fixture covers
the parsing variants we care about (year header, plain row, ``<sup>``
footnote-ref row, ``&nbsp;``-separator row).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS
from helios_fusion.training.swpc_sep_archive import (
    ARCHIVE_URL,
    TRUTH_WINDOW_HOURS_POST_MAX,
    event_truth_labels,
    fetch_sep_event_list,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "swpc_archive"
_TRIMMED = _FIXTURE_DIR / "seps_trimmed_v2.html"
_FULL = _FIXTURE_DIR / "seps_full_2026-05-17.html"


@pytest.fixture(scope="module")
def trimmed_html() -> str:
    return _TRIMMED.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def full_html() -> str:
    return _FULL.read_text(encoding="utf-8")


def test_archive_url_is_canonical_umbra_endpoint() -> None:
    """The primary archive URL is the umbra.nascom.nasa.gov SEP list."""
    assert ARCHIVE_URL.startswith("https://umbra.nascom.nasa.gov/SEP/")


def test_fetch_sep_event_list_parses_trimmed_fixture(trimmed_html: str) -> None:
    """The trimmed fixture covers years 2000 + 2005 + 2017; expect at
    least 4 rows (Bastille, Nov 2000, Jan 2005, Sept 5 2017, Sept 10 2017)."""
    df = fetch_sep_event_list(html=trimmed_html)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 4
    # Bastille Day 2000 SEP onset.
    bastille = df[(df["year"] == 2000) & (df["start_utc"].dt.month == 7)]
    assert len(bastille) >= 1
    row = bastille.iloc[0]
    assert row["start_utc"] == pd.Timestamp("2000-07-14 10:45:00", tz="UTC")
    # Peak flux 24,000 pfu.
    assert int(row["peak_flux_pfu"]) == 24000


def test_fetch_sep_event_list_handles_sup_footnote(trimmed_html: str) -> None:
    """The Jan 16 2005 row has a <sup><a>...</a></sup> footnote between
    the start cell and the second <td>; ensure we still parse it."""
    df = fetch_sep_event_list(html=trimmed_html)
    midcycle = df[(df["year"] == 2005) & (df["start_utc"].dt.month == 1)]
    assert len(midcycle) == 1
    row = midcycle.iloc[0]
    assert row["start_utc"] == pd.Timestamp("2005-01-16 02:10:00", tz="UTC")


def test_fetch_sep_event_list_handles_nbsp_separator(trimmed_html: str) -> None:
    """The 2017 rows use &nbsp; between Mon and day in the Max column."""
    df = fetch_sep_event_list(html=trimmed_html)
    sep_2017 = df[df["year"] == 2017]
    assert len(sep_2017) == 2
    onsets = sorted(sep_2017["start_utc"].tolist())
    assert onsets[0] == pd.Timestamp("2017-09-05 00:40:00", tz="UTC")
    assert onsets[1] == pd.Timestamp("2017-09-10 16:45:00", tz="UTC")


def test_fetch_full_archive_covers_all_table_3_1_years(full_html: str) -> None:
    """The recorded full archive should cover every year a Table 3-1
    training event lives in."""
    df = fetch_sep_event_list(html=full_html)
    years_present = set(df["year"].unique())
    expected_years = {2000, 2003, 2005, 2006, 2012, 2017}
    missing = expected_years - years_present
    assert not missing, f"archive missing years: {missing}"


def test_event_truth_labels_marks_onset_window(trimmed_html: str) -> None:
    """Truth labels mark grid timestamps inside the [start, max + 24h]
    window as 1; everything else 0."""
    df = fetch_sep_event_list(html=trimmed_html)
    sep_2017 = next(e for e in TRAINING_EVENTS if e.event_id == "sep_2017")
    labels = event_truth_labels(
        sep_2017.event_id,
        (sep_2017.window_start, sep_2017.window_end),
        archive_df=df,
        cadence_hours=1.0,
    )
    assert set(labels["observed"].unique()).issubset({0, 1})
    n_positive = int(labels["observed"].sum())
    # Two onsets (Sep 05 and Sep 10) -> two coverage windows -> non-zero
    # positive count.
    assert n_positive > 0


def test_event_truth_labels_returns_zero_outside_archive_coverage(
    trimmed_html: str,
) -> None:
    """An event with no archive entries in its window yields all-zero
    truth labels (degenerate case)."""
    df = fetch_sep_event_list(html=trimmed_html)
    # Trimmed fixture omits 2003 + 2006 + 2012; halloween_2003 should
    # therefore have 0 positive rows against this fixture.
    halloween = next(e for e in TRAINING_EVENTS if e.event_id == "halloween_2003")
    labels = event_truth_labels(
        halloween.event_id,
        (halloween.window_start, halloween.window_end),
        archive_df=df,
        cadence_hours=12.0,
    )
    assert int(labels["observed"].sum()) == 0


def test_truth_window_post_max_constant_is_24h() -> None:
    """Methodology lock: truth-window half-width is 24 hours post-peak."""
    assert TRUTH_WINDOW_HOURS_POST_MAX == 24


def test_full_archive_per_event_label_counts(full_html: str) -> None:
    """Every Table 3-1 event should generate at least some positive
    truth labels against the full archive (sanity check)."""
    df = fetch_sep_event_list(html=full_html)
    for ev in TRAINING_EVENTS:
        labels = event_truth_labels(
            ev.event_id,
            (ev.window_start, ev.window_end),
            archive_df=df,
            cadence_hours=1.0,
        )
        n_pos = int(labels["observed"].sum())
        assert n_pos > 0, f"event {ev.event_id} got 0 positive truth labels"


def test_year_rollover_in_max_date_handled() -> None:
    """A row with a Max date that wraps from Dec to Jan should be parsed
    correctly (the year is bumped on the Max side only)."""
    fake_html = """
    <table>
    <tr><td><td><td><td><strong>2024</strong><td><td><td><td><td>
    <tr><td>Dec 31/2300<td>Jan 01/0500<td align = right>50<td>
        <td>Halo/31 2200<td>Dec 31/2200<td>X1<td>S00W00<td align = right>9999</tr>
    </table>
    """
    df = fetch_sep_event_list(html=fake_html)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["start_utc"] == pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
    assert row["max_utc"] == pd.Timestamp("2025-01-01 05:00:00", tz="UTC")
