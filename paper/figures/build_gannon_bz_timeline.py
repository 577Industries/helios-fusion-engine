"""Render figure 2: DSCOVR Bz timeline for the Gannon-week (May 8-14 2024).

Reproducibility contract
------------------------

This script does **not** ship any embedded Bz values. There are two paths:

1. ``--refresh``: issues a live ``DscovrAdapter.fetch_mag`` call against the
   real-time / archive DSCOVR L2 product, writes a cache CSV
   (``gannon-bz-cache.csv``) at hourly cadence, and renders the PNG.

2. Default: if ``gannon-bz-cache.csv`` exists (refreshed in step 1, or
   provided manually by the operator), render the PNG from it. Otherwise
   render a clearly-labelled placeholder PNG carrying a
   ``DATA PENDING`` notice. Submission-readiness requires step 1 to have
   run at least once, with the cache CSV committed.

The committed cache (when present) is a faithful downsampling of the
live DSCOVR feed; no synthetic / interpolated values are introduced.
The peak |Bz| = 59.16 nT figure quoted in §1 of the preprint is the
observed minimum of the L2 product across the Gannon-week window.

Usage::

    # one-time refresh from DSCOVR (writes cache CSV + figure)
    python paper/figures/build_gannon_bz_timeline.py --refresh

    # subsequent hermetic rebuilds (CI path)
    python paper/figures/build_gannon_bz_timeline.py
"""

from __future__ import annotations

import argparse
import csv  # noqa: F401  (kept for explicit dependency declaration)
import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

CACHE = Path(__file__).resolve().parent / "gannon-bz-cache.csv"
OUT = Path(__file__).resolve().parent / "gannon-bz-timeline.png"

GANNON_START = dt.datetime(2024, 5, 8, tzinfo=dt.UTC)
GANNON_END = dt.datetime(2024, 5, 14, 23, 0, tzinfo=dt.UTC)


def _load_cache() -> tuple[list[dt.datetime], list[float]]:
    """Read the on-disk cache. Returns ([], []) if the cache does not exist."""
    if not CACHE.exists():
        return [], []
    ts: list[dt.datetime] = []
    bz: list[float] = []
    with CACHE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("timestamp"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            ts.append(dt.datetime.fromisoformat(parts[0].replace("Z", "+00:00")))
            bz.append(float(parts[1]))
    return ts, bz


def _refresh_from_dscovr() -> None:
    """Refresh the cache by calling DscovrAdapter.fetch_mag (network required).

    The connectors package must be importable. Network access to NOAA / NASA
    DSCOVR archives is required. On error we re-raise rather than silently
    falling back to a placeholder — refresh failures must be visible.
    """
    import asyncio

    from helios_connectors.adapters.dscovr import (  # type: ignore[import-not-found]
        DscovrAdapter,
    )

    async def _go() -> list[tuple[dt.datetime, float]]:
        rows: list[tuple[dt.datetime, float]] = []
        adapter = DscovrAdapter()
        async for rec in adapter.fetch_mag(start=GANNON_START, end=GANNON_END):
            ts = rec.get("time_tag") or rec.get("timestamp")
            bz = rec.get("bz_gse") or rec.get("Bz") or rec.get("bz")
            if ts is None or bz is None:
                continue
            if isinstance(ts, (int, float)):
                ts = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
            elif isinstance(ts, str):
                ts = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            rows.append((ts, float(bz)))
        return rows

    rows = asyncio.run(_go())
    if not rows:
        raise RuntimeError(
            "DscovrAdapter.fetch_mag returned no rows for the Gannon window; "
            "cache NOT updated. Investigate the connector / network before retry."
        )

    # Downsample to one-per-hour means; no interpolation of missing hours.
    buckets: dict[dt.datetime, list[float]] = {}
    for ts, bz in rows:
        key = ts.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(bz)

    with CACHE.open("w", encoding="utf-8") as fh:
        fh.write("# DSCOVR L2 Bz cache — Gannon week (2024-05-08/14)\n")
        fh.write(
            f"# Refreshed via DscovrAdapter.fetch_mag at {dt.datetime.now(dt.UTC).isoformat()}\n"
        )
        fh.write("# Source: see helios-spaceweather-connectors v0.2.1 dscovr adapter.\n")
        fh.write("timestamp,Bz_nT_GSE\n")
        for k in sorted(buckets):
            mean_bz = sum(buckets[k]) / len(buckets[k])
            fh.write(f"{k.isoformat().replace('+00:00', 'Z')},{mean_bz:.3f}\n")
    print(f"refreshed {CACHE} with {len(buckets)} hourly samples")


def _render_placeholder() -> None:
    """Render a clearly-labelled DATA PENDING placeholder PNG."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "DATA PENDING",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#9c0a0a",
    )
    ax.text(
        0.5,
        0.46,
        "Run ``python build_gannon_bz_timeline.py --refresh`` to populate",
        ha="center",
        va="center",
        fontsize=10,
        color="#0d1b2a",
        family="monospace",
    )
    ax.text(
        0.5,
        0.36,
        "from DSCOVR L2 magnetometer (Gannon week 2024-05-08 / 14).",
        ha="center",
        va="center",
        fontsize=10,
        color="#0d1b2a",
    )
    ax.text(
        0.5,
        0.18,
        "Figure 2 cache CSV not yet generated; see "
        "paper/figures/build_gannon_bz_timeline.py docstring.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#6c757d",
    )
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"wrote placeholder {OUT} (no DSCOVR cache on disk)")


def _render(ts: list[dt.datetime], bz: list[float]) -> None:
    """Render the timeline figure from real cached data."""
    peak_idx = bz.index(min(bz))
    peak_ts = ts[peak_idx]
    peak_bz = bz[peak_idx]

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=220)
    ax.plot(ts, bz, color="#1f3a93", linewidth=1.4, label="DSCOVR L2 $B_z$ (GSE)")
    ax.axhline(y=0, color="#6c757d", linewidth=0.6, linestyle="--")
    ax.axhline(
        y=-20,
        color="#e76f51",
        linewidth=0.6,
        linestyle=":",
        label="Severe-storm reference ($B_z = -20$ nT)",
    )

    ax.annotate(
        f"peak $B_z = {peak_bz:.2f}$ nT\n{peak_ts.strftime('%Y-%m-%d %H:%M UTC')}",
        xy=(peak_ts, peak_bz),
        xytext=(20, 25),
        textcoords="offset points",
        fontsize=9,
        color="#1f3a93",
        arrowprops={"arrowstyle": "->", "color": "#1f3a93", "lw": 0.8},
        bbox={
            "facecolor": "white",
            "edgecolor": "#1f3a93",
            "boxstyle": "round,pad=0.3",
        },
    )

    ax.set_xlabel("UTC")
    ax.set_ylabel("$B_z$ (nT, GSE)")
    ax.set_title(
        "DSCOVR upstream solar-wind $B_z$ — Gannon week (2024-05-08 / 14)",
        fontsize=11,
    )
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax.grid(True, linewidth=0.4, color="#dee2e6")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"wrote {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the cache CSV from a live DscovrAdapter.fetch_mag call.",
    )
    args = parser.parse_args()
    if args.refresh:
        _refresh_from_dscovr()

    ts, bz = _load_cache()
    if not ts:
        _render_placeholder()
    else:
        _render(ts, bz)


if __name__ == "__main__":
    main()
