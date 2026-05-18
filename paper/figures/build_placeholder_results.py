"""Render the §4 results-section placeholder figures.

Sprint D writes the real plots:

* ``helios-program/results/<date>-killgate-reliability-diagrams.png``
* ``helios-program/results/<date>-killgate-bootstrap-distributions.png``

At preprint-fill-in time the operator (or the fill-in agent) copies those
into ``paper/figures/`` and the LaTeX source picks them up. Until then,
this script writes clearly-labelled placeholders so the LaTeX build is
green and the §4 ``\\includegraphics{...}`` calls do not fail.

Usage::

    python paper/figures/build_placeholder_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent


def _placeholder(name: str, label: str, sub: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.66,
        label,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#9c0a0a",
    )
    ax.text(
        0.5,
        0.48,
        sub,
        ha="center",
        va="center",
        fontsize=10,
        color="#0d1b2a",
    )
    ax.text(
        0.5,
        0.28,
        "Filled at preprint-completion time from "
        "``helios-program/results/<date>-killgate-*.png``.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#6c757d",
        family="monospace",
    )
    out = OUT_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote placeholder {out}")
    plt.close(fig)


def main() -> None:
    _placeholder(
        "reliability-diagrams.png",
        "RELIABILITY DIAGRAMS — DATA PENDING",
        "Per-Kp-stratum reliability (quiet / moderate / extreme); Sprint D output.",
    )
    _placeholder(
        "bootstrap-distributions.png",
        "BOOTSTRAP DISTRIBUTIONS — DATA PENDING",
        "HSS / Brier / CRPS bootstrap distributions over hold-out resamples.",
    )


if __name__ == "__main__":
    main()
