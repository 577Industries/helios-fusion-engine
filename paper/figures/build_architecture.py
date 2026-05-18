"""Render the HELIOS architecture diagram (figure 1) as a PNG.

Source-of-truth Mermaid file is ``figures/architecture.mmd``. This Python
build produces a layout that does not require the ``mmdc`` CLI in CI,
keeping the build self-contained.

Usage::

    python paper/figures/build_architecture.py

Outputs ``paper/figures/architecture.png`` at 220 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def main() -> None:
    """Render architecture.png."""
    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=220)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Palette aligned with the helios-program Pages site (blue/teal).
    blue = "#1f3a93"
    teal = "#2a9d8f"
    grey = "#6c757d"
    text = "#0d1b2a"

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        sub: str,
        ver: str,
        color: str = blue,
        text_color: str = "white",
    ) -> tuple[float, float]:
        """Draw a labeled rectangular node; return its centre."""
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h - 0.22,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            x + w / 2,
            y + h - 0.50,
            sub,
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color,
        )
        ax.text(
            x + w / 2,
            y + 0.15,
            ver,
            ha="center",
            va="center",
            fontsize=7.0,
            style="italic",
            color=text_color,
        )
        return (x + w / 2, y + h / 2)

    def arrow(p1: tuple[float, float], p2: tuple[float, float], style: str = "-|>") -> None:
        """Draw a directed arrow between two centre points."""
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops={
                "arrowstyle": style,
                "color": text,
                "lw": 1.2,
                "shrinkA": 14,
                "shrinkB": 14,
            },
        )

    # Row 2 (top): the orchestration meta-repo, dotted-linked to everything.
    p_program = box(3.6, 3.8, 2.8, 1.0, "helios-program", "plan + companion", "v0.2.0", color=grey)

    # Row 1 (middle): the technical chain A -> B -> C plus the private weights F.
    p_spec = box(
        0.2,
        2.0,
        2.2,
        1.4,
        "helios-provenance-spec",
        "JSON Schema + pydantic v2",
        "v0.1.0",
        color=teal,
    )
    p_conn = box(
        2.7,
        2.0,
        2.4,
        1.4,
        "helios-spaceweather-connectors",
        "6 federal adapters",
        "v0.2.1",
        color=blue,
    )
    p_fuse = box(
        5.4,
        2.0,
        2.4,
        1.4,
        "helios-fusion-engine",
        "BMA + isotonic + conformal",
        "v0.1.2",
        color=blue,
    )
    p_priv = box(
        8.1, 2.0, 1.7, 1.4, "helios-fusion-internal", "trained weights", "private", color="#7f1d1d"
    )

    # Row 0 (bottom): a leaf consumer of B.
    p_gan = box(
        2.7, 0.3, 2.4, 1.2, "gannon-storm-rtk-analysis", "retrospective", "v0.1.0", color=teal
    )

    # Hard dependency arrows.
    arrow(p_spec, p_conn)
    arrow(p_spec, p_fuse)
    arrow(p_conn, p_fuse)
    arrow(p_fuse, p_priv)
    arrow(p_conn, p_gan)

    # Dotted ownership/orchestration links from the meta-repo.
    for child in [p_spec, p_conn, p_fuse, p_gan]:
        ax.annotate(
            "",
            xy=child,
            xytext=p_program,
            arrowprops={
                "arrowstyle": "-",
                "color": grey,
                "lw": 0.8,
                "linestyle": (0, (2, 2)),
                "shrinkA": 14,
                "shrinkB": 14,
            },
        )

    fig.tight_layout(pad=0.4)

    out = Path(__file__).resolve().parent / "architecture.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
