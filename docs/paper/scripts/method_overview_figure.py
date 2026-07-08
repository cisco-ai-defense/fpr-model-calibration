r"""Method overview figure (vector PDF).

Two-row pipeline diagram:
  fit:        [X_1,...,X_n] -> [fit s_1] -> [sample knots via s_1] -> [fit s_2]
  inference:                    [raw score x] -> [s_2] -> [calibrated]

The shipped $s_2$ appears in both rows with the same green styling, making
the fit-time artifact and the inference-time mapping visually identical.
"""

from __future__ import annotations

# ruff: noqa: E402,I001
import os
import tempfile
from pathlib import Path

mpl_cache_dir = Path(tempfile.gettempdir()) / "fprcal_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK = "#1a1a1a"
MUTED = "#5a5a5a"
GREEN = "#2f6b3d"
NODE_FACE = "#fafafa"
INPUT_FACE = "#eef2f7"
SHIP_FACE = "#eaf1e2"


def _node(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    title: str,
    subtitle: str = "",
    facecolor: str = NODE_FACE,
    edgecolor: str = INK,
    edgewidth: float = 1.0,
    title_color: str = INK,
    title_weight: str = "normal",
    title_size: float = 11.5,
    sub_size: float = 8.8,
    sub_color: str = MUTED,
    title_dy: float = 0.18,
    sub_dy: float = -0.22,
) -> tuple[float, float, float, float]:
    """Place a rounded-rectangle node; return its bbox (x0, y0, x1, y1)."""
    cx, cy = center
    w, h = size
    x0, y0 = cx - w / 2, cy - h / 2
    box = mpatches.FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=edgewidth,
        zorder=2,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(
            cx,
            cy + title_dy,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight=title_weight,
            zorder=3,
        )
        ax.text(
            cx,
            cy + sub_dy,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=sub_color,
            zorder=3,
        )
    else:
        ax.text(
            cx,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight=title_weight,
            zorder=3,
        )
    return x0, y0, x0 + w, y0 + h


def _arrow(
    ax: plt.Axes,
    xy0: tuple[float, float],
    xy1: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.1,
    mut: float = 12,
) -> None:
    ax.add_patch(
        mpatches.FancyArrowPatch(
            xy0,
            xy1,
            arrowstyle="-|>",
            mutation_scale=mut,
            linewidth=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def _row_of_nodes(
    ax: plt.Axes,
    y: float,
    specs: list[dict],
    *,
    box_w: float,
    box_h: float,
    gap: float,
) -> list[tuple[float, float, float, float]]:
    """Place nodes in a row, left-aligned at x=x0, with constant gap between."""
    bboxes = []
    n = len(specs)
    total = n * box_w + (n - 1) * gap
    x_start = (ax.get_xlim()[0] + ax.get_xlim()[1] - total) / 2
    for i, spec in enumerate(specs):
        cx = x_start + box_w / 2 + i * (box_w + gap)
        bbox = _node(ax, (cx, y), (box_w, box_h), **spec)
        bboxes.append(bbox)
    # Connect with arrows.
    for i in range(n - 1):
        _arrow(ax, (bboxes[i][2], y), (bboxes[i + 1][0], y))
    return bboxes


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Two rows with a dividing rule between.
    y_fit = 3.70
    y_inf = 1.10
    y_rule = 2.35

    top_specs = [
        dict(
            title=r"$X_1,\ldots,X_n$",
            subtitle="benign sample",
            facecolor=INPUT_FACE,
        ),
        dict(
            title=r"fit $s_1$",
            subtitle=r"rank + Filliben $p$",
        ),
        dict(
            title=r"sample knots via $s_1$",
            subtitle=r"at log-spaced $p$",
        ),
        dict(
            title=r"fit $s_2$",
            subtitle=r"raw $\mapsto$ calibrated",
            facecolor=SHIP_FACE,
            edgecolor=GREEN,
            edgewidth=1.6,
            title_color=GREEN,
            title_weight="bold",
            sub_color=GREEN,
        ),
    ]

    top_bboxes = _row_of_nodes(
        ax,
        y_fit,
        top_specs,
        box_w=2.90,
        box_h=1.10,
        gap=0.55,
    )

    # Horizontal rule between rows.
    rule_x0 = top_bboxes[0][0]
    rule_x1 = top_bboxes[-1][2]
    ax.plot(
        [rule_x0, rule_x1],
        [y_rule, y_rule],
        color="#c8c8c8",
        linewidth=0.8,
        zorder=1,
    )

    # Prominent row banners to the left of the boxes.
    banner_x = rule_x0 - 0.35
    ax.text(
        banner_x,
        y_fit,
        "FIT",
        ha="right",
        va="center",
        fontsize=16,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        banner_x,
        y_fit - 0.55,
        "time",
        ha="right",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )
    ax.text(
        banner_x,
        y_inf,
        "INFERENCE",
        ha="right",
        va="center",
        fontsize=16,
        color=GREEN,
        fontweight="bold",
    )
    ax.text(
        banner_x,
        y_inf - 0.55,
        "time",
        ha="right",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )

    # Bottom row: 3 boxes, same widths/gaps so the shipped s_2 sits visually
    # under the top s_2.
    bot_specs = [
        dict(
            title=r"raw score $x$",
            subtitle="from detector",
            facecolor=INPUT_FACE,
        ),
        dict(
            title=r"$s_2$",
            subtitle=r"raw $\mapsto$ calibrated",
            facecolor=SHIP_FACE,
            edgecolor=GREEN,
            edgewidth=1.6,
            title_color=GREEN,
            title_weight="bold",
            title_size=13.5,
            sub_color=GREEN,
        ),
        dict(
            title=r"$s_2(x) \in [0,1]$",
            subtitle="stable FPR meaning",
        ),
    ]
    # Bottom row spans the same horizontal extent as top row.
    bot_box_w = 3.20
    bot_gap = 1.25
    bot_bboxes = _row_of_nodes(
        ax,
        y_inf,
        bot_specs,
        box_w=bot_box_w,
        box_h=1.10,
        gap=bot_gap,
    )
    # Silence unused warning.
    _ = bot_bboxes

    pdf_path = output_dir / "method_overview.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.10)
    png_path = output_dir / "method_overview.png"
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight", pad_inches=0.10)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
