"""Simulate FPR-label error for sorted benign thresholds.

The experiment isolates the edge-of-sample effect from Section 2.2 by drawing
Uniform(0, 1) benign calibration samples, selecting kth-largest thresholds,
and computing each selected threshold's true FPR as 1 - threshold.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import os
import tempfile
from pathlib import Path

mpl_cache_dir = Path(tempfile.gettempdir()) / "fpr_model_calibration_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260507
N_REPEATS = 50_000
N_VALUES = (100, 1_000, 10_000)
N_RANKS = 90
MAX_CHUNK_VALUES = 5_000_000

METHODS = {
    r"Sample rank $k/n$": {"color": "blue", "linestyle": "--", "linewidth": 1.5},
    r"Mean $k/(n+1)$": {"color": "green", "linestyle": ":", "linewidth": 1.8},
    "Filliben median": {"color": "red", "linestyle": "-", "linewidth": 2.2},
}


def _rank_grid(n: int) -> np.ndarray:
    """Log-spaced top-tail ranks, from the largest sample to n/2."""
    max_rank = max(2, n // 2)
    ranks = np.rint(np.logspace(0, np.log10(max_rank), N_RANKS)).astype(int)
    return np.unique(np.clip(ranks, 1, n))


def _filliben_upper_tail(rank_from_top: np.ndarray, n: int) -> np.ndarray:
    """Filliben plotting positions for upper-tail ranks."""
    position = (rank_from_top - 0.3175) / (n + 0.365)
    endpoint = 1.0 - 0.5 ** (1.0 / n)
    position = position.astype(float)
    position[rank_from_top == 1] = endpoint
    position[rank_from_top == n] = 0.5 ** (1.0 / n)
    return position


def _assigned_coordinates(rank_from_top: np.ndarray, n: int) -> dict[str, np.ndarray]:
    return {
        r"Sample rank $k/n$": rank_from_top / n,
        r"Mean $k/(n+1)$": rank_from_top / (n + 1),
        "Filliben median": _filliben_upper_tail(rank_from_top, n),
    }


def _simulate_for_n(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    ranks = _rank_grid(n)
    assigned = _assigned_coordinates(ranks, n)

    true_coordinates = np.empty((N_REPEATS, len(ranks)), dtype=np.float64)
    kth_from_bottom = n - ranks
    chunk_size = max(1, min(N_REPEATS, MAX_CHUNK_VALUES // n))
    row_start = 0
    while row_start < N_REPEATS:
        row_end = min(row_start + chunk_size, N_REPEATS)
        samples = rng.random((row_end - row_start, n))
        selected = np.partition(samples, kth_from_bottom, axis=1)[:, kth_from_bottom]
        true_coordinates[row_start:row_end] = 1.0 - selected
        row_start = row_end

    true_median = np.median(true_coordinates, axis=0)
    relative_error = {method: [] for method in METHODS}
    mean_abs_relative_error = {method: [] for method in METHODS}

    for rank_index, rank in enumerate(ranks):
        true_fpr_median = true_median[rank_index]
        true_fpr_samples = true_coordinates[:, rank_index]
        for method, coordinates in assigned.items():
            coordinate = coordinates[ranks == rank][0]
            error = (coordinate - true_fpr_median) / true_fpr_median * 100.0
            relative_error[method].append(error)
            mean_abs_error = (
                np.mean(np.abs(coordinate - true_fpr_samples)) / true_fpr_median * 100.0
            )
            mean_abs_relative_error[method].append(mean_abs_error)

    result: dict[str, np.ndarray] = {
        "true_median": true_median,
    }
    for method in METHODS:
        result[f"{method}_relative_error"] = np.array(relative_error[method])
        result[f"{method}_mean_abs_relative_error"] = np.array(mean_abs_relative_error[method])
    return result


def _plot_metric(
    results: dict[int, dict[str, np.ndarray]],
    metric_suffix: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(N_VALUES),
        figsize=(12, 3.7),
        sharex=False,
    )

    for col, n in enumerate(N_VALUES):
        result = results[n]
        x = result["true_median"]

        ax = axes[col]

        for method, style in METHODS.items():
            ax.plot(
                x,
                result[f"{method}_{metric_suffix}"],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=0.85,
            )

        ax.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Median true FPR (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"$n={n:,}$")
        ax.set_xlim(float(x.min()), 0.5)
        ax.grid(True, alpha=0.3, which="both")

        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(output_path)


def main() -> None:
    rng = np.random.default_rng(SEED)
    output_dir = Path(__file__).resolve().parents[1] / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {n: _simulate_for_n(n, rng) for n in N_VALUES}

    _plot_metric(
        results=results,
        metric_suffix="relative_error",
        ylabel="Median-centered FPR-label error (%)",
        output_path=output_dir / "plotting_position_error_simulation.png",
    )
    _plot_metric(
        results=results,
        metric_suffix="mean_abs_relative_error",
        ylabel="Mean abs. error / median true FPR (%)",
        output_path=output_dir / "plotting_position_absolute_error_simulation.png",
    )


if __name__ == "__main__":
    main()
