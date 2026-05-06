"""Fit FPR calibration on the Credit Card Fraud ROC and produce the demo figure.

Reads ``examples/credit_card_roc.npz`` (produced by
``examples/generate_credit_card_roc.py``), fits a calibration pipeline on
benign scores from the 30% calibration-fit subset of the holdout, and writes
a four-panel validation figure to ``examples/credit_card_validation.png``.

All four panels are evaluated on the full 70% holdout. The calibration-fit
subset is shown alongside the full holdout on the ROC and FPR-threshold
panels so the reader can confirm the fit subset is representative of the
full holdout distribution.

The four panels:

1. ROC (TPR vs FPR, log x-axis): full holdout vs calibration-fit subset.
2. FPR -> threshold: same two populations, plus the fitted isotonic line.
3. Calibrated score vs empirical FPR: expected contract vs pipeline output
   on the full holdout.
4. Relative calibration error (%) across FPR on the full holdout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fpr_model_calibration.calibration import fit_calibration_pipeline, fpr_to_calibrated

ROC_FILE = Path(__file__).parent / "credit_card_roc.npz"
FIGURE = Path(__file__).parent / "credit_card_validation.png"


def compute_empirical_roc(
    benign: np.ndarray, attack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical (threshold, FPR, TPR) at each unique benign score.

    Uses benign thresholds only so every reported FPR is an exact ``k / n_benign``.
    """
    sorted_benign = np.sort(benign)
    thresholds = np.unique(sorted_benign)
    n_benign = len(benign)

    fprs = np.empty_like(thresholds, dtype=np.float64)
    tprs = np.empty_like(thresholds, dtype=np.float64)
    for i, t in enumerate(thresholds):
        fprs[i] = (n_benign - np.searchsorted(sorted_benign, t, side="left")) / n_benign
        tprs[i] = np.mean(attack >= t) if len(attack) > 0 else 0.0
    return thresholds, fprs, tprs


def _min_fpr_line(ax, min_fpr_fit: float) -> None:
    ax.axvline(
        min_fpr_fit,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Fit-subset min FPR ({min_fpr_fit:.1e})",
    )


def plot_four_panels(
    holdout_scores: np.ndarray,
    holdout_labels: np.ndarray,
    fit_mask: np.ndarray,
    pipeline,
    save_path: Path,
) -> None:
    """Four-panel calibration-quality figure on the holdout set."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    holdout_benign = holdout_scores[holdout_labels == 0]
    holdout_attack = holdout_scores[holdout_labels == 1]
    fit_benign = holdout_scores[(holdout_labels == 0) & fit_mask]
    fit_attack = holdout_scores[(holdout_labels == 1) & fit_mask]

    ho_thresh, ho_fpr, ho_tpr = compute_empirical_roc(holdout_benign, holdout_attack)
    fit_thresh, fit_fpr, fit_tpr = compute_empirical_roc(fit_benign, fit_attack)
    min_fpr_fit = 1.0 / len(fit_benign)

    # Panel 1: ROC, full holdout vs calibration-fit subset.
    ax = axes[0, 0]
    mask = ho_fpr > 0
    ax.plot(
        ho_fpr[mask],
        ho_tpr[mask],
        marker=".",
        markersize=2,
        linestyle="-",
        linewidth=0.8,
        color="blue",
        alpha=0.85,
        label=f"Full holdout (n={len(holdout_benign):,})",
    )
    fmask = fit_fpr > 0
    ax.plot(
        fit_fpr[fmask],
        fit_tpr[fmask],
        marker="x",
        markersize=3,
        linestyle="-",
        linewidth=0.6,
        color="green",
        alpha=0.7,
        label=f"Calibration-fit subset (n={len(fit_benign):,})",
    )
    _min_fpr_line(ax, min_fpr_fit)
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: full holdout vs calibration-fit subset")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: FPR -> threshold, same populations as Panel 1 + fitted isotonic.
    ax = axes[0, 1]
    mask_full = (ho_fpr > 0) & (ho_thresh > 0) & (ho_thresh < 1)
    ax.plot(
        ho_fpr[mask_full],
        ho_thresh[mask_full],
        marker=".",
        markersize=2,
        linestyle="-",
        linewidth=0.8,
        color="blue",
        alpha=0.85,
        label=f"Full holdout (n={len(holdout_benign):,})",
    )
    mask_fit_panel = (fit_fpr > 0) & (fit_thresh > 0) & (fit_thresh < 1)
    ax.plot(
        fit_fpr[mask_fit_panel],
        fit_thresh[mask_fit_panel],
        marker="x",
        markersize=3,
        linestyle="-",
        linewidth=0.6,
        color="green",
        alpha=0.7,
        label=f"Calibration-fit subset (n={len(fit_benign):,})",
    )

    # Red isotonic line: the fitted (FPR, threshold) map. The stored isotonic
    # lives in MinMaxScaler(feature_range=(0, 0.99)) space; invert the rescaler
    # so the curve shares the raw-score y-axis with the point clouds above.
    rescaler = pipeline.named_steps["rescale"]
    fpr_knots = pipeline.fpr_to_score_.X_thresholds_
    score_knots_raw = rescaler.inverse_transform(
        pipeline.fpr_to_score_.y_thresholds_.reshape(-1, 1)
    ).ravel()
    fpr_grid = np.logspace(-6, 0, 200)
    fpr_grid_clipped = np.clip(fpr_grid, fpr_knots.min(), fpr_knots.max())
    score_fit_curve = np.interp(fpr_grid_clipped, fpr_knots, score_knots_raw)
    ax.plot(
        fpr_grid,
        score_fit_curve,
        "-",
        color="red",
        linewidth=2,
        alpha=0.85,
        label="Isotonic fit",
    )
    _min_fpr_line(ax, min_fpr_fit)
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("Threshold (raw detector score)")
    ax.set_title("FPR -> threshold: full holdout vs calibration-fit subset")
    ax.set_xlim(1e-6, 1)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: expected vs actual calibrated score on the full holdout.
    ax = axes[1, 0]
    expected_fprs = np.logspace(-6, 0, 200)
    expected_cal = fpr_to_calibrated(expected_fprs)
    ax.plot(
        expected_fprs,
        expected_cal,
        "r--",
        linewidth=2,
        alpha=0.8,
        label="Expected (FPR -> calibrated)",
    )
    eval_mask = (ho_fpr > 0) & (ho_thresh > 0) & (ho_thresh < 1)
    actual_cal = pipeline.predict(ho_thresh[eval_mask].reshape(-1, 1))
    ax.plot(
        ho_fpr[eval_mask],
        actual_cal,
        marker=".",
        markersize=2,
        linestyle="none",
        color="blue",
        alpha=0.5,
        label="Full holdout",
    )
    _min_fpr_line(ax, min_fpr_fit)
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("Calibrated score")
    ax.set_title("Calibration: contract vs pipeline output")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 4: relative calibration error on the full holdout.
    ax = axes[1, 1]
    expected_at_fpr = fpr_to_calibrated(ho_fpr[eval_mask])
    valid = expected_at_fpr > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (actual_cal - expected_at_fpr) / expected_at_fpr * 100
    rel_err = np.nan_to_num(rel_err, nan=0, posinf=0, neginf=0)
    fpr_valid = ho_fpr[eval_mask][valid]
    err_valid = rel_err[valid]
    order = np.argsort(fpr_valid)
    ax.plot(
        fpr_valid[order],
        err_valid[order],
        "-",
        color="blue",
        linewidth=1,
        alpha=0.7,
        label="Full holdout",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    _min_fpr_line(ax, min_fpr_fit)
    ax.set_xscale("log")
    ax.set_xlabel("Empirical FPR")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Calibration error")
    ax.set_xlim(1e-6, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {save_path}")


def main() -> None:
    if not ROC_FILE.exists():
        raise SystemExit(
            f"Missing {ROC_FILE}. Run `python examples/generate_credit_card_roc.py` first."
        )

    data = np.load(ROC_FILE)
    # Use the Logistic Regression scores: GBDT's sigmoid output saturates
    # around FPR ~3e-4 (benign mass tied at score=1.0), which truncates the
    # ROC tail. LR's linear margin keeps near-continuous scores deep into
    # the tail, so the validation figure can show sub-1e-4 behavior.
    holdout_scores = data["holdout_scores_lr"]
    holdout_labels = data["holdout_labels"]
    fit_mask = data["fit_mask"]

    holdout_benign_n = int((holdout_labels == 0).sum())
    fit_benign_n = int(((holdout_labels == 0) & fit_mask).sum())
    print(
        f"Holdout: {len(holdout_scores):,} rows "
        f"({holdout_benign_n:,} benign, {int(holdout_labels.sum())} positive)."
    )
    print(
        f"Fit subset: {int(fit_mask.sum()):,} rows "
        f"({fit_benign_n:,} benign, {int(holdout_labels[fit_mask].sum())} positive)."
    )

    fit_benign_scores = holdout_scores[(holdout_labels == 0) & fit_mask]
    pipeline = fit_calibration_pipeline(fit_benign_scores, n_knots=10000, keep_debug=True)
    plot_four_panels(holdout_scores, holdout_labels, fit_mask, pipeline, FIGURE)


if __name__ == "__main__":
    main()
