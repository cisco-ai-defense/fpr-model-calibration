"""Fit FPR calibration on the Credit Card Fraud ROC and produce the demo figure.

Reads ``examples/credit_card_roc.npz`` (produced by
``examples/generate_credit_card_roc.py``), fits a calibration pipeline on the
calibration-split benign scores, and writes a four-panel validation figure to
``examples/credit_card_validation.png``.

The four panels:

1. ROC: TPR vs FPR on the eval split (log x-axis).
2. FPR -> threshold isotonic fit with sampled knots overlaid.
3. Calibrated score vs FPR: the fixed log-scale contract vs pipeline output on
   eval points.
4. Relative calibration error (%) across FPR.
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


def plot_four_panels(
    calib_benign: np.ndarray,
    eval_scores: np.ndarray,
    eval_labels: np.ndarray,
    pipeline,
    save_path: Path,
) -> None:
    """Four-panel calibration-quality figure on the eval split."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    eval_benign = eval_scores[eval_labels == 0]
    eval_attack = eval_scores[eval_labels == 1]
    eval_thresh, eval_fpr, eval_tpr = compute_empirical_roc(eval_benign, eval_attack)
    min_fpr_calib = 1.0 / len(calib_benign)

    # Panel 1: ROC.
    ax = axes[0, 0]
    mask = eval_fpr > 0
    ax.step(
        eval_fpr[mask],
        eval_tpr[mask],
        where="post",
        color="orange",
        linewidth=1.5,
        alpha=0.9,
        label="Eval (RandomForest)",
    )
    ax.axvline(
        min_fpr_calib,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Calibration min FPR ({min_fpr_calib:.1e})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("TPR")
    ax.set_title("ROC on Credit Card Fraud eval split")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 2: FPR -> threshold isotonic fit.
    ax = axes[0, 1]
    mask_fit = (eval_fpr > 0) & (eval_thresh > 0) & (eval_thresh < 1)
    ax.plot(
        eval_fpr[mask_fit],
        eval_thresh[mask_fit],
        ".",
        color="orange",
        markersize=2,
        alpha=0.5,
        label="Eval",
    )

    fpr_knots = pipeline.fpr_to_score_.X_thresholds_
    score_knots = pipeline.fpr_to_score_.y_thresholds_
    fpr_grid = np.logspace(-6, 0, 200)
    fpr_grid_clipped = np.clip(fpr_grid, fpr_knots.min(), fpr_knots.max())
    score_fit = np.interp(fpr_grid_clipped, fpr_knots, score_knots)
    ax.plot(
        fpr_grid,
        score_fit,
        "-",
        color="red",
        linewidth=2,
        alpha=0.8,
        label="Isotonic fit (calibration split)",
    )
    ax.plot(
        pipeline.sampled_fprs_,
        pipeline.sampled_scores_,
        "go",
        markersize=4,
        alpha=0.7,
        label=f"Sampled knots ({len(pipeline.sampled_fprs_)})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("Threshold")
    ax.set_title("FPR -> threshold (isotonic fit)")
    ax.set_xlim(1e-6, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: expected vs actual calibrated score.
    ax = axes[1, 0]
    expected_fprs = np.logspace(-6, 0, 200)
    expected_cal = fpr_to_calibrated(expected_fprs)
    ax.plot(
        expected_fprs, expected_cal, "r--", linewidth=2, alpha=0.8, label="Expected (FPR -> cal)"
    )
    eval_mask = (eval_fpr > 0) & (eval_thresh > 0) & (eval_thresh < 1)
    actual_cal = pipeline.predict(eval_thresh[eval_mask].reshape(-1, 1))
    ax.plot(eval_fpr[eval_mask], actual_cal, "b.", markersize=2, alpha=0.5, label="Pipeline (eval)")
    ax.set_xscale("log")
    ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("Calibrated score")
    ax.set_title("Calibration: expected vs actual")
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 4: relative calibration error.
    ax = axes[1, 1]
    expected_at_fpr = fpr_to_calibrated(eval_fpr[eval_mask])
    valid = expected_at_fpr > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (actual_cal - expected_at_fpr) / expected_at_fpr * 100
    rel_err = np.nan_to_num(rel_err, nan=0, posinf=0, neginf=0)
    fpr_valid = eval_fpr[eval_mask][valid]
    err_valid = rel_err[valid]
    order = np.argsort(fpr_valid)
    ax.plot(
        fpr_valid[order],
        err_valid[order],
        "-",
        color="blue",
        linewidth=1,
        alpha=0.7,
        label="Pipeline (eval)",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
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
    calib_scores = data["calib_scores"]
    calib_labels = data["calib_labels"]
    eval_scores = data["eval_scores"]
    eval_labels = data["eval_labels"]

    calib_benign = calib_scores[calib_labels == 0]
    print(
        f"Calibration split: {len(calib_scores):,} rows "
        f"({len(calib_benign):,} benign, {int(calib_labels.sum())} positive)."
    )
    print(
        f"Eval split:        {len(eval_scores):,} rows "
        f"({int((eval_labels == 0).sum()):,} benign, {int(eval_labels.sum())} positive)."
    )

    pipeline = fit_calibration_pipeline(calib_benign, n_knots=10000)
    plot_four_panels(calib_benign, eval_scores, eval_labels, pipeline, FIGURE)


if __name__ == "__main__":
    main()
