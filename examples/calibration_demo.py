"""Fit FPR calibration on the Credit Card Fraud ROC and produce the demo figure.

Reads ``examples/credit_card_roc.npz`` (produced by
``examples/generate_credit_card_roc.py``), fits a calibration pipeline on
benign scores from the 30% calibration-fit subset of the holdout, and writes
a four-panel validation figure and LaTeX table to ``examples/``.
It also copies both generated artifacts into ``docs/paper/figures/`` so the
paper cannot drift from the latest evaluation run.

All four panels are evaluated on the full 70% holdout. The calibration-fit
subset is shown alongside the full holdout on the ROC and FPR-threshold
panels so the reader can confirm the fit subset is representative of the
full holdout distribution.

The four panels:

1. ROC (TPR vs FPR, log x-axis): full holdout vs calibration-fit subset.
2. FPR -> threshold: same two populations, plus the fitted isotonic line.
3. Calibrated score vs empirical FPR: expected contract vs pipeline output
   on the full holdout.
4. Relative FPR error (%) across FPR on the full holdout.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fpr_model_calibration.calibration import fit_calibration_pipeline, fpr_to_calibrated

ROC_FILE = Path(__file__).parent / "credit_card_roc.npz"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE = Path(__file__).parent / "credit_card_validation.png"
TABLE = Path(__file__).parent / "credit_card_eval_table.tex"
PAPER_FIGURE = REPO_ROOT / "docs" / "paper" / "figures" / FIGURE.name
PAPER_TABLE = REPO_ROOT / "docs" / "paper" / "figures" / TABLE.name

_CAL_TO_LOG_FPR_X = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99])
_CAL_TO_LOG_FPR_Y = np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -10.0])
_TABLE_TARGET_FPRS = (1e-1, 1e-2, 1e-3, 1e-4)


def calibrated_to_fpr(calibrated: np.ndarray) -> np.ndarray:
    """Invert the paper's calibrated-score contract back to FPR."""
    calibrated = np.asarray(calibrated, dtype=np.float64)
    log_fpr = np.interp(
        np.clip(calibrated, _CAL_TO_LOG_FPR_X[0], _CAL_TO_LOG_FPR_X[-1]),
        _CAL_TO_LOG_FPR_X,
        _CAL_TO_LOG_FPR_Y,
    )
    return 10.0**log_fpr


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

    # Panel 4: relative FPR error on the full holdout.
    ax = axes[1, 1]
    predicted_fpr = calibrated_to_fpr(actual_cal)
    empirical_fpr = ho_fpr[eval_mask]
    valid = empirical_fpr > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (predicted_fpr - empirical_fpr) / empirical_fpr * 100
    rel_err = np.nan_to_num(rel_err, nan=0, posinf=0, neginf=0)
    fpr_valid = empirical_fpr[valid]
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
    ax.set_ylabel("Relative FPR error (%)")
    ax.set_title("FPR calibration error")
    ax.set_xlim(1e-6, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {save_path}")


def _threshold_for_calibrated_score(pipeline, target_calibrated: float) -> float:
    """Find the raw score threshold whose calibrated score equals target."""
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        calibrated = float(pipeline.predict(np.array([[mid]], dtype=np.float64))[0])
        if calibrated < target_calibrated:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _format_target_fpr(fpr: float) -> str:
    if fpr >= 1e-1:
        return f"{fpr * 100:.0f}\\%"
    if fpr >= 1e-2:
        return f"{fpr * 100:.0f}\\%"
    if fpr >= 1e-3:
        return f"{fpr * 100:.1f}\\%"
    return f"{fpr * 100:.2f}\\%"


def _format_heldout_fpr(fpr: float) -> str:
    if fpr >= 1e-2:
        return f"{fpr * 100:.3f}\\%"
    return f"{fpr * 100:.4f}\\%"


def _latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def write_eval_table(
    holdout_scores: np.ndarray,
    holdout_labels: np.ndarray,
    fit_mask: np.ndarray,
    pipeline,
    save_path: Path,
) -> None:
    """Write the held-out-from-fit evaluation table used by the paper."""
    eval_mask = ~fit_mask
    eval_scores = holdout_scores[eval_mask]
    eval_labels = holdout_labels[eval_mask]
    train_scores = holdout_scores[fit_mask]
    train_labels = holdout_labels[fit_mask]
    train_benign = train_scores[train_labels == 0]
    benign = eval_scores[eval_labels == 0]

    rows: list[str] = []
    for target_fpr in _TABLE_TARGET_FPRS:
        target_calibrated = float(fpr_to_calibrated(target_fpr))
        threshold = _threshold_for_calibrated_score(pipeline, target_calibrated)
        train_fpr = float(np.mean(train_benign >= threshold))
        heldout_fpr = float(np.mean(benign >= threshold))
        actual_calibrated = float(pipeline.predict(np.array([[threshold]], dtype=np.float64))[0])
        predicted_fpr = float(calibrated_to_fpr(np.array([actual_calibrated]))[0])
        rel_fpr_error = (predicted_fpr - heldout_fpr) / heldout_fpr * 100.0
        rows.append(
            "    "
            f"{target_calibrated:.2f} & "
            f"{_format_target_fpr(target_fpr)} & "
            f"{threshold:.4f} & "
            f"{_format_heldout_fpr(train_fpr)} & "
            f"{_format_heldout_fpr(heldout_fpr)} & "
            f"${rel_fpr_error:+.2f}\\%$ \\\\"
        )

    table = "\n".join(
        [
            "% Generated by examples/calibration_demo.py; do not edit by hand.",
            "\\begin{table}[!htb]",
            "  \\centering",
            "  \\caption{Training and held-out FPR at calibrated score anchors.",
            f"  Training FPR uses the {_latex_int(len(train_benign))} calibration-fit benign rows; held-out FPR uses the {_latex_int(len(benign))} benign rows not used to fit calibration.",
            "  Negative relative error means held-out FPR was above target.}",
            "  \\label{tab:eval}",
            "  \\begin{tabular}{rrrrrr}",
            "    \\toprule",
            "    Calibrated score & Target FPR & Raw score & Training FPR & Held-out FPR & Rel. error \\\\",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    save_path.write_text(table, encoding="utf-8")
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
    write_eval_table(holdout_scores, holdout_labels, fit_mask, pipeline, TABLE)
    shutil.copy2(FIGURE, PAPER_FIGURE)
    shutil.copy2(TABLE, PAPER_TABLE)
    print(f"Copied {FIGURE} to {PAPER_FIGURE}")
    print(f"Copied {TABLE} to {PAPER_TABLE}")


if __name__ == "__main__":
    main()
