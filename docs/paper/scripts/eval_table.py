"""Reproduce the independent evaluation numbers reported in Table 3.

The calibration pipeline is fit on benign rows from a 30% stratified slice of
the detector holdout. For each calibrated anchor, this script finds the raw
threshold from the fitted pipeline and measures FPR and TPR on the disjoint
held-out-from-fit complement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fprcal.calibration import fit_calibration_pipeline, fpr_to_calibrated

ROC_FILE = Path(__file__).resolve().parents[3] / "examples" / "credit_card_roc.npz"


def threshold_for_calibrated_score(pipeline, target_calibrated: float) -> float:
    """Find the raw-score threshold for one calibrated anchor."""
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


def main() -> None:
    data = np.load(ROC_FILE)
    scores = data["holdout_scores_lr"]
    labels = data["holdout_labels"]
    fit_mask = data["fit_mask"]

    fit_benign = scores[(labels == 0) & fit_mask]
    eval_benign = scores[(labels == 0) & ~fit_mask]
    eval_attack = scores[(labels == 1) & ~fit_mask]

    pipeline = fit_calibration_pipeline(fit_benign, n_knots=10000)

    target_fprs = (0.10, 0.01, 0.001, 0.0001)

    print(f"Calibration-fit benign: {len(fit_benign):,}")
    print(f"Held-out-from-fit benign: {len(eval_benign):,}")
    print(f"Held-out-from-fit attack: {len(eval_attack):,}")
    print()
    header = (
        f"{'Target FPR':>10} {'Threshold':>10} {'Fit FPR':>12} "
        f"{'Eval FPR':>12} {'Eval TPR':>10} {'Rel err %':>10}"
    )
    print(header)
    print("-" * len(header))

    for target_fpr in target_fprs:
        target_calibrated = float(fpr_to_calibrated(target_fpr))
        threshold = threshold_for_calibrated_score(pipeline, target_calibrated)
        fit_fpr = float(np.mean(fit_benign >= threshold))
        eval_fpr = float(np.mean(eval_benign >= threshold))
        eval_tpr = float(np.mean(eval_attack >= threshold))
        rel_err = (target_fpr - eval_fpr) / eval_fpr * 100.0

        print(
            f"{target_fpr:>10.4%} {threshold:>10.4f} {fit_fpr:>12.4%} "
            f"{eval_fpr:>12.4%} {eval_tpr:>10.3f} {rel_err:>+10.2f}"
        )


if __name__ == "__main__":
    main()
