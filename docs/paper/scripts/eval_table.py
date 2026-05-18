"""Compute the eval-split Table 3 numbers for the paper.

For each target FPR in {10%, 1%, 0.1%, 0.01%}, find the empirical threshold
on the calibration-fit subset benign set that hits that FPR, then measure
actual FPR and TPR on the full holdout. Apply the released calibration
pipeline to the threshold and compare against the expected calibrated value
from the log-scale anchor table.

The holdout is scored end-to-end; the calibration-fit subset is a 30%
stratified slice of the holdout used to fit the pipeline. Reported FPR/TPR
are computed on the full holdout (which contains both fit and held-out-from-
fit rows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fpr_model_calibration.calibration import fit_calibration_pipeline

ROC_FILE = Path(__file__).resolve().parents[3] / "examples" / "credit_card_roc.npz"


def main() -> None:
    data = np.load(ROC_FILE)
    scores = data["holdout_scores_lr"]
    labels = data["holdout_labels"]
    fit_mask = data["fit_mask"]

    calib_benign = scores[(labels == 0) & fit_mask]
    # Eval on the held-out-from-fit rows only, so reported FPR is not biased
    # by the fit-subset (whose FPR is near the target by construction).
    eval_benign = scores[(labels == 0) & ~fit_mask]
    eval_attack = scores[(labels == 1) & ~fit_mask]

    pipeline = fit_calibration_pipeline(calib_benign, n_knots=10000)

    anchors = [
        (0.10, 0.10),
        (0.01, 0.30),
        (0.001, 0.50),
        (0.0001, 0.70),
    ]

    print(f"Calibration-fit benign: {len(calib_benign):,}")
    print(f"Held-out-from-fit benign: {len(eval_benign):,}")
    print(f"Held-out-from-fit attack: {len(eval_attack):,}")
    print()
    header = (
        f"{'Target FPR':>10} {'Threshold':>10} {'Eval FPR':>12} {'Eval TPR':>10} "
        f"{'Exp cal':>8} {'Act cal':>8} {'Rel err %':>10}"
    )
    print(header)
    print("-" * len(header))

    for target_fpr, expected_cal in anchors:
        # Threshold on calibration set that hits target_fpr.
        sorted_cb = np.sort(calib_benign)
        n = len(sorted_cb)
        idx = int(np.clip(np.ceil((1 - target_fpr) * n) - 1, 0, n - 1))
        threshold = sorted_cb[idx]

        # Measure on eval split.
        eval_fpr = float(np.mean(eval_benign >= threshold))
        eval_tpr = float(np.mean(eval_attack >= threshold))

        # Apply pipeline to threshold.
        actual_cal = float(pipeline.predict(np.array([[threshold]]))[0])

        rel_err = (actual_cal - expected_cal) / expected_cal * 100

        print(
            f"{target_fpr:>10.0%} {threshold:>10.4f} {eval_fpr:>12.4%} "
            f"{eval_tpr:>10.3f} {expected_cal:>8.2f} {actual_cal:>8.3f} "
            f"{rel_err:>+10.2f}"
        )


if __name__ == "__main__":
    main()
