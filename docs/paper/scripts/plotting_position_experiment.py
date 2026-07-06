"""Compare naive k/n vs Filliben plotting position on Credit Card Fraud.

Fits two calibration pipelines on the same benign calibration split, one
with each plotting position, and measures relative calibration error at
each anchor FPR on the held-out eval split.

Naive:   FPR_k = 1 - k/n     (what the released code does)
Filliben: FPR_k = 1 - (k - 0.3175) / (n + 0.365)

The Filliben formula is a median-unbiased approximation to the Beta plotting
position of the k-th order statistic of a uniform sample of size n.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from fpr_model_calibration.calibration import (
    _CAL_KNOTS,
    _LOG_FPR_KNOTS,
    SCORE_MAX,
    _extrapolate_score,
    _sample_fpr_values,
)

ROC_FILE = Path(__file__).resolve().parents[3] / "examples" / "credit_card_roc.npz"


def fit_with_position(
    benign_scores: np.ndarray,
    position: str,
    n_knots: int = 10000,
) -> Pipeline:
    """Fit calibration pipeline with a chosen plotting position.

    position: 'naive' for 1 - k/n, 'filliben' for Filliben order-statistic medians.
    """
    benign_scores = np.asarray(benign_scores, dtype=np.float64).ravel()
    rescaler = MinMaxScaler(feature_range=(0, SCORE_MAX))
    scores_scaled = rescaler.fit_transform(benign_scores.reshape(-1, 1)).ravel()

    sorted_scores = np.sort(scores_scaled)
    n = len(sorted_scores)
    k = np.arange(1, n + 1)

    if position == "naive":
        fprs = 1 - k / n
    elif position == "filliben":
        fprs = 1 - (k - 0.3175) / (n + 0.365)
        endpoint = 0.5 ** (1 / n)
        fprs[0] = endpoint
        fprs[-1] = 1 - endpoint
    else:
        raise ValueError(f"unknown position: {position}")

    fpr_to_score = IsotonicRegression(increasing=False, out_of_bounds="clip")
    fpr_to_score.fit(fprs, sorted_scores)

    min_fpr = fprs.min()
    fpr1, fpr2 = fprs[-2], fprs[-1]
    score1, score2 = sorted_scores[-2], sorted_scores[-1]

    sampled_fprs = _sample_fpr_values(n_knots)
    first_spline_edge_fprs = np.array([fprs[0], fpr1, fpr2], dtype=np.float64)
    sampled_fprs = np.unique(np.concatenate((sampled_fprs, first_spline_edge_fprs)))
    knot_scores, knot_cal = [], []
    for fpr in sampled_fprs:
        if fpr >= min_fpr:
            score = float(fpr_to_score.predict([[fpr]])[0])
        else:
            score = _extrapolate_score(fpr, fpr1, score1, fpr2, score2)
            score = float(np.clip(score, 0, SCORE_MAX))
        log_fpr = np.log10(max(fpr, 1e-10))
        cal = float(np.interp(log_fpr, _LOG_FPR_KNOTS, _CAL_KNOTS))
        knot_scores.append(score)
        knot_cal.append(cal)

    knot_scores_arr = np.append(np.array(knot_scores), [SCORE_MAX, 1.0])
    knot_cal_arr = np.append(np.array(knot_cal), [0.99, 1.0])

    score_to_cal = IsotonicRegression(increasing=True, out_of_bounds="clip")
    score_to_cal.fit(knot_scores_arr, knot_cal_arr)

    return Pipeline([("rescale", rescaler), ("isotonic", score_to_cal)])


def threshold_for_target_fpr(
    pipeline: Pipeline, eval_benign: np.ndarray, target_fpr: float
) -> tuple[float, float]:
    """Walk pipeline output to find the calibrated score corresponding to the
    target FPR on the eval benign set, then compute the achieved FPR.

    Returns (calibrated_at_target, achieved_fpr).
    """
    cal = pipeline.predict(eval_benign.reshape(-1, 1))
    sorted_cal = np.sort(cal)
    n = len(sorted_cal)
    # Count how many eval-benign scores exceed each threshold; pick the
    # smallest calibrated threshold whose achieved FPR is <= target.
    achieved = np.arange(n, 0, -1) / n  # FPR at each sorted calibrated score
    idx = np.searchsorted(-achieved, -target_fpr, side="left")
    idx = min(idx, n - 1)
    return float(sorted_cal[idx]), float(achieved[idx])


def main() -> None:
    data = np.load(ROC_FILE)
    scores = data["holdout_scores_lr"]
    labels = data["holdout_labels"]
    fit_mask = data["fit_mask"]
    calib_benign = scores[(labels == 0) & fit_mask]
    eval_benign = scores[(labels == 0) & ~fit_mask]

    anchors = {
        0.10: 0.10,
        0.01: 0.30,
        0.001: 0.50,
        0.0001: 0.70,
    }

    print(f"Calibration benign n = {len(calib_benign):,}")
    print(f"Eval benign n        = {len(eval_benign):,}")
    print()

    rows = []
    for position in ("naive", "filliben"):
        pipeline = fit_with_position(calib_benign, position)
        eval_cal = pipeline.predict(eval_benign.reshape(-1, 1))

        # For each anchor FPR, measure the eval-benign fraction at or above
        # the anchor-expected calibrated value.
        for target_fpr, expected_cal in anchors.items():
            # Achieved FPR at this calibrated threshold on eval:
            achieved = float(np.mean(eval_cal >= expected_cal))
            rel_err = (achieved - target_fpr) / target_fpr * 100
            rows.append(
                {
                    "position": position,
                    "target_fpr": target_fpr,
                    "expected_cal": expected_cal,
                    "achieved_fpr": achieved,
                    "rel_err_pct": rel_err,
                }
            )

    print(f"{'position':<10} {'target':>10} {'cal':>6} {'achieved':>12} {'rel_err %':>10}")
    for r in rows:
        print(
            f"{r['position']:<10} {r['target_fpr']:>10.1e} {r['expected_cal']:>6.2f} "
            f"{r['achieved_fpr']:>12.4e} {r['rel_err_pct']:>+10.2f}"
        )

    print()
    print("# Delta: naive - filliben, relative error in percentage points")
    print(f"{'target_fpr':>10} {'naive%':>10} {'filliben%':>12} {'delta pp':>10}")
    for tgt in anchors:
        n_err = next(
            r["rel_err_pct"] for r in rows if r["position"] == "naive" and r["target_fpr"] == tgt
        )
        f_err = next(
            r["rel_err_pct"] for r in rows if r["position"] == "filliben" and r["target_fpr"] == tgt
        )
        print(f"{tgt:>10.1e} {n_err:>+10.2f} {f_err:>+12.2f} {n_err - f_err:>+10.2f}")


if __name__ == "__main__":
    main()
