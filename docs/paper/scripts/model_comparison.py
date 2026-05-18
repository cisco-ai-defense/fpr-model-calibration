"""Compare GBDT and Logistic Regression calibration on Credit Card Fraud.

Loads scores from ``examples/credit_card_roc.npz`` (produced with a 30/70
train/holdout split; calibration fit on a 30% stratified subset of the
holdout). Fits a separate calibration pipeline for each model on the
fit-subset benign scores and evaluates on the full holdout.

For each (model, target_fpr) cell reports:
- calibration threshold picked on the fit-subset benign set
- achieved FPR on the full holdout
- TPR on the full holdout attacks
- expected vs actual calibrated value
- relative calibration error

Also reports the score-saturation floor: the smallest FPR reachable before
the raw detector score hits its asymptote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fpr_model_calibration.calibration import fit_calibration_pipeline

ROC_FILE = Path(__file__).resolve().parents[3] / "examples" / "credit_card_roc.npz"


def evaluate(
    name: str,
    calib_benign: np.ndarray,
    eval_benign: np.ndarray,
    eval_attack: np.ndarray,
    anchors: list[tuple[float, float]],
) -> None:
    print(f"\n# {name}")
    print(f"  Calib benign: {len(calib_benign):,} ({len(np.unique(calib_benign))} unique scores)")
    print(f"  Eval benign:  {len(eval_benign):,} ({len(np.unique(eval_benign))} unique scores)")
    print(
        f"  Raw score: min={eval_benign.min():.3e}, "
        f"max={eval_benign.max():.3e}, "
        f"frac at max: {np.mean(eval_benign == eval_benign.max()):.3%}"
    )

    pipeline = fit_calibration_pipeline(calib_benign, n_knots=10000)

    header = (
        f"  {'Target FPR':>10} {'Threshold':>10} {'Eval FPR':>10} "
        f"{'Eval TPR':>10} {'Exp cal':>8} {'Act cal':>8} {'Rel err':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for target_fpr, expected_cal in anchors:
        sorted_cb = np.sort(calib_benign)
        n = len(sorted_cb)
        idx = int(np.clip(np.ceil((1 - target_fpr) * n) - 1, 0, n - 1))
        threshold = sorted_cb[idx]
        eval_fpr = float(np.mean(eval_benign >= threshold))
        eval_tpr = float(np.mean(eval_attack >= threshold)) if len(eval_attack) else 0.0
        actual_cal = float(pipeline.predict(np.array([[threshold]]))[0])
        rel_err = (actual_cal - expected_cal) / expected_cal * 100
        print(
            f"  {target_fpr:>10.0%} {threshold:>10.4f} {eval_fpr:>10.4%} "
            f"{eval_tpr:>10.3f} {expected_cal:>8.2f} {actual_cal:>8.3f} "
            f"{rel_err:>+9.2f}%"
        )

    # Score-saturation floor: smallest FPR reachable before the raw score
    # hits its asymptote.
    score_max = calib_benign.max()
    n_at_cap = int(np.sum(calib_benign >= score_max))
    floor_fpr = n_at_cap / len(calib_benign)
    cal_at_cap = float(pipeline.predict(np.array([[score_max]]))[0])
    print(
        f"  Saturation floor: {n_at_cap}/{len(calib_benign)} benign "
        f"hit score={score_max:.4f} (FPR floor {floor_fpr:.2e}, "
        f"calibrated {cal_at_cap:.3f})"
    )


def customer_draws(
    name: str,
    calib_benign: np.ndarray,
    eval_benign: np.ndarray,
    n_customer: int,
    n_draws: int,
    calibrated_threshold: float,
    rng: np.random.Generator,
) -> None:
    """Simulate n_draws customers each getting n_customer eval-benign samples.

    Measures the per-customer FPR distribution at the given calibrated
    threshold. Demonstrates that the calibration contract holds in
    expectation across customer-scale deployments, and that individual
    customer FPR has Binomial sampling variance around the target.
    """
    pipeline = fit_calibration_pipeline(calib_benign, n_knots=10000)
    eval_cal = pipeline.predict(eval_benign.reshape(-1, 1))

    customer_fprs = np.array(
        [
            np.mean(rng.choice(eval_cal, size=n_customer, replace=False) >= calibrated_threshold)
            for _ in range(n_draws)
        ]
    )
    print(f"\n# {name}: {n_draws} customer draws of {n_customer:,} benign each")
    print(f"  Calibrated threshold: {calibrated_threshold}")
    print(
        f"  Per-customer FPR: mean={customer_fprs.mean():.4%}, "
        f"std={customer_fprs.std():.4%}, "
        f"p05-p95=[{np.percentile(customer_fprs, 5):.4%}, "
        f"{np.percentile(customer_fprs, 95):.4%}]"
    )


def main() -> None:
    data = np.load(ROC_FILE)
    labels = data["holdout_labels"]
    fit_mask = data["fit_mask"]

    gbdt = data["holdout_scores_gbdt"]
    lr = data["holdout_scores_lr"]

    gbdt_cb = gbdt[(labels == 0) & fit_mask]
    gbdt_eb = gbdt[labels == 0]
    gbdt_ea = gbdt[labels == 1]

    lr_cb = lr[(labels == 0) & fit_mask]
    lr_eb = lr[labels == 0]
    lr_ea = lr[labels == 1]

    anchors = [(0.10, 0.10), (0.01, 0.30), (0.001, 0.50), (0.0001, 0.70)]

    evaluate("GBDT (HistGradientBoosting)", gbdt_cb, gbdt_eb, gbdt_ea, anchors)
    evaluate("Logistic Regression", lr_cb, lr_eb, lr_ea, anchors)

    # Customer-draw simulation: each customer observes 10K benign samples,
    # 500 draws to characterize the per-customer FPR distribution at the
    # block threshold (calibrated = 0.5 -> target 0.1% FPR).
    rng = np.random.default_rng(0)
    customer_draws(
        "GBDT", gbdt_cb, gbdt_eb, n_customer=10000, n_draws=500, calibrated_threshold=0.5, rng=rng
    )
    customer_draws(
        "Logistic Regression",
        lr_cb,
        lr_eb,
        n_customer=10000,
        n_draws=500,
        calibrated_threshold=0.5,
        rng=rng,
    )


if __name__ == "__main__":
    main()
