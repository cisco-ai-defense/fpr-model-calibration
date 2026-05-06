"""Generate detector scores on Credit Card Fraud for the calibration demo.

Downloads the Credit Card Fraud Detection dataset from OpenML (data_id=1597),
trains two detectors (HistGradientBoosting and LogisticRegression) on a
stratified 30% split, scores the held-out calibration (30%) and eval (40%)
splits, and saves the (score, label) pairs as an NPZ file.

Two models are produced so the paper can show the calibration contract holds
across different classifier families: a non-parametric boosted-tree detector
and a linear detector. Both expose continuous scores.

Run this once to produce ``examples/credit_card_roc.npz``. The downstream demo
``examples/calibration_demo.py`` reads the NPZ, so users do not need to rerun
the models to explore calibration.

Provenance and license notes live in ``examples/credit_card_readme.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MAX_ITER = 500
OUTPUT = Path(__file__).parent / "credit_card_roc.npz"


def _score_dict(model, x_calib, x_eval) -> dict[str, np.ndarray]:
    return {
        "calib": model.predict_proba(x_calib)[:, 1].astype(np.float64),
        "eval": model.predict_proba(x_eval)[:, 1].astype(np.float64),
    }


def main() -> None:
    print("Fetching Credit Card Fraud Detection (OpenML data_id=1597)...")
    data = fetch_openml(name="creditcard", version=1, as_frame=False, parser="liac-arff")
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64).ravel()

    n_total = len(y)
    n_pos = int(y.sum())
    print(f"Loaded {n_total:,} rows, {n_pos} positives (rate {n_pos / n_total:.4%})")

    # Three-way stratified split, biased toward the held-out evaluation
    # tail so sub-0.01% FPR can be resolved:
    #   - 30% for model training (with the model's own internal validation
    #     holdout, e.g. HistGradientBoosting's 10% early-stopping split).
    #   - 20% for the calibration pipeline, fit on benign scores only.
    #   - 50% for eval, used for every reported number and figure.
    # At 284,807 rows the eval set is ~142K benign, enough for a Clopper-
    # Pearson +/- 50% window at FPR ~10^-4 (Table 2 in the paper).
    x_train, x_cal_eval, y_train, y_cal_eval = train_test_split(
        x, y, test_size=0.70, stratify=y, random_state=SEED
    )
    # Within the 70% ROC block: 20/70 = calibration, 50/70 = eval.
    x_calib, x_eval, y_calib, y_eval = train_test_split(
        x_cal_eval, y_cal_eval, test_size=5 / 7, stratify=y_cal_eval, random_state=SEED
    )

    print(
        f"Splits: train={len(y_train):,} ({int(y_train.sum())} pos), "
        f"calib={len(y_calib):,} ({int(y_calib.sum())} pos), "
        f"eval={len(y_eval):,} ({int(y_eval.sum())} pos)"
    )

    # --- HistGradientBoosting ---
    print(f"Training HistGradientBoostingClassifier(max_iter={MAX_ITER})...")
    gbdt = HistGradientBoostingClassifier(
        max_iter=MAX_ITER,
        learning_rate=0.10,
        max_depth=6,
        random_state=SEED,
    )
    gbdt.fit(x_train, y_train)
    print(f"  Boosting rounds (early-stopped): {gbdt.n_iter_}")
    gbdt_scores = _score_dict(gbdt, x_calib, x_eval)

    # --- LogisticRegression ---
    print("Training LogisticRegression(C=1.0) with StandardScaler...")
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]
    )
    lr.fit(x_train, y_train)
    lr_scores = _score_dict(lr, x_calib, x_eval)

    np.savez_compressed(
        OUTPUT,
        # Primary scores: GBDT (used by calibration_demo.py).
        calib_scores=gbdt_scores["calib"],
        calib_labels=y_calib.astype(np.int8),
        eval_scores=gbdt_scores["eval"],
        eval_labels=y_eval.astype(np.int8),
        # Secondary scores: Logistic Regression (used for cross-model comparison).
        calib_scores_lr=lr_scores["calib"],
        eval_scores_lr=lr_scores["eval"],
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
