# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate detector scores on Credit Card Fraud for the calibration demo.

Downloads the Credit Card Fraud Detection dataset from OpenML (data_id=1597),
trains two detectors (HistGradientBoosting and LogisticRegression) on a
stratified 30% split, and scores the full 70% holdout. Saves holdout scores
and labels as an NPZ file along with a boolean mask identifying which holdout
rows are used to fit the calibration pipeline (30% of the holdout,
stratified).

Two detectors are produced so the paper can show the calibration contract
holds across different classifier families: a non-parametric boosted-tree
detector and a linear detector. Both expose continuous scores.

The calibration pipeline is fit on benign rows selected by the fit mask.
The complementary held-out-from-fit rows supply the paper's evaluation table
and primary figure curves, while the first two panels show the fit subset for
comparison.

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


def main() -> None:
    print("Fetching Credit Card Fraud Detection (OpenML data_id=1597)...")
    data = fetch_openml(name="creditcard", version=1, as_frame=False, parser="liac-arff")
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64).ravel()

    n_total = len(y)
    n_pos = int(y.sum())
    print(f"Loaded {n_total:,} rows, {n_pos} positives (rate {n_pos / n_total:.4%})")

    # Two-level stratified split:
    #   - 30% trains the detector.
    #   - 70% is scored and then split between calibration fit and evaluation.
    # Within the holdout, 30% is carved out (stratified) to fit the
    # calibration pipeline. The complementary 70% supplies the primary
    # evaluation curves and table, with the fit subset shown separately in
    # the first two figure panels.
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=0.70, stratify=y, random_state=SEED
    )
    holdout_idx = np.arange(len(y_holdout))
    fit_idx, _ = train_test_split(
        holdout_idx, test_size=0.70, stratify=y_holdout, random_state=SEED + 1
    )
    fit_mask = np.zeros(len(y_holdout), dtype=bool)
    fit_mask[fit_idx] = True

    print(
        f"Splits: train={len(y_train):,} ({int(y_train.sum())} pos), "
        f"holdout={len(y_holdout):,} ({int(y_holdout.sum())} pos), "
        f"fit-subset (of holdout)={int(fit_mask.sum()):,} "
        f"({int(y_holdout[fit_mask].sum())} pos)"
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
    gbdt_holdout_scores = gbdt.predict_proba(x_holdout)[:, 1].astype(np.float64)

    # --- LogisticRegression ---
    print("Training LogisticRegression(C=1.0) with StandardScaler...")
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]
    )
    lr.fit(x_train, y_train)
    lr_holdout_scores = lr.predict_proba(x_holdout)[:, 1].astype(np.float64)

    np.savez_compressed(
        OUTPUT,
        holdout_scores_gbdt=gbdt_holdout_scores,
        holdout_scores_lr=lr_holdout_scores,
        holdout_labels=y_holdout.astype(np.int8),
        fit_mask=fit_mask,
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
