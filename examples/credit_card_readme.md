# Credit Card Fraud Detection ROC

This directory holds the ROC points and derived calibration figure produced from the publicly available **Credit Card Fraud Detection** dataset. The repository does not redistribute the dataset itself; only the derived score arrays are committed so the calibration demo runs without any download.

## Dataset

- **Source:** [OpenML data_id 1597](https://www.openml.org/d/1597) (mirrored from the original Kaggle release at `mlg-ulb/creditcardfraud`).
- **Authors:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
- **License:** Database Contents License (DbCL v1.0) — permissive, allows redistribution with attribution.
- **Size:** 284,807 transactions over two days, 492 fraud cases (0.172%, or 1 in 579).
- **Features:** 28 PCA-anonymized numerical features (`V1`–`V28`), plus `Time` and `Amount`. Label column `Class` with 1 = fraud, 0 = benign.

The positive rate of 0.172% sits in the window where whole-curve FPR calibration matters most: too rare for probability calibration to work reliably, common enough that a production system sees fraud on a daily basis.

## Pipeline

The ROC points in `credit_card_roc.npz` were produced by `generate_credit_card_roc.py`:

1. Load the dataset via `sklearn.datasets.fetch_openml(name="creditcard", version=1)`. No authentication required; the loader caches to `~/scikit_learn_data` on first use.
2. Stratified three-way split (seed 42, preserving the positive rate in every split):
   - **Train** (30%, 85,442 rows): fit the detector.
   - **Calibration** (30%, 85,442 rows): benign scores feed `fit_calibration_pipeline`.
   - **Eval** (40%, 113,923 rows): the held-out set used for the validation plot.
3. Train a `RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)` on the train split. 1000 trees gives enough granularity for FPR levels down to 10⁻⁴.
4. Score the calibration and eval splits via `predict_proba(...)[:, 1]` and save both arrays (scores + labels) to `credit_card_roc.npz` with `np.savez_compressed`.

Split counts that land in the NPZ:

| Split | Rows | Benign | Fraud |
|---|---|---|---|
| Calibration | 85,442 | 85,295 | 147 |
| Eval        | 113,923 | 113,726 | 197 |

The eval split has 113,726 benign samples, so the empirically observable FPR floor is ≈ 8.8 × 10⁻⁶. FPR estimation precision follows `n ≈ 16/p` for ±50% at 95% confidence, which means the plot resolves calibration cleanly down to FPR ≈ 1.4 × 10⁻⁴.

## Regenerating from scratch

```bash
python examples/generate_credit_card_roc.py
python examples/calibration_demo.py
```

The first command downloads the dataset on first run (about 67 MB), trains the model (under two minutes on a laptop CPU), and writes `credit_card_roc.npz`. The second reads the NPZ, fits calibration on the benign calibration-split scores, writes `credit_card_validation.png` and `credit_card_eval_table.tex`, and copies both artifacts into `docs/paper/figures/`.

The generator is deterministic given the seed; a second run produces an identical NPZ file.

## Attribution

If you redistribute the derived NPZ outside this repository, credit the original authors per DbCL v1.0:

> Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium on Computational Intelligence and Data Mining*, 159-166.
