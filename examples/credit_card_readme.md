# Credit Card Fraud Detection evaluation

This directory holds detector scores and derived calibration artifacts produced from the publicly available **Credit Card Fraud Detection** dataset. The repository does not redistribute the dataset itself; only the derived score arrays are committed so the calibration demo runs without downloading the source data.

## Dataset

- **Source:** [OpenML data_id 1597](https://www.openml.org/d/1597) (mirrored from the original Kaggle release at `mlg-ulb/creditcardfraud`).
- **Authors:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
- **License:** Database Contents License (DbCL v1.0) — permissive, allows redistribution with attribution.
- **Size:** 284,807 transactions over two days, 492 fraud cases (0.172%, or 1 in 579).
- **Features:** 28 PCA-anonymized numerical features (`V1`–`V28`), plus `Time` and `Amount`. Label column `Class` with 1 = fraud, 0 = benign.

The positive rate of 0.172% sits in the window where whole-curve FPR calibration matters most: too rare for probability calibration to work reliably, common enough that a production system sees fraud on a daily basis.

## Pipeline

The score arrays in `credit_card_roc.npz` were produced by `generate_credit_card_roc.py`:

1. Load the dataset via `sklearn.datasets.fetch_openml(name="creditcard", version=1)`. No authentication required; the loader caches to `~/scikit_learn_data` on first use.
2. Use a stratified 30% split (85,442 rows) to train the detectors and score the remaining 70% holdout (199,365 rows).
3. Select a stratified 30% slice of the detector holdout as the calibration-fit subset; the complementary 70% remains held out from calibration fitting.
4. Train `HistGradientBoostingClassifier(max_iter=500, learning_rate=0.10, max_depth=6, random_state=42)` and a standardized `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` pipeline.
5. Save both detectors' full-holdout scores, the holdout labels, and the calibration-fit mask with `np.savez_compressed`.

The committed NPZ contains these split counts:

| Split | Rows | Benign | Fraud |
|---|---|---|---|
| Detector train | 85,442 | 85,294 | 148 |
| Calibration fit | 59,809 | 59,706 | 103 |
| Held out from calibration fit | 139,556 | 139,315 | 241 |

`calibration_demo.py` fits the calibration pipeline on the 59,706 benign calibration-fit scores. Table 3 and every primary blue figure curve use the disjoint held-out-from-fit subset; the first two figure panels show the calibration-fit subset separately for comparison. The smallest nonzero empirical FPR is `1 / 139,315 = 7.18 × 10⁻⁶` on the evaluation subset, while the smallest rank supported by the calibration-fit sample is `1 / 59,706 = 1.67 × 10⁻⁵`.

## Regenerating from scratch

```bash
python examples/generate_credit_card_roc.py
python examples/calibration_demo.py
```

The first command downloads the dataset on first run, trains both detectors, and writes `credit_card_roc.npz`. The second reads the NPZ, uses the logistic-regression scores for their finer tail resolution, writes `credit_card_validation.png` and `credit_card_eval_table.tex`, and copies both artifacts into `docs/paper/figures/`.

The generators fix their random seeds so repeated runs in the same software environment reproduce the splits and model fits.

## Attribution

If you redistribute the derived NPZ outside this repository, credit the original authors per DbCL v1.0:

> Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium on Computational Intelligence and Data Mining*, 159-166.
