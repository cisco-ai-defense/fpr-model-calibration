# fpr-model-calibration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Whole-curve False Positive Rate calibration for ML detector scores, with a fixed log-scale interpretability contract (`0.5 = 0.1% FPR`, `0.7 = 0.01%`, `0.85 = 0.001%`).

## Why

Raw scores from ML detectors drift across model releases and are not comparable across detector categories. Downstream product rules break on every retrain. This package calibrates detector scores to FPR on benign traffic, then applies a fixed log-scale transform so every calibrated value has the same FPR meaning across model versions and detector categories.

## Install

```bash
pip install -e .
```

Requires Python 3.12+, numpy, scikit-learn, joblib.

## Usage

```python
from fpr_model_calibration import fit_calibration_pipeline
import joblib

pipeline = fit_calibration_pipeline(benign_scores, n_knots=10000)
joblib.dump(pipeline, "calibration.pkl")

# In production:
pipeline = joblib.load("calibration.pkl")
calibrated = pipeline.predict(raw_scores.reshape(-1, 1))
```

The first-pass FPR-to-threshold spline uses Filliben median-centered plotting
positions by default. Pass `plotting_position="mean"` to use the mean-centered
`k/(n+1)` positions instead.

## Demo

Reproduce the evaluation figure on the Credit Card Fraud Detection dataset (OpenML, 284K rows, 0.172% positives):

```bash
# One-time: download the data, train both detectors, save holdout scores
python examples/generate_credit_card_roc.py

# Fit on the calibration subset and evaluate its held-out complement
python examples/calibration_demo.py
```

Writes `examples/credit_card_validation.png`. See [examples/credit_card_readme.md](examples/credit_card_readme.md) for dataset provenance.

## License

Apache License 2.0. See [LICENSE](LICENSE).
