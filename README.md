# FPRCal

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/fprcal.svg)](https://pypi.org/project/fprcal/)

Stable low-FPR calibration for detection model scores across model releases, with a fixed log-scale interpretability contract (`0.5 = 0.1% FPR`, `0.7 = 0.01%`, `0.85 = 0.001%`).

## Why

Raw scores from ML detectors drift across model releases and are not comparable across detector categories. Downstream product rules break on every retrain. This package calibrates detector scores to FPR on benign traffic, then applies a fixed log-scale transform so every calibrated value has the same FPR meaning across model versions and detector categories.

## Installation

FPRCal supports Python 3.12 and 3.13. Install the latest release from
[PyPI](https://pypi.org/project/fprcal/):

```bash
python -m pip install fprcal
```

Pin a release when reproducibility matters:

```bash
python -m pip install "fprcal==0.1.0"
```

Upgrade an existing installation with:

```bash
python -m pip install --upgrade fprcal
```

Verify the installed version:

```bash
python -c "import fprcal; print(fprcal.__version__)"
```

Projects that use [uv](https://docs.astral.sh/uv/) can add FPRCal as a
dependency with `uv add fprcal`. Pip and uv install numpy, scikit-learn, and
joblib as package dependencies.

### Development installation

Install the repository and its development dependencies in editable mode:

```bash
git clone https://github.com/cisco-ai-defense/fpr-model-calibration.git
cd fpr-model-calibration
python -m pip install -e ".[dev]"
```

## Quick start

```python
import joblib
import numpy as np

from fprcal import fit_calibration_pipeline

benign_scores = np.array([0.01, 0.04, 0.08, 0.15, 0.25, 0.40])
pipeline = fit_calibration_pipeline(benign_scores, n_knots=1000)

raw_scores = np.array([0.05, 0.20, 0.80])
calibrated_scores = pipeline.predict(raw_scores.reshape(-1, 1))
print(calibrated_scores)

joblib.dump(pipeline, "calibration.pkl")
```

Load the serialized scikit-learn pipeline with `joblib.load` and call
`predict` on an `(n_samples, 1)` array of raw detector scores.

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

Maintainers can find the release procedure in [RELEASING.md](RELEASING.md).
