# FPRCal

[![CI](https://github.com/cisco-ai-defense/fpr-model-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/cisco-ai-defense/fpr-model-calibration/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fprcal.svg)](https://pypi.org/project/fprcal/)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.05481-b31b1b.svg)](https://arxiv.org/abs/2607.05481)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/LICENSE)

**FPRCal calibrates detector scores to a fixed false-positive-rate scale, so
thresholds retain the same operational meaning after retraining.**

Fit it on representative benign scores and it returns a standard scikit-learn
pipeline. Each detector release gets its own calibration artifact, while
downstream systems keep the same calibrated thresholds.

[Paper](https://arxiv.org/abs/2607.05481) ·
[PyPI](https://pypi.org/project/fprcal/) ·
[Example](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/examples/calibration_demo.py) ·
[Contributing](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/CONTRIBUTING.md)

![FPRCal fits a two-spline calibration map offline and ships only the raw-score-to-calibrated-score pipeline for inference.](https://raw.githubusercontent.com/cisco-ai-defense/fpr-model-calibration/main/docs/paper/figures/method_overview.png)

## The score contract

FPRCal maps a raw detector score to a calibrated threshold on a fixed log-FPR
scale. The same calibrated threshold therefore carries the same target benign
FPR for each independently calibrated model release.

| Calibrated threshold | Target benign FPR | Operational meaning |
| ---: | ---: | --- |
| `0.10` | 10% | 1 in 10 benign samples flagged |
| `0.30` | 1% | 1 in 100 |
| `0.50` | 0.1% | 1 in 1,000 |
| `0.70` | 0.01% | 1 in 10,000 |
| `0.85` | 0.001% | 1 in 100,000 |
| `0.95` | 0.0001% | 1 in 1,000,000 |

The pipeline emits values no greater than `0.99`, so a threshold of `1.0`
flags nothing.

## Installation

FPRCal supports Python 3.12 and 3.13.

```bash
python -m pip install fprcal
```

Pin a release when reproducibility matters:

```bash
python -m pip install "fprcal==0.1.0"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add fprcal
```

## Quick start

`fit_calibration_pipeline` accepts a one-dimensional array of benign detector
scores in `[0, 1]`, where higher values indicate stronger detection confidence.
The returned object is a fitted scikit-learn `Pipeline`.

```python
import joblib
import numpy as np

from fprcal import fit_calibration_pipeline

# Replace this synthetic sample with scores from representative benign traffic.
rng = np.random.default_rng(7)
benign_scores = rng.beta(a=2.0, b=20.0, size=100_000)

calibrator = fit_calibration_pipeline(benign_scores)

raw_scores = np.array([0.05, 0.20, 0.80])
calibrated_scores = calibrator.predict(raw_scores.reshape(-1, 1))

joblib.dump(calibrator, "calibration.joblib")
restored = joblib.load("calibration.joblib")
assert np.allclose(
    calibrated_scores,
    restored.predict(raw_scores.reshape(-1, 1)),
)
```

Fit one calibration artifact for each detector release using that release's
benign scores. Ship the serialized pipeline with the detector and call
`predict` on an `(n_samples, 1)` array at inference time.

## How it works

1. FPRCal sorts the benign scores and assigns median-centered Filliben plotting
   positions to estimate the fresh-sample tail probability at each threshold.
2. A temporary monotone spline maps target FPR values to raw-score thresholds
   on a fixed log-spaced grid.
3. A second monotone spline maps raw scores to the fixed calibrated scale. Only
   this bounded, standard scikit-learn pipeline ships to production.

Set `plotting_position="mean"` to use mean-centered `k/(n+1)` positions instead
of the default Filliben positions. Set `keep_debug=True` only for diagnostics;
the retained fit-time arrays increase serialized artifact size with the number
of benign samples.

## Low-FPR reliability

Benign sample count limits the FPR range that data can support. For target FPR
`p` and relative 95% half-width `r`, the normal-approximation planning rule is
`n ≈ 4 / (r²p)`. Estimating 0.1% FPR with a 25% relative half-width therefore
requires about 64,000 representative benign samples. Values below the smallest
observed tail FPR are extrapolations, not additional statistical evidence.

The [paper](https://arxiv.org/abs/2607.05481) derives the planning rule and
discusses finite-sample limits, plotting-position bias, and benign-distribution
drift.

## When to refit

Fit a new calibration artifact for every detector release and whenever the
benign score distribution changes. The fixed scale standardizes target FPR
meaning across artifacts; it cannot correct a calibration set that no longer
represents production traffic. Evaluate detector quality separately by comparing
true-positive rate at the same target FPR.

## Evaluation

On one held-out Credit Card Fraud Detection split, the paper reports relative
FPR error no greater than 2.3% from 10% through 0.1% FPR and 7.2% at 0.01% FPR.
Across calibration sets from 1,000 to 10 million benign samples, the serialized
production artifact remained below 200 KB. See the
[full evaluation](https://arxiv.org/abs/2607.05481) and the
[reproduction guide](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/examples/credit_card_readme.md) for the dataset split,
confidence interval, detector choice, and commands.

## Paper

Read [*Full-range Binary Classifier Calibration for Stable Model Updates in
Production*](https://arxiv.org/abs/2607.05481) for the method, statistical
analysis, evaluation, and limitations. The checked-in
[paper directory](https://github.com/cisco-ai-defense/fpr-model-calibration/tree/main/docs/paper)
contains the source and generated figures.

## Development

```bash
git clone https://github.com/cisco-ai-defense/fpr-model-calibration.git
cd fpr-model-calibration
uv sync --all-extras --locked
uv run pytest --cov=fprcal --cov-report=term-missing
```

See [CONTRIBUTING.md](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/CONTRIBUTING.md)
for the complete validation suite,
[SECURITY.md](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/SECURITY.md)
for private vulnerability reporting, and
[RELEASING.md](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/RELEASING.md)
for the maintainer release procedure.

## License

FPRCal is available under the
[Apache License 2.0](https://github.com/cisco-ai-defense/fpr-model-calibration/blob/main/LICENSE).
