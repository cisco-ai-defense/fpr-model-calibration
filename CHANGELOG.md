# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Initial release

### Added

- `fit_calibration_pipeline(benign_scores, n_knots=10000)` fits a whole-curve FPR calibrator on benign scores and returns an sklearn `Pipeline` of `MinMaxScaler` and `IsotonicRegression`.
- `fpr_to_calibrated(fpr)` applies the fixed log-scale FPR→[0,1] map used by the calibrator (`0.5 = 0.1% FPR`, `0.7 = 0.01%`, `0.85 = 0.001%`).
- Two-step knot-subsampling procedure bounds artifact size at about 10K knots (~80 KB) regardless of training-set size.
- Unit tests covering FPR-sample generation, calibration fitting, round-trip calibration, and edge cases (all-equal scores, single-score input, out-of-range values).
- Validation example that fits the pipeline on a held-out benign sample and reports calibration error vs target FPR across log-spaced buckets.
