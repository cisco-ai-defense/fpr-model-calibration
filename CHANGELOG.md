# Changelog

All notable changes to this project are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-24

### Added

- Support for Python 3.14, declared in the package classifiers and tested in CI and in the post-publish verification.

### Changed

- `fit_calibration_pipeline` is about 2.2 to 2.6 times faster. It interpolates the first-pass FPR-to-score curve directly instead of refitting it with `IsotonicRegression`, and builds the knot grid without a Python loop. Fitted pipelines match those from 0.1.0 to within 1e-12.

### Fixed

- `fit_calibration_pipeline` now rejects an empty `benign_scores` array with its own `ValueError` ("At least two benign scores are required for calibration") instead of an error raised inside scikit-learn's `MinMaxScaler`.
- `fit_calibration_pipeline` rejects NaN benign scores with its own `ValueError` instead of a scikit-learn error.
- `fit_calibration_pipeline` rejects an `n_knots` that is not an integer of at least 11. Smaller values used to produce a calibrator with only the decade knots, and non-integers such as `1e4` failed inside NumPy.
- Pipelines clip raw scores outside `[0, 1]` before calibration, so outputs stay within `[0, 0.99]`. A raw score of 1.5 used to calibrate to 1.0. Pipelines fitted with earlier versions keep the previous behavior.

## [0.1.0] - 2026-07-08

### Added

- `fit_calibration_pipeline(benign_scores, n_knots=10000)` fits a whole-curve FPR calibrator on benign scores and returns an sklearn `Pipeline` of `MinMaxScaler` and `IsotonicRegression`.
- `fpr_to_calibrated(fpr)` applies the fixed log-scale FPR→[0,1] map used by the calibrator (`0.5 = 0.1% FPR`, `0.7 = 0.01%`, `0.85 = 0.001%`).
- Two-step knot-subsampling procedure bounds artifact size at about 10K knots (~80 KB) regardless of training-set size.
- Unit tests covering FPR-sample generation, calibration fitting, round-trip calibration, and edge cases (all-equal scores, single-score input, out-of-range values).
- Validation example that fits the pipeline on a held-out benign sample and reports calibration error vs target FPR across log-spaced buckets.
- Build validation and tokenless PyPI publishing through GitHub trusted publishing.
- Distribution and import package named FPRCal (`fprcal`).

[unreleased]: https://github.com/cisco-ai-defense/fpr-model-calibration/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cisco-ai-defense/fpr-model-calibration/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cisco-ai-defense/fpr-model-calibration/releases/tag/v0.1.0
