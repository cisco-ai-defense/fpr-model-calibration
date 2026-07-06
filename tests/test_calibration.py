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

"""Unit tests for fpr_model_calibration.calibration module."""

import math

import numpy as np
import pytest

from fpr_model_calibration.calibration import (
    SCORE_MAX,
    _plotting_positions,
    _sample_fpr_values,
    fit_calibration_pipeline,
    fpr_to_calibrated,
)


def _beta_cdf_integer_params(x: float, alpha: int, beta: int) -> float:
    """Beta CDF for positive integer parameters via a binomial tail."""
    n = alpha + beta - 1
    return sum(math.comb(n, j) * (x**j) * ((1.0 - x) ** (n - j)) for j in range(alpha, n + 1))


def _beta_median_integer_params(alpha: int, beta: int) -> float:
    """Numerically invert the Beta CDF for small integer-parameter tests."""
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _beta_cdf_integer_params(mid, alpha, beta) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


class TestSampleFprValues:
    """Tests for _sample_fpr_values function."""

    def test_includes_explicit_knots(self):
        """All explicit FPR knot values must be included."""
        explicit_knots = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        sampled = _sample_fpr_values(n_total=1000)

        for knot in explicit_knots:
            assert np.any(np.isclose(sampled, knot, rtol=1e-9)), f"Missing knot: {knot}"

    def test_returns_correct_count(self):
        """Should return approximately n_total samples."""
        for n in [100, 500, 1000]:
            sampled = _sample_fpr_values(n_total=n)
            assert len(sampled) >= n * 0.9, f"Too few samples: {len(sampled)} < {n * 0.9}"
            assert len(sampled) <= n * 1.1, f"Too many samples: {len(sampled)} > {n * 1.1}"

    def test_sorted_increasing(self):
        """FPR values must be sorted in increasing order."""
        sampled = _sample_fpr_values(n_total=1000)
        assert np.all(np.diff(sampled) > 0), "Not strictly increasing"

    def test_log_spaced_between_knots(self):
        """Points between knots should be approximately log-spaced."""
        sampled = _sample_fpr_values(n_total=1000)

        mask = (sampled > 1e-3) & (sampled < 1e-2)
        between = sampled[mask]

        if len(between) > 2:
            log_vals = np.log10(between)
            diffs = np.diff(log_vals)
            mean_diff = np.mean(diffs)
            assert np.all(diffs > mean_diff * 0.5), "Not log-spaced"
            assert np.all(diffs < mean_diff * 1.5), "Not log-spaced"

    def test_bounds(self):
        """FPR values must be in (0, 1] range."""
        sampled = _sample_fpr_values(n_total=1000)
        assert np.all(sampled > 0), "Contains non-positive values"
        assert np.all(sampled <= 1.0), "Contains values > 1"

    def test_no_duplicates(self):
        """Should not contain duplicate values."""
        sampled = _sample_fpr_values(n_total=1000)
        assert len(sampled) == len(np.unique(sampled)), "Contains duplicates"


class TestPlottingPositions:
    """Tests for order-statistic FPR plotting positions."""

    def test_rejects_unknown_method(self):
        """Only documented plotting-position methods are accepted."""
        with pytest.raises(ValueError, match="plotting_position"):
            _plotting_positions(10, "median")  # ty: ignore[invalid-argument-type]

    def test_rejects_single_sample(self):
        """Two top-tail endpoints are required for fitting and extrapolation."""
        with pytest.raises(ValueError, match="At least two benign scores"):
            _plotting_positions(1, "filliben")

    def test_mean_positions_use_expected_order_statistic_location(self):
        """Mean-centered positions should be k/(n+1), ordered by sorted score."""
        n = 5
        actual = _plotting_positions(n, "mean")
        expected = np.array([5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
        np.testing.assert_allclose(actual, expected)

    def test_filliben_positions_use_endpoint_formulas(self):
        """Filliben endpoints use exact uniform order-statistic medians."""
        n = 10
        actual = _plotting_positions(n, "filliben")
        endpoint = 0.5 ** (1.0 / n)
        assert actual[0] == pytest.approx(endpoint)
        assert actual[-1] == pytest.approx(1.0 - endpoint)

    def test_filliben_positions_use_interior_formula(self):
        """Filliben interior positions should use the published approximation."""
        n = 10
        actual = _plotting_positions(n, "filliben")
        rank_from_top = np.array([9.0, 5.0, 2.0])
        expected = (rank_from_top - 0.3175) / (n + 0.365)
        np.testing.assert_allclose(actual[[1, 5, 8]], expected)

    def test_filliben_is_closer_than_mean_to_exact_beta_medians(self):
        """Filliben should better approximate median-centered FPR coordinates."""
        n = 50
        filliben = _plotting_positions(n, "filliben")
        mean = _plotting_positions(n, "mean")

        filliben_error = 0.0
        mean_error = 0.0
        for idx in range(n):
            rank_from_top = n - idx
            exact = _beta_median_integer_params(rank_from_top, n - rank_from_top + 1)
            filliben_error += abs(filliben[idx] - exact)
            mean_error += abs(mean[idx] - exact)

        assert filliben_error < mean_error * 0.1


class TestFprToCalibrated:
    """Tests for fpr_to_calibrated function."""

    def test_knot_values(self):
        """Fixed FPR knots must map to exact calibrated values."""
        knots = [
            (1.0, 0.0),
            (0.1, 0.1),
            (0.01, 0.3),
            (0.001, 0.5),
            (0.0001, 0.7),
            (1e-5, 0.85),
            (1e-6, 0.95),
        ]
        for fpr, expected in knots:
            actual = fpr_to_calibrated(np.array([fpr]))[0]
            assert np.isclose(actual, expected, rtol=1e-6), (
                f"FPR={fpr}: expected {expected}, got {actual}"
            )

    def test_monotonic(self):
        """Lower FPR should give higher calibrated score."""
        fprs = np.logspace(-6, 0, 100)
        calibrated = fpr_to_calibrated(fprs)
        assert np.all(np.diff(calibrated) <= 0), "Not monotonically decreasing"

    def test_bounds(self):
        """Calibrated values must be in [0, 0.99]."""
        fprs = np.logspace(-10, 0, 100)
        calibrated = fpr_to_calibrated(fprs)
        assert np.all(calibrated >= 0), "Contains negative values"
        assert np.all(calibrated <= 0.99), "Contains values > 0.99"


class TestFitCalibrationPipeline:
    """Tests for fit_calibration_pipeline function."""

    def test_pipeline_output_range(self, benign_scores):
        """Valid pipeline inputs must produce outputs in [0, SCORE_MAX]."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        test_scores = np.linspace(0, 1, 100)
        calibrated = pipeline.predict(test_scores.reshape(-1, 1))
        assert np.all(calibrated >= 0), "Output contains negative values"
        assert np.all(calibrated <= SCORE_MAX), "Output contains values above SCORE_MAX"

    def test_boundary_score_zero(self, benign_scores):
        """Score=0 should map to calibrated near 0."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        calibrated = pipeline.predict([[0.0]])[0]
        assert calibrated < 0.1, f"Score=0 should give low calibrated, got {calibrated}"

    def test_boundary_score_one(self, benign_scores):
        """Raw score 1 must remain below the flag-nothing threshold."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        calibrated = pipeline.predict([[1.0]])[0]
        assert calibrated > 0.9, f"Score=1 should give high calibrated, got {calibrated}"
        assert calibrated <= SCORE_MAX, f"Score=1 exceeds SCORE_MAX: {calibrated}"

    def test_monotonic_output(self, benign_scores):
        """Higher scores should give higher calibrated values."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        test_scores = np.linspace(0, 1, 100)
        calibrated = pipeline.predict(test_scores.reshape(-1, 1))
        assert np.all(np.diff(calibrated) >= -1e-6), "Output not monotonically increasing"

    def test_rejects_invalid_scores(self):
        """Should reject scores outside [0, 1]."""
        with pytest.raises(ValueError):
            fit_calibration_pipeline(np.array([-0.1, 0.5, 0.9]))
        with pytest.raises(ValueError):
            fit_calibration_pipeline(np.array([0.1, 0.5, 1.1]))

    def test_no_debug_attributes_by_default(self, benign_scores):
        """Default pipeline must not carry debug attrs that inflate serialized size."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        assert not hasattr(pipeline, "fpr_to_score_")
        assert not hasattr(pipeline, "sampled_fprs_")
        assert not hasattr(pipeline, "sampled_scores_")

    def test_debug_attributes_when_requested(self, benign_scores):
        """keep_debug=True attaches debug artifacts for plotting."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100, keep_debug=True)
        assert hasattr(pipeline, "fpr_to_score_")
        assert hasattr(pipeline, "sampled_fprs_")
        assert hasattr(pipeline, "sampled_scores_")

    def test_retains_first_spline_edge_fprs(self, benign_scores):
        """The second fit must retain the first spline's fitted edge locations."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100, keep_debug=True)
        fitted_fprs = _plotting_positions(len(benign_scores), "filliben")
        expected_edges = fitted_fprs[[0, -2, -1]]
        sampled_fprs = pipeline.sampled_fprs_  # ty: ignore[unresolved-attribute]

        assert np.all(np.isin(expected_edges, sampled_fprs)), (
            f"Missing fitted edge FPRs: {expected_edges[~np.isin(expected_edges, sampled_fprs)]}"
        )
        assert np.all(np.diff(sampled_fprs) > 0), "Augmented FPR grid is not strictly increasing"

    def test_default_plotting_position_is_filliben(self, benign_scores):
        """Default fit should use median-centered plotting positions."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        assert pipeline.plotting_position_ == "filliben"  # ty: ignore[unresolved-attribute]

    def test_accepts_mean_plotting_position(self, benign_scores):
        """Mean-centered plotting positions are available as an explicit option."""
        pipeline = fit_calibration_pipeline(
            benign_scores,
            n_knots=100,
            plotting_position="mean",
        )
        assert pipeline.plotting_position_ == "mean"  # ty: ignore[unresolved-attribute]

    def test_rejects_unknown_plotting_position(self, benign_scores):
        """Unknown plotting-position choices should fail before fitting."""
        with pytest.raises(ValueError, match="plotting_position"):
            fit_calibration_pipeline(
                benign_scores,
                n_knots=100,
                plotting_position="naive",  # ty: ignore[invalid-argument-type]
            )

    def test_filliben_first_spline_is_more_accurate_on_median_sample(self):
        """Filliben-labeled knots recover known uniform median positions."""
        n = 200
        true_fprs = _plotting_positions(n, "filliben")
        benign_scores = 1.0 - true_fprs

        filliben_pipeline = fit_calibration_pipeline(
            benign_scores,
            n_knots=1000,
            keep_debug=True,
            plotting_position="filliben",
        )
        mean_pipeline = fit_calibration_pipeline(
            benign_scores,
            n_knots=1000,
            keep_debug=True,
            plotting_position="mean",
        )

        idx = np.arange(10, n - 10, 10)
        target_fprs = true_fprs[idx]
        expected_scores = SCORE_MAX * (1.0 - target_fprs)

        filliben_pred = filliben_pipeline.fpr_to_score_.predict(target_fprs)  # ty: ignore[unresolved-attribute]
        mean_pred = mean_pipeline.fpr_to_score_.predict(target_fprs)  # ty: ignore[unresolved-attribute]

        filliben_error = np.mean(np.abs(filliben_pred - expected_scores))
        mean_error = np.mean(np.abs(mean_pred - expected_scores))

        assert filliben_error < 1e-12
        assert mean_error > filliben_error + 1e-4

    def test_mean_first_spline_is_more_accurate_on_mean_sample(self):
        """Mean-labeled knots recover known uniform mean positions."""
        n = 200
        true_fprs = _plotting_positions(n, "mean")
        benign_scores = 1.0 - true_fprs

        mean_pipeline = fit_calibration_pipeline(
            benign_scores,
            n_knots=1000,
            keep_debug=True,
            plotting_position="mean",
        )
        filliben_pipeline = fit_calibration_pipeline(
            benign_scores,
            n_knots=1000,
            keep_debug=True,
            plotting_position="filliben",
        )

        idx = np.arange(10, n - 10, 10)
        target_fprs = true_fprs[idx]
        expected_scores = SCORE_MAX * (1.0 - target_fprs)

        mean_pred = mean_pipeline.fpr_to_score_.predict(target_fprs)  # ty: ignore[unresolved-attribute]
        filliben_pred = filliben_pipeline.fpr_to_score_.predict(target_fprs)  # ty: ignore[unresolved-attribute]

        mean_error = np.mean(np.abs(mean_pred - expected_scores))
        filliben_error = np.mean(np.abs(filliben_pred - expected_scores))

        assert mean_error < 1e-12
        assert filliben_error > mean_error + 1e-4

    def test_deep_tail_honors_contract(self):
        """Pipeline.predict must honor the log anchor contract down to 1e-9.

        With a uniform calibration sample, threshold = 1 - FPR (exact), so
        feeding raw = 1 - target_FPR should produce the contract's calibrated
        value at that FPR. Tolerance is generous because the deep-tail region
        has irreducible sampling variance.
        """
        rng = np.random.RandomState(0)
        scores = rng.uniform(0, 1, size=1_000_000)
        pipeline = fit_calibration_pipeline(scores, n_knots=10000)

        targets = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        raws = np.array([1 - t for t in targets])
        expected = fpr_to_calibrated(np.array(targets))
        actual = pipeline.predict(raws.reshape(-1, 1))

        for t, a, e in zip(targets, actual, expected, strict=True):
            # 0.06 absolute tolerance: one anchor-step of the contract (the
            # contract itself changes by 0.05-0.20 per decade of FPR).
            assert abs(a - e) < 0.06, f"FPR={t:.0e}: contract={e:.3f}, pipeline={a:.3f}"

    def test_above_calib_max_stays_in_range(self):
        """Scores above calib.max must produce calibrated < 1.0 and >= the
        calibrated value at calib.max. No silent collapse to the flag-nothing
        terminal just because the calibration sample did not reach raw=1.0."""
        rng = np.random.RandomState(0)
        # Calibration sample bounded above well below 1.0.
        scores = rng.uniform(0, 0.5, size=10_000)
        pipeline = fit_calibration_pipeline(scores, n_knots=1000)

        probe = np.linspace(0.0, 1.0, 1001).reshape(-1, 1)
        out = pipeline.predict(probe)

        assert out[-1] == pytest.approx(0.99, abs=1e-6), (
            f"raw=1.0 must map to calibrated=0.99 (SCORE_MAX), got {out[-1]}"
        )
        # No interior point should reach or exceed 1.0.
        assert np.all(out[:-1] < 1.0), "Interior probe produced calibrated >= 1.0"
        # Strictly monotone non-decreasing.
        assert np.all(np.diff(out) >= 0), "Output not monotone non-decreasing"
