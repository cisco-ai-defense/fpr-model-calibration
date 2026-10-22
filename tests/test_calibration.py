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

"""Unit tests for fprcal.calibration module."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import betaincinv

from fprcal.calibration import (
    SCORE_MAX,
    _plotting_positions,
    _sample_fpr_values,
    fit_calibration_pipeline,
    fpr_to_calibrated,
)

# FPR decades named by the calibration contract, from 1e-10 to 1.0.
CONTRACT_DECADES = 10.0 ** np.arange(-10, 1)
HELD_OUT_SIZE = 200_000


@pytest.fixture(scope="module")
def held_out_calibrated() -> NDArray[np.float64]:
    """Calibrated scores for unseen benign data drawn from the fitted distribution."""
    rng = np.random.default_rng(0)
    pipeline = fit_calibration_pipeline(rng.beta(2, 8, HELD_OUT_SIZE))
    return pipeline.predict(rng.beta(2, 8, HELD_OUT_SIZE).reshape(-1, 1))


class TestSampleFprValues:
    """Tests for the log-spaced FPR grid behind the stored knots."""

    @pytest.mark.parametrize("n_total", [100, 1000, 10_000])
    def test_grid_covers_every_decade_with_uniform_log_spacing(self, n_total: int) -> None:
        """The grid spans 1e-10 to 1.0, keeps each decade, and is evenly log-spaced."""
        sampled = _sample_fpr_values(n_total)
        log_steps = np.diff(np.log10(sampled))

        assert sampled[0] == 1e-10
        assert sampled[-1] == 1.0
        assert np.isclose(sampled[:, None], CONTRACT_DECADES, rtol=1e-12, atol=0).any(axis=0).all()
        assert np.all(log_steps > 0), "Grid is not strictly increasing"
        np.testing.assert_allclose(log_steps, log_steps[0], rtol=1e-6)
        assert 0.9 * n_total <= len(sampled) <= n_total


class TestPlottingPositions:
    """Tests for the FPR assigned to each sorted benign score."""

    def test_mean_positions_are_rank_over_n_plus_one(self) -> None:
        """Mean-centered positions are k/(n+1), ordered by ascending score."""
        expected = np.array([5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
        np.testing.assert_allclose(_plotting_positions(5, "mean"), expected)

    def test_filliben_positions_follow_published_formula(self) -> None:
        """Filliben uses exact endpoint medians and (r - 0.3175)/(n + 0.365) inside."""
        n = 10
        expected = (n - np.arange(n) - 0.3175) / (n + 0.365)
        expected[[0, -1]] = [0.5 ** (1.0 / n), 1.0 - 0.5 ** (1.0 / n)]
        np.testing.assert_allclose(_plotting_positions(n, "filliben"), expected)

    def test_filliben_tracks_exact_order_statistic_medians(self) -> None:
        """Filliben is exact at the endpoints and far closer than k/(n+1) inside."""
        n = 50
        rank_from_top = n - np.arange(n)
        exact = betaincinv(rank_from_top, n - rank_from_top + 1, 0.5)
        filliben = _plotting_positions(n, "filliben")
        mean = _plotting_positions(n, "mean")

        np.testing.assert_allclose(filliben[[0, -1]], exact[[0, -1]], rtol=1e-10)
        assert np.abs(filliben - exact).sum() < 0.1 * np.abs(mean - exact).sum()


class TestFprToCalibrated:
    """Tests for the fixed FPR-to-calibrated contract."""

    @pytest.mark.parametrize(
        ("fpr", "expected"),
        [
            pytest.param(1.0, 0.0, id="anchor-1"),
            pytest.param(1e-1, 0.1, id="anchor-1e-1"),
            pytest.param(1e-2, 0.3, id="anchor-1e-2"),
            pytest.param(1e-3, 0.5, id="anchor-1e-3"),
            pytest.param(1e-4, 0.7, id="anchor-1e-4"),
            pytest.param(1e-5, 0.85, id="anchor-1e-5"),
            pytest.param(1e-6, 0.95, id="anchor-1e-6"),
            pytest.param(1e-10, 0.99, id="anchor-1e-10"),
            pytest.param(10**-0.5, 0.05, id="between-1-and-1e-1"),
            pytest.param(10**-2.5, 0.4, id="between-1e-2-and-1e-3"),
            pytest.param(10**-5.5, 0.9, id="between-1e-5-and-1e-6"),
            pytest.param(1e-8, 0.97, id="between-1e-6-and-1e-10"),
            pytest.param(0.0, 0.99, id="clip-zero"),
            pytest.param(1e-12, 0.99, id="clip-below-1e-10"),
            pytest.param(1.5, 0.0, id="clip-above-1"),
        ],
    )
    def test_maps_fpr_to_piecewise_log_linear_contract(self, fpr: float, expected: float) -> None:
        """Anchors map exactly, values between interpolate in log10(FPR), others clip."""
        assert float(fpr_to_calibrated(fpr)) == pytest.approx(expected, abs=1e-12)

    def test_is_monotone_decreasing_and_bounded(self) -> None:
        """Lower FPR never lowers the calibrated score, which stays within [0, 0.99]."""
        calibrated = fpr_to_calibrated(np.logspace(-12, 0.5, 200))
        assert np.all(np.diff(calibrated) <= 0), "Not monotonically decreasing"
        assert calibrated.min() == 0.0
        assert calibrated.max() == 0.99


class TestFitCalibrationPipeline:
    """Tests for fit_calibration_pipeline function."""

    def test_maps_unit_interval_monotonically_into_calibrated_range(self, benign_scores):
        """Raw scores in [0, 1] map monotonically into [0, SCORE_MAX], low at 0, high at 1."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100)
        calibrated = pipeline.predict(np.linspace(0, 1, 101).reshape(-1, 1))

        assert np.all(np.diff(calibrated) >= 0), "Output not monotonically non-decreasing"
        assert calibrated.min() >= 0
        assert calibrated.max() <= SCORE_MAX
        assert calibrated[0] < 0.1, f"Score 0 should calibrate low, got {calibrated[0]}"
        assert calibrated[-1] > 0.9, f"Score 1 should calibrate high, got {calibrated[-1]}"

    @pytest.mark.parametrize("target_fpr", [1e-1, 1e-2, 1e-3])
    def test_calibrated_threshold_has_target_fpr_on_held_out_benign(
        self, held_out_calibrated: NDArray[np.float64], target_fpr: float
    ) -> None:
        """Thresholding at fpr_to_calibrated(target) flags about target of unseen benign scores.

        The tolerance is four binomial standard errors of the observed rate, counting
        sampling noise from both the fitting sample and the held-out sample.
        """
        observed = float(np.mean(held_out_calibrated >= fpr_to_calibrated(target_fpr)))
        rel_tol = 4 * np.sqrt(2 * (1 - target_fpr) / (target_fpr * HELD_OUT_SIZE))
        assert observed == pytest.approx(target_fpr, rel=rel_tol)

    @pytest.mark.parametrize("scores", [[-0.1, 0.5, 0.9], [0.1, 0.5, 1.1]])
    def test_rejects_scores_outside_unit_interval(self, scores: list[float]) -> None:
        """Benign scores must lie in [0, 1]."""
        with pytest.raises(ValueError, match=r"Scores must be in \[0, 1\]"):
            fit_calibration_pipeline(np.array(scores))

    def test_rejects_a_single_score(self) -> None:
        """Fitting and tail extrapolation need at least two benign scores."""
        with pytest.raises(ValueError, match="At least two benign scores"):
            fit_calibration_pipeline(np.array([0.5]))

    def test_rejects_unknown_plotting_position(self, benign_scores):
        """Unknown plotting-position choices should fail before fitting."""
        with pytest.raises(ValueError, match="plotting_position"):
            fit_calibration_pipeline(
                benign_scores,
                n_knots=100,
                plotting_position="naive",  # ty: ignore[invalid-argument-type]
            )

    def test_records_plotting_position_with_filliben_default(self, benign_scores):
        """The pipeline records its plotting position; Filliben is the default."""
        default = fit_calibration_pipeline(benign_scores, n_knots=100)
        mean = fit_calibration_pipeline(benign_scores, n_knots=100, plotting_position="mean")
        assert default.plotting_position_ == "filliben"  # ty: ignore[unresolved-attribute]
        assert mean.plotting_position_ == "mean"  # ty: ignore[unresolved-attribute]

    @pytest.mark.parametrize("keep_debug", [False, True])
    def test_attaches_debug_artifacts_only_on_request(self, benign_scores, keep_debug: bool):
        """Debug attributes inflate serialized size, so only keep_debug=True attaches them."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100, keep_debug=keep_debug)
        for name in ("fpr_to_score_", "sampled_fprs_", "sampled_scores_"):
            assert hasattr(pipeline, name) is keep_debug, name

    def test_knot_grid_keeps_first_spline_edges_and_monotone_scores(self, benign_scores):
        """The knot grid keeps the first spline's edges, and scores fall as FPR rises."""
        pipeline = fit_calibration_pipeline(benign_scores, n_knots=100, keep_debug=True)
        sampled_fprs = pipeline.sampled_fprs_  # ty: ignore[unresolved-attribute]
        sampled_scores = pipeline.sampled_scores_  # ty: ignore[unresolved-attribute]
        edges = _plotting_positions(len(benign_scores), "filliben")[[0, -2, -1]]

        missing = edges[~np.isin(edges, sampled_fprs)]
        assert missing.size == 0, f"Missing fitted edge FPRs: {missing}"
        assert np.all(np.diff(sampled_fprs) > 0), "Augmented FPR grid is not strictly increasing"
        assert sampled_scores.shape == sampled_fprs.shape
        assert np.all(np.diff(sampled_scores) <= 0), "Knot scores rise with FPR"

    @pytest.mark.parametrize(
        ("sample_method", "other_method"), [("filliben", "mean"), ("mean", "filliben")]
    )
    def test_first_spline_recovers_positions_of_matching_sample(
        self, sample_method, other_method
    ) -> None:
        """A sample placed at one method's positions is recovered exactly only by that method."""
        n = 200
        true_fprs = _plotting_positions(n, sample_method)
        target_fprs = true_fprs[10 : n - 10 : 10]
        expected_scores = SCORE_MAX * (1.0 - target_fprs)

        errors = {}
        for method in (sample_method, other_method):
            pipeline = fit_calibration_pipeline(
                1.0 - true_fprs, n_knots=1000, keep_debug=True, plotting_position=method
            )
            predicted = pipeline.fpr_to_score_.predict(target_fprs)  # ty: ignore[unresolved-attribute]
            errors[method] = np.mean(np.abs(predicted - expected_scores))

        assert errors[sample_method] < 1e-12
        assert errors[other_method] > errors[sample_method] + 1e-4

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

    def test_rejects_empty_scores(self):
        """An empty sample fails with this module's minimum-size error."""
        with pytest.raises(ValueError, match="At least two benign scores"):
            fit_calibration_pipeline(np.array([], dtype=np.float64))
