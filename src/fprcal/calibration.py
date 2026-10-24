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

"""FPR-based calibration using sklearn Pipeline.

Two-step fitting, single-step inference:
1. Interpolate the benign empirical CDF: FPR -> score (discarded after fitting).
2. Fit final IsotonicRegression: score -> calibrated (kept in pipeline).

The final pipeline stores about ``n_knots`` (score, calibrated) knots regardless
of training set size.
"""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SCORE_MAX: float = 0.99
PlottingPosition = Literal["filliben", "mean"]

# FPR-to-calibrated mapping knots (piecewise linear in log10(FPR) space).
_FPR_CAL_KNOTS: list[tuple[int, float]] = [
    (0, 0.0),  # 100% FPR    -> cal=0.0
    (-1, 0.1),  # 10% FPR     -> cal=0.1
    (-2, 0.3),  # 1% FPR      -> cal=0.3
    (-3, 0.5),  # 0.1% FPR    -> cal=0.5
    (-4, 0.7),  # 0.01% FPR   -> cal=0.7
    (-5, 0.85),  # 0.001% FPR  -> cal=0.85
    (-6, 0.95),  # 0.0001% FPR -> cal=0.95
    (-10, 0.99),  # ~0% FPR     -> cal=0.99
]

_sorted_idx = np.argsort([k[0] for k in _FPR_CAL_KNOTS])
_LOG_FPR_KNOTS: NDArray[np.float64] = np.array(
    [_FPR_CAL_KNOTS[i][0] for i in _sorted_idx], dtype=np.float64
)
_CAL_KNOTS: NDArray[np.float64] = np.array(
    [_FPR_CAL_KNOTS[i][1] for i in _sorted_idx], dtype=np.float64
)

# FPR decades the knot grid always contains, covering the contract from 1e-10 to 1.0.
_DECADE_FPRS: tuple[float, ...] = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fpr_to_calibrated(fpr: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Map FPR values to calibrated [0, 1] scores via the fixed log-scale anchors.

    Parameters
    ----------
    fpr : array-like or float
        False positive rate in ``[0, 1]``. Values outside this range are clipped
        to ``[1e-10, 1.0]`` before interpolation.

    Returns
    -------
    ndarray
        Calibrated scores in ``[0, 0.99]``. The anchors are: FPR=1 -> 0.0,
        FPR=0.1 -> 0.1, FPR=0.01 -> 0.3, FPR=0.001 -> 0.5, FPR=0.0001 -> 0.7,
        FPR=1e-5 -> 0.85, FPR=1e-6 -> 0.95, FPR=1e-10 -> 0.99.
    """
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    log_fpr = np.log10(np.clip(fpr_arr, 1e-10, 1.0))
    return np.interp(log_fpr, _LOG_FPR_KNOTS, _CAL_KNOTS)


def _sample_fpr_values(n_total: int = 1000) -> NDArray[np.float64]:
    """Sample FPR values including all explicit knots, with log-spaced fill between.

    Returns FPR values (not log) sorted in increasing order. Covers the full
    log-anchor contract range from 1e-10 to 1.0 so the second isotonic has
    knots across every decade the contract defines.
    """
    n_gaps = len(_DECADE_FPRS) - 1
    points_per_gap = (n_total - len(_DECADE_FPRS)) // n_gaps

    # Each decade contributes its lower edge plus evenly log-spaced interior points.
    decades = [
        np.concatenate(([lo], np.logspace(np.log10(lo), np.log10(hi), points_per_gap + 2)[1:-1]))
        for lo, hi in zip(_DECADE_FPRS[:-1], _DECADE_FPRS[1:], strict=True)
    ]
    return np.unique(np.concatenate([*decades, [_DECADE_FPRS[-1]]]))


def _plotting_positions(n: int, method: PlottingPosition) -> NDArray[np.float64]:
    """Return upper-tail FPR plotting positions for sorted scores.

    The returned array aligns with scores sorted in ascending order. Entry ``i``
    is the plotting-position FPR for ``sorted_scores[i]`` under the inclusive
    threshold convention ``score >= threshold``.
    """
    if n < 2:
        raise ValueError("At least two benign scores are required for calibration")

    rank_from_top = n - np.arange(n, dtype=np.float64)

    if method == "mean":
        return rank_from_top / (n + 1)

    if method == "filliben":
        positions = (rank_from_top - 0.3175) / (n + 0.365)
        endpoint = 0.5 ** (1.0 / n)
        positions[0] = endpoint
        positions[-1] = 1.0 - endpoint
        return positions

    raise ValueError("plotting_position must be 'filliben' or 'mean'")


def fit_calibration_pipeline(
    benign_scores: NDArray[np.float64],
    n_knots: int = 10000,
    *,
    keep_debug: bool = False,
    plotting_position: PlottingPosition = "filliben",
) -> Pipeline:
    """Fit a whole-curve FPR calibration pipeline from benign scores.

    The pipeline maps raw detector scores in ``[0, 1]`` to calibrated values in
    ``[0, 0.99]``, where every calibrated value has a stable FPR meaning
    (``0.5 = 0.1% FPR``, ``0.7 = 0.01%``, and so on).

    Two-step fitting:

    1. Interpolate the benign empirical CDF to get ``FPR -> score``. The ordered
       (FPR, score) pairs are already monotone, so this equals an isotonic fit.
    2. Sample ~``n_knots`` FPR values on a log-spaced base grid, add the first
       spline's fitted edge FPRs, and compute ``(score, calibrated)`` pairs.
    3. Fit the final ``IsotonicRegression`` on those pairs as ``score -> calibrated``.

    Parameters
    ----------
    benign_scores : ndarray
        Scores from benign/negative samples, in ``[0, 1]``.
    n_knots : int, optional
        Approximate number of knots to store in the final pipeline, at least 11
        (one per FPR decade from 1e-10 to 1). Default 10000.
    keep_debug : bool, optional
        When True, attach an ``IsotonicRegression`` of the first-pass curve as
        ``fpr_to_score_`` and the sampled FPR/score arrays to the returned
        Pipeline for plotting. These
        attributes serialize under ``joblib.dump`` and inflate artifact size
        with ``len(benign_scores)``, so the default is False for production use.
    plotting_position : {"filliben", "mean"}, optional
        FPR coordinate assigned to each sorted benign threshold in the temporary
        FPR-to-score spline. ``"filliben"`` uses median-centered order-statistic
        plotting positions and is the default. ``"mean"`` uses the mean-centered
        ``k/(n+1)`` positions.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted sklearn Pipeline with ``MinMaxScaler`` and ``IsotonicRegression``
        stages. Call ``pipeline.predict(scores.reshape(-1, 1))`` to calibrate.
        Raw scores outside ``[0, 1]`` are clipped to that range before calibration.

    Raises
    ------
    ValueError
        If any benign score is NaN or falls outside ``[0, 1]``, fewer than two
        benign scores are given, ``n_knots`` is not an integer of at least 11, or
        ``plotting_position`` is not ``"filliben"`` or ``"mean"``.
    """
    if (
        isinstance(n_knots, bool)
        or not isinstance(n_knots, numbers.Integral)
        or n_knots < len(_DECADE_FPRS)
    ):
        raise ValueError(
            f"n_knots must be an integer of at least {len(_DECADE_FPRS)} (one per FPR "
            f"decade from 1e-10 to 1), got {n_knots!r}"
        )

    benign_scores = np.asarray(benign_scores, dtype=np.float64).ravel()

    # Written as "not inside [0, 1]" so NaN, which fails every comparison, is rejected.
    outside = ~((benign_scores >= 0) & (benign_scores <= 1))
    if outside.any():
        raise ValueError(
            f"Scores must be in [0, 1], got {np.count_nonzero(outside)} value(s) outside "
            f"that range or NaN, such as {benign_scores[outside][0]}"
        )

    # Plotting positions depend only on the sample size, so computing them before
    # any fitting rejects samples too small to calibrate with this module's error.
    fprs = _plotting_positions(len(benign_scores), plotting_position)

    # Fit the rescaler on the input domain [0, 1] rather than on the
    # observed calibration range. This makes the rescaler a pure
    # ``raw * SCORE_MAX`` map, so inference scores above calib.max (but
    # still in [0, 1]) land inside the isotonic's training range rather
    # than clipping to calibrated 1.0. Isotonic is invariant to monotone
    # rescale, so calibrated outputs for scores in [0, calib.max] are
    # unchanged; scores in (calib.max, 1.0] now get honest calibrated
    # values instead of the "flag nothing" terminal. ``clip=True`` maps raw
    # inference scores outside [0, 1] onto that range, so outputs never
    # exceed SCORE_MAX.
    rescaler = MinMaxScaler(feature_range=(0, SCORE_MAX), clip=True)
    rescaler.fit(np.array([[0.0], [1.0]]))
    scores_scaled = rescaler.transform(benign_scores.reshape(-1, 1)).ravel()

    # Empirical CDF: (FPR, score) pairs.
    # Evaluation counts still use the inclusive convention
    # FPR(threshold) = P(score >= threshold). The first spline maps ordered
    # thresholds to population FPR coordinates, so it uses a plotting position
    # for each rank.
    sorted_scores = np.sort(scores_scaled)

    fpr1, fpr2 = fprs[-2], fprs[-1]
    score1, score2 = sorted_scores[-2], sorted_scores[-1]

    # Sample FPR values for the knots, keeping the first spline's edge FPRs.
    sampled_fprs = _sample_fpr_values(n_knots)
    first_spline_edge_fprs = np.array([fprs[0], fpr1, fpr2], dtype=np.float64)
    sampled_fprs = np.unique(np.concatenate((sampled_fprs, first_spline_edge_fprs)))

    # Steps 1 and 2: read each sampled FPR's score off the empirical FPR -> score
    # curve and pair it with the contract's calibrated value. Scores ascend as
    # plotting positions descend, so the pairs are already monotone and their
    # isotonic fit is the linear interpolation through them; np.interp needs
    # ascending x, hence the reversal. Below the smallest fitted FPR, extrapolate
    # linearly from the two highest benign thresholds instead of clipping to the
    # first spline's top score.
    interpolated = np.interp(sampled_fprs, fprs[::-1], sorted_scores[::-1])
    slope = (score2 - score1) / (fpr2 - fpr1)
    extrapolated = np.clip(score1 + slope * (sampled_fprs - fpr1), 0, SCORE_MAX)
    knot_scores_arr = np.where(sampled_fprs >= fpr2, interpolated, extrapolated)
    knot_calibrated_arr = fpr_to_calibrated(sampled_fprs)

    # Boundary pairs for the final rescaled-score-to-calibrated-score fit.
    # Valid raw inputs are rescaled to at most SCORE_MAX and do not reach x=1.0.
    knot_scores_arr = np.append(knot_scores_arr, [SCORE_MAX, 1.0])
    knot_calibrated_arr = np.append(knot_calibrated_arr, [0.99, 1.0])

    # Step 3: final IsotonicRegression (score -> calibrated).
    score_to_cal = IsotonicRegression(increasing=True, out_of_bounds="clip")
    score_to_cal.fit(knot_scores_arr, knot_calibrated_arr)

    pipeline = Pipeline([("rescale", rescaler), ("isotonic", score_to_cal)])
    setattr(pipeline, "plotting_position_", plotting_position)  # noqa: B010

    if keep_debug:
        # Debug artifacts for plotting; attached only when keep_debug=True so
        # production pipelines serialize to a bounded size via joblib.dump.
        fpr_to_score = IsotonicRegression(increasing=False, out_of_bounds="clip")
        setattr(pipeline, "fpr_to_score_", fpr_to_score.fit(fprs, sorted_scores))  # noqa: B010
        setattr(pipeline, "sampled_fprs_", sampled_fprs)  # noqa: B010
        setattr(pipeline, "sampled_scores_", knot_scores_arr[:-2])  # noqa: B010

    return pipeline
