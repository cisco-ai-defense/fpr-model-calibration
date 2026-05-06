"""FPR-based calibration using sklearn Pipeline.

Two-step fitting, single-step inference:
1. Fit temporary IsotonicRegression: FPR -> score (discarded after fitting).
2. Fit final IsotonicRegression: score -> calibrated (kept in pipeline).

The final pipeline stores about ``n_knots`` (score, calibrated) knots regardless
of training set size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SCORE_MAX: float = 0.99

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
_FPR_KNOTS: NDArray[np.float64] = 10.0**_LOG_FPR_KNOTS


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

    Returns FPR values (not log) sorted in increasing order.
    """
    # Explicit knot FPRs (excluding 1e-10 which is a theoretical limit).
    explicit_fprs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    n_explicit = len(explicit_fprs)
    n_fill = n_total - n_explicit
    n_gaps = n_explicit - 1
    points_per_gap = n_fill // n_gaps

    all_fprs: list[float] = []
    for i in range(n_gaps):
        fpr_lo = explicit_fprs[i]
        fpr_hi = explicit_fprs[i + 1]
        fill = np.logspace(np.log10(fpr_lo), np.log10(fpr_hi), points_per_gap + 2)[1:-1]
        all_fprs.append(fpr_lo)
        all_fprs.extend(fill)
    all_fprs.append(explicit_fprs[-1])

    return np.array(sorted(set(all_fprs)), dtype=np.float64)


def _extrapolate_score(fpr: float, fpr1: float, score1: float, fpr2: float, score2: float) -> float:
    """Linear extrapolation from two (FPR, score) points."""
    slope = (score2 - score1) / (fpr2 - fpr1)
    return score1 + slope * (fpr - fpr1)


def fit_calibration_pipeline(benign_scores: NDArray[np.float64], n_knots: int = 10000) -> Pipeline:
    """Fit a whole-curve FPR calibration pipeline from benign scores.

    The pipeline maps raw detector scores to calibrated values in ``[0, 1]``
    where every calibrated value has a stable FPR meaning (``0.5 = 0.1% FPR``,
    ``0.7 = 0.01%``, and so on).

    Two-step fitting:

    1. Fit a temporary ``IsotonicRegression`` on the benign empirical CDF to get
       ``FPR -> score``.
    2. Sample ~``n_knots`` FPR values on a log-spaced grid (concentrated at low
       FPR), compute ``(score, calibrated)`` pairs.
    3. Fit the final ``IsotonicRegression`` on those pairs as ``score -> calibrated``.

    Parameters
    ----------
    benign_scores : ndarray
        Scores from benign/negative samples, in ``[0, 1]``.
    n_knots : int, optional
        Approximate number of knots to store in the final pipeline. Default 10000.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted sklearn Pipeline with ``MinMaxScaler`` and ``IsotonicRegression``
        stages. Call ``pipeline.predict(scores.reshape(-1, 1))`` to calibrate.

    Raises
    ------
    ValueError
        If any benign score falls outside ``[0, 1]``.
    """
    benign_scores = np.asarray(benign_scores, dtype=np.float64).ravel()

    if np.any(benign_scores < 0) or np.any(benign_scores > 1):
        raise ValueError(
            f"Scores must be in [0, 1], got [{benign_scores.min()}, {benign_scores.max()}]"
        )

    rescaler = MinMaxScaler(feature_range=(0, SCORE_MAX))
    scores_scaled = rescaler.fit_transform(benign_scores.reshape(-1, 1)).ravel()

    # Empirical CDF: (score, FPR) pairs.
    sorted_scores = np.sort(scores_scaled)
    n = len(sorted_scores)
    fprs = 1 - np.arange(1, n + 1) / n

    # Step 1: temporary IsotonicRegression (FPR -> score).
    fpr_to_score = IsotonicRegression(increasing=False, out_of_bounds="clip")
    fpr_to_score.fit(fprs, sorted_scores)

    min_fpr = 1 / n
    fpr1, fpr2 = fprs[-2], fprs[-1]
    score1, score2 = sorted_scores[-2], sorted_scores[-1]

    # Step 2: sample FPR values, compute (score, calibrated) pairs.
    sampled_fprs = _sample_fpr_values(n_knots)

    knot_scores: list[float] = []
    knot_calibrated: list[float] = []

    for fpr in sampled_fprs:
        if fpr >= min_fpr:
            score = float(fpr_to_score.predict([[fpr]])[0])
        else:
            score = _extrapolate_score(fpr, fpr1, score1, fpr2, score2)
            score = float(np.clip(score, 0, SCORE_MAX))

        log_fpr = np.log10(max(fpr, 1e-10))
        calibrated = float(np.interp(log_fpr, _LOG_FPR_KNOTS, _CAL_KNOTS))

        knot_scores.append(score)
        knot_calibrated.append(calibrated)

    knot_scores_arr = np.array(knot_scores, dtype=np.float64)
    knot_calibrated_arr = np.array(knot_calibrated, dtype=np.float64)

    # Anchors: (0.99, 0.99), (1.0, 1.0) so threshold=1 means "flag nothing".
    knot_scores_arr = np.append(knot_scores_arr, [SCORE_MAX, 1.0])
    knot_calibrated_arr = np.append(knot_calibrated_arr, [0.99, 1.0])

    # Step 3: final IsotonicRegression (score -> calibrated).
    score_to_cal = IsotonicRegression(increasing=True, out_of_bounds="clip")
    score_to_cal.fit(knot_scores_arr, knot_calibrated_arr)

    pipeline = Pipeline([("rescale", rescaler), ("isotonic", score_to_cal)])

    # Attach debug artifacts. sklearn Pipeline allows arbitrary attribute
    # assignment; these are kept around for the validation plot, not used at
    # inference. The type checker does not know about these, so a light cast
    # keeps ty happy without changing runtime behavior.
    setattr(pipeline, "fpr_to_score_", fpr_to_score)  # noqa: B010
    setattr(pipeline, "sampled_fprs_", sampled_fprs)  # noqa: B010
    setattr(pipeline, "sampled_scores_", knot_scores_arr[:-2])  # noqa: B010

    return pipeline
