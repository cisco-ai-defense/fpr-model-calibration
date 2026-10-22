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

"""Performance checks for fitting and inference.

The two ratio tests compare timings taken on the same machine, so they catch
per-knot and super-linear regressions regardless of runner speed. The absolute
budgets leave more than 15x headroom over the cost measured when they were
added, so a slow CI runner does not fail them. Deselect with
``-m "not performance"``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from fprcal import fit_calibration_pipeline

pytestmark = pytest.mark.performance


def _best_time(fn: Callable[[], object], repeats: int) -> float:
    """Return the fastest of several wall-clock runs, which filters scheduler noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_fit_cost_does_not_scale_with_knot_count() -> None:
    """A 100x larger knot grid must not multiply fit time by more than 10x."""
    scores = np.random.default_rng(0).beta(2, 8, 10_000)
    few = _best_time(lambda: fit_calibration_pipeline(scores, n_knots=100), repeats=7)
    many = _best_time(lambda: fit_calibration_pipeline(scores, n_knots=10_000), repeats=7)
    assert many < 10 * few, (
        f"n_knots=10000 took {many * 1e3:.1f} ms vs {few * 1e3:.1f} ms for n_knots=100"
    )


def test_fit_time_scales_near_linearly_with_training_size() -> None:
    """Ten times more benign scores must cost less than twenty times the fit time."""
    scores = np.random.default_rng(0).beta(2, 8, 1_000_000)
    small = _best_time(lambda: fit_calibration_pipeline(scores[:100_000]), repeats=3)
    large = _best_time(lambda: fit_calibration_pipeline(scores), repeats=3)
    assert large < 20 * small, f"1M scores took {large:.3f} s vs {small:.3f} s for 100k"


def test_million_score_fit_and_inference_meet_time_budget() -> None:
    """Fitting on and scoring one million values each stay within their time budgets."""
    rng = np.random.default_rng(0)
    scores = rng.beta(2, 8, 1_000_000)
    probe = rng.random(1_000_000).reshape(-1, 1)
    pipeline = fit_calibration_pipeline(scores)

    fit_seconds = _best_time(lambda: fit_calibration_pipeline(scores), repeats=3)
    predict_seconds = _best_time(lambda: pipeline.predict(probe), repeats=3)

    assert fit_seconds < 2.0, f"Fitting 1M scores took {fit_seconds:.3f} s"
    assert predict_seconds < 1.0, f"Scoring 1M values took {predict_seconds:.3f} s"
