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

"""Regression tests: serialized pipelines stay bounded and restore faithfully."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest

from fprcal import fit_calibration_pipeline

ARTIFACT_SIZE_LIMIT_BYTES = 200_000


@pytest.mark.parametrize("n_benign", [10_000, 100_000])
def test_artifact_is_bounded_and_restores_identical_predictions(n_benign: int) -> None:
    """A default fit serializes to a bounded size and loads back to the same predictions."""
    rng = np.random.default_rng(seed=0)
    pipeline = fit_calibration_pipeline(rng.random(n_benign), n_knots=10_000)

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    size = buf.tell()
    buf.seek(0)
    restored = joblib.load(buf)

    assert size < ARTIFACT_SIZE_LIMIT_BYTES, (
        f"Serialized size {size} bytes exceeded limit {ARTIFACT_SIZE_LIMIT_BYTES} "
        f"for n_benign={n_benign}"
    )
    probe = np.linspace(0.0, 1.0, 1001).reshape(-1, 1)
    np.testing.assert_array_equal(restored.predict(probe), pipeline.predict(probe))


def test_debug_artifact_is_larger_than_production() -> None:
    """keep_debug=True is heavier; ensure the default path is strictly smaller."""
    rng = np.random.default_rng(seed=0)
    benign_scores = rng.random(50_000)

    prod = fit_calibration_pipeline(benign_scores, n_knots=10_000)
    debug = fit_calibration_pipeline(benign_scores, n_knots=10_000, keep_debug=True)

    prod_buf = io.BytesIO()
    debug_buf = io.BytesIO()
    joblib.dump(prod, prod_buf)
    joblib.dump(debug, debug_buf)

    assert prod_buf.tell() < debug_buf.tell()
