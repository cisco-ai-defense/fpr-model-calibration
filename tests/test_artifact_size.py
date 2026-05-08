# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: serialized pipeline stays bounded regardless of training size."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest

from fpr_model_calibration import fit_calibration_pipeline

ARTIFACT_SIZE_LIMIT_BYTES = 200_000


@pytest.mark.parametrize("n_benign", [10_000, 100_000])
def test_artifact_size_independent_of_training_size(n_benign: int) -> None:
    """Default (production) fit must serialize to a bounded size."""
    rng = np.random.default_rng(seed=0)
    benign_scores = rng.random(n_benign)

    pipeline = fit_calibration_pipeline(benign_scores, n_knots=10_000)

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    size = buf.tell()

    assert size < ARTIFACT_SIZE_LIMIT_BYTES, (
        f"Serialized size {size} bytes exceeded limit {ARTIFACT_SIZE_LIMIT_BYTES} "
        f"for n_benign={n_benign}"
    )


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
