# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def benign_scores() -> NDArray[np.float64]:
    """Deterministic benign-score array used across calibration tests."""
    rng = np.random.default_rng(42)
    return rng.random(1000)
