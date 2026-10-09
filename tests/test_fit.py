import math

import numpy as np
import pytest
from fast_ecod import InductiveECOD


def test_fit_returns_self(data: np.ndarray):
    """
    Check that fitting returns the same model for chaining.

    :param data: Gaussian feature matrix.
    """
    model = InductiveECOD()
    assert model.fit(data.tolist()) is model


@pytest.mark.parametrize(
    "x",
    [
        [],
        [[1.0, 2.0], [3.0]],
        [[1.0, math.nan]],
        [[1.0, math.inf]],
    ],
    ids=["empty", "ragged", "nan", "inf"],
)
def test_fit_rejects_bad_input(x: list[list[float]]):
    """
    Check that fitting on malformed input raises.

    :param x: Malformed feature matrix.
    """
    with pytest.raises(ValueError):
        InductiveECOD().fit(x)
