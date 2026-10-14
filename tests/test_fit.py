import math

import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD


def test_fit_returns_self(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that fitting returns the same model for chaining.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    model = model_cls()
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
def test_fit_rejects_bad_input(
    x: list[list[float]], model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that fitting on malformed input raises.

    :param x: Malformed feature matrix.
    :param model_cls: Model class under test.
    """
    with pytest.raises(ValueError):
        model_cls().fit(x)
