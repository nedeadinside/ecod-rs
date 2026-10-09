import numpy as np
import pytest

N_FEATURES, N_SAMPLES = 3, 50


@pytest.fixture
def data() -> np.ndarray:
    """
    Build gaussian matrix.

    :return: Matrix of shape (features, samples).
    """
    return np.random.default_rng(0).normal(size=(N_FEATURES, N_SAMPLES))
