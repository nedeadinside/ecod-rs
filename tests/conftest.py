import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD

N_FEATURES, N_SAMPLES = 3, 50


@pytest.fixture
def data() -> np.ndarray:
    """
    Build gaussian matrix.

    :return: Matrix.
    """
    return np.random.default_rng(0).normal(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture(params=[InductiveECOD, ECOD], ids=["inductive", "ecod"])
def model_cls(request: pytest.FixtureRequest) -> type[InductiveECOD | ECOD]:
    """
    Model class under test.

    :param request: Pytest request with the parametrized class.
    :return: Model class.
    """
    return request.param
