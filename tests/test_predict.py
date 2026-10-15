import pytest
from ecod_rs import ECOD, InductiveECOD


def test_predict_before_fit_raises(model_cls: type[InductiveECOD | ECOD]):
    """
    Check that predicting with an unfitted model raises.

    :param model_cls: Model class under test.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        model_cls().predict([[1.0, 2.0]])
