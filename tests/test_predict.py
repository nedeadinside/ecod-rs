import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD


def test_predict_returns_labels_array(
    data: np.ndarray, model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that predict yields an int64 array of 0/1 labels, one per sample.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    labels = model_cls().fit(data).predict(data)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64 and labels.shape == (len(data),)
    assert set(np.unique(labels)) <= {0, 1}


def test_predict_before_fit_raises(model_cls: type[InductiveECOD | ECOD]):
    """
    Check that predicting with an unfitted model raises.

    :param model_cls: Model class under test.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        model_cls().predict([[1.0, 2.0]])
