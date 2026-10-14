import math

import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD


def test_scores_shape(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that scoring yields one finite float per predicted sample.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    model = model_cls().fit(data.tolist())
    scores = model.decision_function(data[:, :7].tolist())
    assert len(scores) == 7
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores)


def test_default_method_is_max(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that scoring without a method matches the max method.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    x = data.tolist()
    model = model_cls().fit(x)
    assert model.decision_function(x) == model.decision_function(x, "max")


def test_accepts_numpy(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that numpy arrays give the same scores as nested lists.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    from_list = model_cls().fit(data.tolist()).decision_function(data.tolist())
    from_numpy = model_cls().fit(data).decision_function(data)
    assert from_list == from_numpy


def test_predict_before_fit_raises(model_cls: type[InductiveECOD | ECOD]):
    """
    Check that scoring an unfitted model raises.

    :param model_cls: Model class under test.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        model_cls().decision_function([[1.0, 2.0]])


def test_feature_mismatch_raises(
    data: np.ndarray, model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that scoring with a different feature count raises.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    model = model_cls().fit(data.tolist())
    with pytest.raises(ValueError, match="features"):
        model.decision_function(data[:2].tolist())
