import math

import numpy as np
import pytest
from fast_ecod import ECODScoreMethod, InductiveECOD


def test_scores_shape(data: np.ndarray):
    """
    Check that scoring yields one finite float per predicted sample.

    :param data: Gaussian feature matrix.
    """
    model = InductiveECOD().fit(data.tolist())
    scores = model.decision_function(data[:, :7].tolist())
    assert len(scores) == 7
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores)


def test_default_method_is_max(data: np.ndarray):
    """
    Check that scoring without a method matches the max method.

    :param data: Gaussian feature matrix.
    """
    x = data.tolist()
    model = InductiveECOD().fit(x)
    assert model.decision_function(x) == model.decision_function(x, ECODScoreMethod.MAX)


def test_accepts_numpy(data: np.ndarray):
    """
    Check that numpy arrays give the same scores as nested lists.

    :param data: Gaussian feature matrix.
    """
    from_list = InductiveECOD().fit(data.tolist()).decision_function(data.tolist())
    from_numpy = InductiveECOD().fit(data).decision_function(data)
    assert from_list == from_numpy


def test_predict_before_fit_raises():
    """
    Check that scoring an unfitted model raises.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        InductiveECOD().decision_function([[1.0, 2.0]])


def test_feature_mismatch_raises(data: np.ndarray):
    """
    Check that scoring with a different feature count raises.

    :param data: Gaussian feature matrix.
    """
    model = InductiveECOD().fit(data.tolist())
    with pytest.raises(ValueError, match="features"):
        model.decision_function(data[:2].tolist())
