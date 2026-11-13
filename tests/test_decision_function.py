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
    scores = model.decision_function(data[:7].tolist())
    assert len(scores) == 7
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores)


def test_default_method_is_max(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that scoring without a method matches the max method.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    x = data.tolist()
    default = model_cls().fit(x).decision_function(x)
    assert default == model_cls(method="max").fit(x).decision_function(x)


def test_accepts_numpy(data: np.ndarray, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that numpy arrays give the same scores as nested lists.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    from_list = model_cls().fit(data.tolist()).decision_function(data.tolist())
    from_numpy = model_cls().fit(data).decision_function(data)
    assert from_list == from_numpy


@pytest.mark.parametrize(
    "layout",
    [
        np.ascontiguousarray,
        np.asfortranarray,
        lambda a: np.repeat(a, 2, axis=1)[:, ::2],
        lambda a: a[::-1].copy()[::-1],
        lambda a: (a * 100).astype(np.int64),
        lambda a: a[:1],
        lambda a: a[:, :1],
    ],
    ids=["c", "fortran", "strided", "reversed", "int", "one-row", "one-col"],
)
def test_numpy_layouts_match_lists(
    data: np.ndarray, model_cls: type[InductiveECOD | ECOD], layout
):
    """
    Check that any numpy layout or dtype scores like the same values as nested lists.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    :param layout: Function building the numpy input from data.
    """
    x = layout(data)
    from_list = model_cls().fit(x.tolist()).decision_function(x.tolist())
    assert model_cls().fit(x).decision_function(x) == from_list


def test_float32_input_is_accepted(
    data: np.ndarray, model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that float32 arrays are converted to float64.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    """
    x = data.astype(np.float32)
    expected = (
        model_cls().fit(x.astype(np.float64)).decision_function(x.astype(np.float64))
    )
    assert model_cls().fit(x).decision_function(x) == expected


@pytest.mark.parametrize("method", ["linear", "unify"])
def test_predict_proba_shape(
    data: np.ndarray, model_cls: type[InductiveECOD | ECOD], method: str
):
    """
    Check that probabilities are one [inlier, outlier] pair per sample, as in PyOD.

    :param data: Gaussian feature matrix.
    :param model_cls: Model class under test.
    :param method: Probability conversion method.
    """
    proba = np.array(model_cls().fit(data).predict_proba(data[:7], method))
    assert proba.shape == (7, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


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
        model.decision_function(data[:, :2].tolist())
