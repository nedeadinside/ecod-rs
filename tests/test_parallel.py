import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD


@pytest.mark.parametrize("method", ["right", "left", "auto", "pyod-max", "max"])
def test_scores_do_not_depend_on_n_jobs(model_cls: type[InductiveECOD | ECOD], method: str):
    """
    Check that scores are bit-identical for any thread count.

    Data is large enough to split rows into several chunks.

    :param model_cls: Model class under test.
    :param method: Score method.
    """
    x = np.random.default_rng(0).normal(size=(20_000, 5))
    x_new = np.random.default_rng(1).normal(size=(20_000, 5))
    scores = []
    for n_jobs in (1, 4, 0):
        model = model_cls(method=method, n_jobs=n_jobs).fit(x)
        scores.append((list(model.decision_function(x)), list(model.decision_function(x_new))))
    assert scores[0] == scores[1] == scores[2]


@pytest.mark.parametrize("n_jobs", [-1, -8])
def test_negative_n_jobs_raises(model_cls: type[InductiveECOD | ECOD], n_jobs: int):
    """
    Check that negative n_jobs is rejected.

    :param model_cls: Model class under test.
    :param n_jobs: Negative thread count.
    """
    with pytest.raises(ValueError, match="n_jobs"):
        model_cls(n_jobs=n_jobs)
