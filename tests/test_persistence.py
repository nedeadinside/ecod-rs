from pathlib import Path

import numpy as np
import pytest
from ecod_rs import ECOD, InductiveECOD


@pytest.mark.parametrize("level", [None, -5, 1, 22])
def test_save_load_roundtrip(
    data: np.ndarray,
    tmp_path: Path,
    level: int | None,
    model_cls: type[InductiveECOD | ECOD],
):
    """
    Check that a loaded model scores like the saved one.

    :param data: Gaussian feature matrix.
    :param tmp_path: Temporary directory from pytest.
    :param level: Compression level passed to save.
    :param model_cls: Model class under test.
    """
    x = data.tolist()
    model = model_cls().fit(x)
    path = tmp_path / "model.bin"
    model.save(path, compression_level=level)
    assert model_cls.load(path).decision_function(x) == model.decision_function(x)


def test_save_before_fit_raises(tmp_path: Path, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that saving an unfitted model raises.

    :param tmp_path: Temporary directory from pytest.
    :param model_cls: Model class under test.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        model_cls().save(tmp_path / "model.bin")


def test_save_level_is_keyword_only(
    data: np.ndarray, tmp_path: Path, model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that compression level cannot be passed positionally.

    :param data: Gaussian feature matrix.
    :param tmp_path: Temporary directory from pytest.
    :param model_cls: Model class under test.
    """
    model = model_cls().fit(data.tolist())
    with pytest.raises(TypeError):
        model.save(tmp_path / "model.bin", 3)


def test_load_missing_file_raises(
    tmp_path: Path, model_cls: type[InductiveECOD | ECOD]
):
    """
    Check that loading a missing file raises.

    :param tmp_path: Temporary directory from pytest.
    :param model_cls: Model class under test.
    """
    with pytest.raises(FileNotFoundError):
        model_cls.load(tmp_path / "missing.bin")


def test_load_garbage_raises(tmp_path: Path, model_cls: type[InductiveECOD | ECOD]):
    """
    Check that loading a non-model file raises.

    :param tmp_path: Temporary directory from pytest.
    :param model_cls: Model class under test.
    """
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="decompress"):
        model_cls.load(path)
