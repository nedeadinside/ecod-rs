from pathlib import Path

import numpy as np
import pytest
from ecod_rs import InductiveECOD


@pytest.mark.parametrize("level", [None, -5, 1, 22])
def test_save_load_roundtrip(data: np.ndarray, tmp_path: Path, level: int | None):
    """
    Check that a loaded model scores like the saved one.

    :param data: Gaussian feature matrix.
    :param tmp_path: Temporary directory from pytest.
    :param level: Compression level passed to save.
    """
    x = data.tolist()
    model = InductiveECOD().fit(x)
    path = tmp_path / "model.bin"
    model.save(path, compression_level=level)
    assert InductiveECOD.load(path).decision_function(x) == model.decision_function(x)


def test_save_before_fit_raises(tmp_path: Path):
    """
    Check that saving an unfitted model raises.

    :param tmp_path: Temporary directory from pytest.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        InductiveECOD().save(tmp_path / "model.bin")


def test_save_level_is_keyword_only(data: np.ndarray, tmp_path: Path):
    """
    Check that compression level cannot be passed positionally.

    :param data: Gaussian feature matrix.
    :param tmp_path: Temporary directory from pytest.
    """
    model = InductiveECOD().fit(data.tolist())
    with pytest.raises(TypeError):
        model.save(tmp_path / "model.bin", 3)


def test_load_missing_file_raises(tmp_path: Path):
    """
    Check that loading a missing file raises.

    :param tmp_path: Temporary directory from pytest.
    """
    with pytest.raises(FileNotFoundError):
        InductiveECOD.load(tmp_path / "missing.bin")


def test_load_garbage_raises(tmp_path: Path):
    """
    Check that loading a non-model file raises.

    :param tmp_path: Temporary directory from pytest.
    """
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="decompress"):
        InductiveECOD.load(path)
