from pathlib import Path

import numpy as np
import pytest
from fast_ecod import InductiveECOD


def test_save_load_roundtrip(data: np.ndarray, tmp_path: Path):
    """
    Check that a loaded model scores like the saved one.

    :param data: Gaussian feature matrix.
    :param tmp_path: Temporary directory from pytest.
    """
    x = data.tolist()
    model = InductiveECOD().fit(x)
    path = tmp_path / "model.bin"
    model.save(path)
    assert InductiveECOD.load(path).decision_function(x) == model.decision_function(x)


def test_save_before_fit_raises(tmp_path: Path):
    """
    Check that saving an unfitted model raises.

    :param tmp_path: Temporary directory from pytest.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        InductiveECOD().save(tmp_path / "model.bin")


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
    with pytest.raises(ValueError):
        InductiveECOD.load(path)
