# Stub file
import os
from collections.abc import Buffer, Sequence
from typing import Literal, Self, final

type ECODScoreMethod = Literal["left", "right", "max", "pyod-max", "auto"]
type ECODProbaMethod = Literal["linear", "unify"]
type Matrix = Sequence[Sequence[float]] | Buffer

@final
class InductiveECOD:
    def __init__(self, contamination: float = 0.1, method: ECODScoreMethod = "max"): ...
    def fit(self, x: Matrix) -> Self: ...
    def decision_function(self, x: Matrix) -> Sequence[float]: ...
    def predict(self, x: Matrix) -> Sequence[int]: ...
    def predict_proba(
        self, x: Matrix, method: ECODProbaMethod = "linear"
    ) -> Sequence[Sequence[float]]:
        """Return shape (n_samples, 2): [inlier, outlier] per row, as in PyOD."""
    def save(
        self, path: str | os.PathLike[str], *, compression_level: int | None = None
    ) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> InductiveECOD: ...

@final
class ECOD:
    """Transductive: each sample's score depends on the whole batch passed to decision_function."""

    def __init__(self, contamination: float = 0.1, method: ECODScoreMethod = "max"): ...
    def fit(self, x: Matrix) -> Self: ...
    def decision_function(self, x: Matrix) -> Sequence[float]: ...
    def predict(self, x: Matrix) -> Sequence[int]: ...
    def predict_proba(
        self, x: Matrix, method: ECODProbaMethod = "linear"
    ) -> Sequence[Sequence[float]]:
        """Return [inlier, outlier] probabilities, one list per class."""
    def save(
        self, path: str | os.PathLike[str], *, compression_level: int | None = None
    ) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> ECOD: ...
