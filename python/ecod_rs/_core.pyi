# Stub file
import os
from collections.abc import Sequence
from typing import Literal, Self, final

type ECODScoreMethod = Literal["left", "right", "max", "auto"]
type ECODProbaMethod = Literal["linear", "unify"]

@final
class InductiveECOD:
    def __init__(self, contamination: float = 0.1, method: ECODScoreMethod = "max"): ...
    def fit(self, x: Sequence[Sequence[float]]) -> Self: ...
    def decision_function(self, x: Sequence[Sequence[float]]) -> Sequence[float]: ...
    def predict(self, x: Sequence[Sequence[float]]) -> Sequence[int]: ...
    def predict_proba(
        self, x: Sequence[Sequence[float]], method: ECODProbaMethod = "linear"
    ) -> Sequence[Sequence[float]]:
        """Return [inlier, outlier] probabilities, one list per class."""
    def save(
        self, path: str | os.PathLike[str], *, compression_level: int | None = None
    ) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> InductiveECOD: ...

@final
class ECOD:
    """Transductive: each sample's score depends on the whole batch passed to decision_function."""

    def __init__(self, contamination: float = 0.1, method: ECODScoreMethod = "max"): ...
    def fit(self, x: Sequence[Sequence[float]]) -> Self: ...
    def decision_function(self, x: Sequence[Sequence[float]]) -> Sequence[float]: ...
    def predict(self, x: Sequence[Sequence[float]]) -> Sequence[int]: ...
    def predict_proba(
        self, x: Sequence[Sequence[float]], method: ECODProbaMethod = "linear"
    ) -> Sequence[Sequence[float]]:
        """Return [inlier, outlier] probabilities, one list per class."""
    def save(
        self, path: str | os.PathLike[str], *, compression_level: int | None = None
    ) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> ECOD: ...
