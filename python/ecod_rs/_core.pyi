# Stub file
import os
from collections.abc import Sequence
from typing import Literal, Self, final

type ECODScoreMethod = Literal["left", "right", "max", "auto"]

@final
class InductiveECOD:
    def __init__(self): ...
    def fit(self, x: Sequence[Sequence[float]]) -> Self: ...
    def decision_function(
        self,
        x: Sequence[Sequence[float]],
        method: ECODScoreMethod = "max",
    ) -> Sequence[float]: ...
    def save(
        self, path: str | os.PathLike[str], *, compression_level: int | None = None
    ) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> InductiveECOD: ...
