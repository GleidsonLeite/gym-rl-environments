from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class SParameters:
    frequency: NDArray
    S11: NDArray
    S12: NDArray
    S21: NDArray
    S22: NDArray
