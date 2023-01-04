from typing import Dict, List
import numpy as np


class Observation:
    def __init__(self, value: List[float]) -> None:
        self.__value: Dict[str, float] = {"current": np.array([value], dtype=float)}

    @property
    def value(self) -> Dict[str, float]:
        return self.__value

    def get_list(self) -> List[float]:
        return self.__value["current"]
