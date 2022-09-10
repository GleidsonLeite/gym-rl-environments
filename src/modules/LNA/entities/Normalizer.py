from typing import Dict, Tuple

from src.modules.LNA.utils.Normalization import z_inverse_normalize, z_normalize


class LNANormalizer:
    def __init__(
        self,
        parameters: Dict[str, Tuple[float, float]],
        high: float,
        low: float,
    ) -> None:
        self.__parameters = parameters
        self.high = high
        self.low = low

    def transform(
        self,
        state: Dict[str, float],
    ) -> Dict[str, float]:
        return {
            key: z_normalize(
                a=self.low,
                b=self.high,
                x=value,
                x_max=self.__parameters[key][1],
                x_min=self.__parameters[key][0],
            )
            for key, value in state.items()
        }

    def inverse_transform(self, state: Dict[str, float]) -> Dict[str, float]:
        return {
            key: z_inverse_normalize(
                a=self.low,
                b=self.high,
                x=value,
                x_max=self.__parameters[key][1],
                x_min=self.__parameters[key][0],
            )
            for key, value in state.items()
        }
