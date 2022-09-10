import math
from typing import Dict, List, Optional, Tuple
from gym import Env, spaces
import numpy as np
import random

from src.modules.LNA.configuration import LNAMinMaxParameters
from src.modules.LNA.entities.Normalizer import LNANormalizer
from src.modules.LNA.useCases.ExtractLNASParameters import ExtractLNASParametersUseCase
from src.modules.LNA.utils.Normalization import clamp, z_normalize
from src.modules.shared.entities.SParameters import SParameters
from src.modules.shared.errors import AppError


class LNA(Env):

    NUMBER_OF_VARIABLES = 19

    MAX_RL = 1
    MIN_RL = -1
    NDTYPE = np.float32

    def __init__(self) -> None:
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([self.MIN_RL for _ in range(self.NUMBER_OF_VARIABLES)]),
            high=np.array([self.MAX_RL for _ in range(self.NUMBER_OF_VARIABLES)]),
            dtype=self.NDTYPE,
        )

        self.observation_space = spaces.Box(
            low=self.MIN_RL,
            high=self.MAX_RL,
            shape=(self.NUMBER_OF_VARIABLES,),
            dtype=self.NDTYPE,
        )

        self.__normalizer = LNANormalizer(
            high=self.MAX_RL,
            low=self.MIN_RL,
            parameters=LNAMinMaxParameters,
        )

        self.__state = {
            key: z_normalize(
                a=self.MAX_RL,
                b=self.MIN_RL,
                x=random.random(),
                x_max=1,
                x_min=0,
            )
            for key, _ in LNAMinMaxParameters.items()
        }

    def _next_observation(self):

        return np.array(
            [value for _, value in self.__state.items()],
            dtype=self.NDTYPE,
        )

    def _take_action(
        self,
        action: List[float],
        delta_normalizer: Optional[float] = None,
    ) -> None:
        normalizer = 10 if not delta_normalizer else delta_normalizer
        self.__state = {
            key: clamp(
                max_=self.MAX_RL,
                min_=self.MIN_RL,
                value=value
                + z_normalize(
                    a=-1,
                    b=1,
                    x=delta,
                    x_max=self.MAX_RL,
                    x_min=self.MIN_RL,
                )
                / normalizer,
            )
            for (key, value), delta in zip(self.__state.items(), action)
        }

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, Dict]:
        self._take_action(action)

        was_an_error_generated = False
        s_parameters: SParameters = None
        done = False
        try:
            s_parameters = ExtractLNASParametersUseCase.execute(
                circuit_parameters=self.__normalizer.inverse_transform(
                    self.__state,
                ),
                number_of_points=3,
                start_frequency=2.44e9,
                stop_frequency=2.46e9,
                variation="lin",
            )
        except Exception:
            was_an_error_generated = True

        reward = -100 if was_an_error_generated else s_parameters.S21[1]
        done = was_an_error_generated
        if math.isinf(reward):
            raise AppError

        observation = self._next_observation()
        return observation, reward, done, {}

    def reset(self) -> List[float]:
        self.__state = {
            key: z_normalize(
                a=0,
                b=1,
                x=random.random(),
                x_max=self.MAX_RL,
                x_min=self.MIN_RL,
            )
            for key, _ in LNAMinMaxParameters.items()
        }

        return self._next_observation()

    def render(self, mode="human", close=False) -> None:
        circuit_parameters = self.__normalizer.inverse_transform(self.__state)
        for key, value in circuit_parameters.items():
            print(key, value)
