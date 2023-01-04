from typing import Dict, Literal
import requests

from src.modules.shared.entities.SParameters import SParameters
from src.modules.shared.configuration.api import base_url
from src.modules.shared.errors import AppError


class ExtractLNASParametersUseCase:
    @staticmethod
    def execute(
        circuit_parameters: Dict[str, float],
        start_frequency: float,
        stop_frequency: float,
        number_of_points: int,
        variation: Literal["dec", "oct", "lin"] = "lin",
    ) -> SParameters:
        response = None
        success = False

        simulation_configuration = {
            "startFrequency": start_frequency,
            "stopFrequency": stop_frequency,
            "simulationPoints": number_of_points,
            "simulationVariation": variation,
            "cdec": 1e-6,
        }
        while not success:
            try:
                response = requests.post(
                    f"{base_url}/extractSParameters",
                    json=simulation_configuration | circuit_parameters,
                )
                data = response.json()

                if data["error"]:
                    success = True
                    raise AppError
                if response.status_code == 201:
                    success = True
            except AppError as app_error:
                raise app_error

            except Exception as exception:
                pass
        data = response.json()
        return SParameters(
            frequency=data["frequency"],
            S11=data["S11"],
            S12=data["S12"],
            S21=data["S21"],
            S22=data["S22"],
        )
