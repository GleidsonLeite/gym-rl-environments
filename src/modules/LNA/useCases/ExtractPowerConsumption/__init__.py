from typing import Dict
import requests

from src.modules.shared.configuration.api import base_url
from src.modules.shared.errors import AppError


class ExtractPowerConsumptionUseCase:
    @staticmethod
    def execute(
        circuit_parameters: Dict[str, float],
    ) -> float:
        response = None
        success = False
        while not success:
            try:
                response = requests.post(
                    f"{base_url}/extractPowerConsumption",
                    json=circuit_parameters,
                )
                data = response.json()

                if data["error"]:
                    success = True
                    raise AppError
                if response.status_code == 201:
                    success = True
            except AppError as app_error:
                raise app_error

        data = response.json()
        return data["power_consumption"]
