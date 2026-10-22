"""Validated YAML configuration; importing the package needs no simulator."""

from math import isclose
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field, model_validator

from telemetry_reasoning.schemas import (
    LOGGING_DT_S, ContractModel, Nonnegative, Number, TelemetryFeature,
)


class SimulationConfig(ContractModel):
    sim_hz: Annotated[int, Field(strict=True, gt=0)] = 500
    sim_dt_s: Annotated[Number, Field(gt=0)] = 0.002
    logging_hz: Annotated[int, Field(strict=True, gt=0)] = 50
    logging_dt_s: Annotated[Number, Field(gt=0)] = LOGGING_DT_S
    duration_min_s: Annotated[Number, Field(gt=0)] = 6.0
    duration_max_s: Annotated[Number, Field(gt=0)] = 10.0
    mock_duration_s: Annotated[Number, Field(gt=0)] = 8.0

    @model_validator(mode="after")
    def consistent_timing(self) -> "SimulationConfig":
        if not isclose(self.sim_hz * self.sim_dt_s, 1.0):
            raise ValueError("sim_hz and sim_dt_s must be reciprocal")
        if self.logging_hz != 50 or not isclose(self.logging_dt_s, LOGGING_DT_S):
            raise ValueError("the v1 telemetry contract requires 50 Hz / 0.02 s logging")
        if self.sim_hz % self.logging_hz:
            raise ValueError("simulation rate must be an integer multiple of logging rate")
        if not self.duration_min_s <= self.mock_duration_s <= self.duration_max_s:
            raise ValueError("mock_duration_s must lie within the configured duration range")
        if not isclose(self.mock_duration_s * self.logging_hz,
                       round(self.mock_duration_s * self.logging_hz)):
            raise ValueError("mock duration must contain a whole number of logging steps")
        return self


class CompressionConfig(ContractModel):
    k_uniform_samples: Annotated[int, Field(strict=True, ge=0)] = 5
    major_event_thresholds: dict[TelemetryFeature, Nonnegative] = Field(default_factory=dict)


class ProjectConfig(ContractModel):
    simulation: SimulationConfig
    parser: CompressionConfig


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return ProjectConfig.model_validate(yaml.safe_load(stream))
