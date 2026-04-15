from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    parameters: dict[str, Any]
    report: dict[str, Any]
