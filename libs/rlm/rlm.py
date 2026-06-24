from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RLMResult:
    response: str
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.response

class RLM(ABC):
    @abstractmethod
    def completion(self, context: list[str] | str | dict[str, str], query: str) -> Any:
        pass

    @abstractmethod
    def cost_summary(self) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass
