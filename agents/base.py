"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.contracts import TaskContract


class BaseAgent(ABC):
    agent_id: str

    @abstractmethod
    def run(self, contract: TaskContract, **kwargs: Any) -> Any:
        """Execute the assigned contract and return a typed payload."""
