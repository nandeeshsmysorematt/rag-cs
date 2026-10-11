"""Plain data containers shared across modules (no heavy imports)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Passage:
    id: int
    text: str
    title: str = ""
    score: float = 0.0

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text
