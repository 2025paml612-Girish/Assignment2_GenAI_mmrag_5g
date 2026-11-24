from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str


class LLM:
    name = "base"
    generative = True          # False for the extractive fallback

    def generate(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
                 stop: list[str] | None = None, **meta) -> LLMResponse:
        raise NotImplementedError
