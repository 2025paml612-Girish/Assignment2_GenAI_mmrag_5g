"""LLM backends behind one interface.

``make_llm(cfg)`` with ``llm.backend``:
  * ollama        - local open-weight models (default qwen2.5:3b-instruct)
  * openai_compat - any OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM, LM Studio)
  * extractive    - deterministic, no-model fallback that selects the best supporting
                    sentence / table cell from the context (lets the full pipeline,
                    guardrails and evaluation run on any machine)
  * auto          - ollama if reachable, else openai_compat if configured, else extractive
"""
from __future__ import annotations

from .base import LLM, LLMResponse  # noqa: F401
from .backends import ExtractiveLLM, OllamaLLM, OpenAICompatLLM, make_llm  # noqa: F401
