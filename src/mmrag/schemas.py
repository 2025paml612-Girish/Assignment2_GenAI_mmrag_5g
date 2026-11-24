"""Shared data structures (the 'schemas' of the LLD).

Element  -> one parsed unit from a page (text block, table, image)
Chunk    -> one indexable unit with full metadata (see config/ontology.yaml chunk_schema)
Hit      -> a retrieved chunk with its scores
Answer   -> the final, guarded system response
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Element:
    doc_id: str
    source: str
    page: int
    modality: str                    # text | table | image
    content: str                     # text / markdown table / OCR text + caption
    section: str = ""
    bbox: tuple | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    title: str
    page: int
    modality: str
    topic: str
    topics: str
    section: str
    date: str
    sensitivity: str
    text: str
    tokens: int

    def meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float = 0.0
    sparse: float = 0.0
    rank_sources: dict = field(default_factory=dict)


@dataclass
class Answer:
    query: str
    rewritten_query: str
    answer: str
    citations: list[dict]
    hits: list[Hit]
    variant: str
    refused: bool = False
    refusal_reason: str = ""
    grounding: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    timings: dict = field(default_factory=dict)
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved": [h.chunk.chunk_id for h in self.hits],
            "variant": self.variant,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "grounding": round(self.grounding, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 4),
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "from_cache": self.from_cache,
        }
