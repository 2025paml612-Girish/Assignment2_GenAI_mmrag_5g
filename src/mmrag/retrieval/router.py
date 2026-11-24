"""Ontology-based query router.

Turns a query into routing signals that the retriever and the prompts consume:
  * topics     -> metadata pre-filter (narrow candidates before retrieval)
  * doc_type   -> optional filter when the query names a document type
  * modality   -> boost table / image chunks when the query asks for them
  * entities   -> identifiers such as Site-114, C114-2, Lab-07, R24.3.7 (used by memory)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest.metadata import OntologyTagger

ENTITY = re.compile(r"\b(?:Site|Lab|Cluster|Region|Module|Week|Build|TB)[- ]?\d+\b|\b[A-Z]{1,4}\d{2,4}-[A-Z0-9]+\b|"
                    r"\b[A-Z]{1,5}(?:-[A-Z]{1,5})*-\d{2,4}[A-Z]?(?:-\d+)?\b|\bR\d{2}\.\d+\.\d+\b|\b(?:DEF|ISS|POL|FH)-[\w-]+\b|\bn\d{2,3}\b", re.I)


@dataclass
class Route:
    topics: list[str] = field(default_factory=list)
    doc_type: str | None = None
    modality: str | None = None
    entities: list[str] = field(default_factory=list)
    domain_hits: int = 0

    def as_dict(self) -> dict:
        return {"topics": self.topics, "doc_type": self.doc_type, "modality": self.modality,
                "entities": self.entities, "domain_hits": self.domain_hits}


class QueryRouter:
    def __init__(self, ontology: dict):
        self.onto = ontology
        self.tagger = OntologyTagger(ontology)
        self.cues = {m: [c.lower() for c in cs] for m, cs in ontology.get("modality_cues", {}).items()}

    def route(self, query: str) -> Route:
        q = query.lower()
        sc = self.tagger.topic_scores(query)
        topics = [k for k, v in sc.most_common() if v > 0][:2]
        dt_scores = {k: self.tagger._hits(query, ws) for k, ws in self.tagger.doc_types.items()}
        doc_type = max(dt_scores, key=dt_scores.get) if max(dt_scores.values() or [0]) > 0 else None
        mod_scores = {m: sum(1 for c in cs if c in q) for m, cs in self.cues.items()}
        modality = None
        if mod_scores:
            best = max(mod_scores, key=mod_scores.get)
            if mod_scores[best] > 0:
                # image cues are more specific than generic table cues ("what is the")
                if mod_scores.get("image", 0) > 0:
                    best = "image"
                modality = best
        ents = list(dict.fromkeys(m.group(0) for m in ENTITY.finditer(query)))
        domain = self.tagger.domain_hits(query) + len(ents)
        return Route(topics, doc_type, modality, ents, domain)
