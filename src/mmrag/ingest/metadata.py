"""Ontology-driven metadata tagging (doc_type, topic, topics, sensitivity)."""
from __future__ import annotations

import re
from collections import Counter


class OntologyTagger:
    def __init__(self, ontology: dict):
        self.onto = ontology
        self.doc_types = {k: [w.lower() for w in v["keywords"]] for k, v in ontology["doc_types"].items()}
        self.topics = {k: [w.lower() for w in v["keywords"]] for k, v in ontology["topics"].items()}
        self.vocab = sorted({w for ws in self.topics.values() for w in ws} |
                            {w for ws in self.doc_types.values() for w in ws})

    @staticmethod
    def _hits(text: str, words: list[str]) -> int:
        t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]", " ", text.lower())) + " "
        n = 0
        for w in words:
            w = w.strip()
            # long words / phrases: substring match; short tokens: whole-word match
            n += t.count(w) if (len(w) > 6 or " " in w) else t.count(" " + w + " ")
        return n

    def doc_type(self, title: str, text: str) -> str:
        scores = {k: 3 * self._hits(title, ws) + self._hits(text[:4000], ws) for k, ws in self.doc_types.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "architecture"

    def topic_scores(self, text: str) -> Counter:
        return Counter({k: self._hits(text, ws) for k, ws in self.topics.items()})

    def topics_for(self, text: str, top: int = 3) -> tuple[str, str]:
        sc = self.topic_scores(text)
        ranked = [k for k, v in sc.most_common() if v > 0][:top]
        return (ranked[0] if ranked else "ran"), ",".join(ranked)

    def domain_hits(self, text: str) -> int:
        return sum(self.topic_scores(text).values()) + sum(self._hits(text, ws) for ws in self.doc_types.values())
