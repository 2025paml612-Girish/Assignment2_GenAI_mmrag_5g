"""Short-term conversational memory + follow-up query rewriting (E4).

Each session keeps the last ``memory.short_term_turns`` turns: query, rewritten
query, answer, entities and the documents that were cited. A follow-up such as
"And what software build does the O-DU there run?" is detected (anaphora,
leading "and / what about", or no entity of its own) and rewritten into a
standalone query by carrying over the previous turn's entities; the documents
cited in the previous turn are boosted during retrieval.

``memory.rewrite: llm`` asks the LLM to rewrite instead (when a generative model
is available); the rule-based rewrite is the deterministic default.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

ANAPHORA = re.compile(r"\b(it|its|this|that|there|they|them|those|these|same|above|former|latter|the site|the cell)\b", re.I)
LEADING = re.compile(r"^\s*(and|also|what about|how about|then)\b", re.I)


@dataclass
class Turn:
    query: str
    rewritten: str
    answer: str
    entities: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)


class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        self.sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns))

    def history(self, session: str) -> list[Turn]:
        return list(self.sessions[session])

    def clear(self, session: str) -> None:
        self.sessions.pop(session, None)

    def add(self, session: str, turn: Turn) -> None:
        self.sessions[session].append(turn)

    @staticmethod
    def is_followup(query: str, entities: list[str]) -> bool:
        return bool(LEADING.search(query)) or (not entities and (bool(ANAPHORA.search(query)) or len(query.split()) <= 6))

    def rewrite(self, session: str, query: str, entities: list[str], llm=None) -> tuple[str, list[str], bool]:
        """returns (standalone query, documents to boost, was_followup)"""
        hist = self.history(session)
        if not hist or not self.is_followup(query, entities):
            return query, [], False
        last = next((t for t in reversed(hist) if t.entities), hist[-1])
        if llm is not None and getattr(llm, "generative", False):
            prompt = ("Rewrite the follow-up question as a standalone question. Keep identifiers exactly.\n"
                      f"Previous question: {last.rewritten}\nFollow-up: {query}\nStandalone question:")
            try:
                out = llm.generate([{"role": "user", "content": prompt}], max_tokens=60).text.strip().splitlines()[0]
                if out:
                    return out, last.doc_ids, True
            except Exception:
                pass
        missing = [e for e in last.entities if e.lower() not in query.lower()]
        rewritten = query.rstrip("?. ") + (f" (regarding {', '.join(missing)})" if missing else "") + "?"
        return rewritten, last.doc_ids, True
