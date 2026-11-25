"""Output guardrails: grounded + cited answers, refusal when there is no evidence.

1. citations : every answer must cite [n] that exists in the context. A missing
               citation is repaired (attached to the best-supporting context) only
               if the answer is still well grounded; otherwise the answer is refused.
2. grounding : share of the answer's content tokens that occur in the cited
               chunks' full text (reference-free faithfulness proxy).
3. leakage   : the answer must not echo prompt scaffolding (system rules).
"""
from __future__ import annotations

import re

from ..index.bm25 import tokenize

CITE = re.compile(r"\[(\d{1,2})\]")
REFUSAL = re.compile(r"(could not find|cannot find|can't find|no evidence|not (present|available|mentioned) in the context|"
                     r"i don't know|not enough information|insufficient (evidence|information))", re.I)
LEAK = re.compile(r"(Rules:|Use ONLY the numbered context|C \(Character\)|Available action:)")


def grounding_score(answer: str, texts: list[str]) -> float:
    a = [t for t in tokenize(CITE.sub(" ", answer)) if len(t) > 1]
    if not a:
        return 0.0
    pool = set()
    for t in texts:
        pool |= set(tokenize(t))
    return sum(1 for t in a if t in pool) / len(a)


class OutputGuard:
    def __init__(self, cfg):
        g = cfg.get("guardrails.output", {})
        self.require_citation = g.get("require_citation", True)
        self.min_grounding = g.get("min_grounding", 0.35)
        self.refusal = g.get("refusal_text")

    def check(self, answer: str, contexts) -> dict:
        res = {"answer": answer, "refused": False, "reason": "", "grounding": 0.0, "citations": [],
               "citation_repaired": False}
        if not answer.strip():
            res.update(answer=self.refusal, refused=True, reason="output: empty answer")
            return res
        if LEAK.search(answer):
            res.update(answer=self.refusal, refused=True, reason="output: prompt leakage")
            return res
        by_idx = {c.idx: c for c in contexts}
        if REFUSAL.search(answer):
            # A partial "X is not mentioned" next to cited facts is still a useful answer:
            # drop only the refusal sentences. Refuse when nothing cited is left.
            sents = re.split(r"(?<=[.!?])\s+", answer.strip())
            kept = [x for x in sents if not REFUSAL.search(x)]
            if kept and any(int(n) in by_idx for x in kept for n in CITE.findall(x)):
                answer = " ".join(kept)
            else:
                res.update(answer=self.refusal, refused=True, reason="output: model found no evidence")
                return res
        cited = sorted({int(n) for n in CITE.findall(answer) if int(n) in by_idx})
        if not cited:
            # repair: attach to best supporting context if well grounded
            if contexts:
                best = max(contexts, key=lambda c: grounding_score(answer, [c.full_text]))
                g = grounding_score(answer, [best.full_text])
                if g >= self.min_grounding:
                    answer = answer.rstrip() + f" [{best.idx}]"
                    cited = [best.idx]
                    res["citation_repaired"] = True
            if not cited and self.require_citation:
                res.update(answer=self.refusal, refused=True, reason="output: no valid citation")
                return res
        g = grounding_score(answer, [by_idx[i].full_text for i in cited])
        res["grounding"] = g
        if g < self.min_grounding:
            res.update(answer=self.refusal, refused=True, reason=f"output: low grounding ({g:.2f})")
            return res
        res["answer"] = answer
        res["citations"] = [{"n": i, "chunk_id": by_idx[i].chunk_id, "source": by_idx[i].source,
                             "page": by_idx[i].page, "modality": by_idx[i].modality} for i in cited]
        return res
