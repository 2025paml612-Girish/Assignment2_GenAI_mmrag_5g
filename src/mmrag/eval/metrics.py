"""Retrieval, answer and system metrics."""
from __future__ import annotations

import re

import numpy as np

from ..index.bm25 import tokenize


UNITS = r"(?:ms|s|ns|us|db|dbm|mhz|ghz|gb|mb|mbps|min|bytes|deg|%)"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("µ", "u")).strip()


def _collapse_units(s: str) -> str:
    return re.sub(r"(\d)\s+(?=" + UNITS + r"(?![a-z]))", r"\1", s)       # '6 s' -> '6s'


def answer_correct(answer: str, key: str, aliases: list[str] | None = None) -> bool:
    """A correct answer contains the gold key (word-boundary match; '6 s' and '6s'
    are treated as equal). Refusals are never correct."""
    a = _norm(answer)
    forms_a = {a, _collapse_units(a)}
    for k in [key] + list(aliases or []):
        k = _norm(str(k))
        if not k:
            continue
        for kf in {k, _collapse_units(k)}:
            pat = r"(?<![\w.])" + re.escape(kf) + r"(?![\w]|\.\d)"
            if any(re.search(pat, af) for af in forms_a):
                return True
    return False


def token_f1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def first_relevant_rank(hits, doc_id: str, page: int, modality: str | None = None) -> int | None:
    """1-based rank of the first evidence chunk: gold document AND page (AND the
    modality that holds the answer, when given - e.g. the image chunk, not the
    caption text that happens to sit on the same page)."""
    for i, h in enumerate(hits, 1):
        if h.chunk.doc_id == doc_id and h.chunk.page == page and (modality is None or h.chunk.modality == modality):
            return i
    return None


def retrieval_summary(ranks: list[int | None], ks=(1, 3, 5)) -> dict:
    ranks = [None if (r is None or r != r) else int(r) for r in ranks]      # NaN -> None
    n = max(1, len(ranks))
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / n
    return out


def pct(values, q) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0
