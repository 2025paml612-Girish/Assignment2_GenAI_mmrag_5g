"""Sparse lexical index (BM25Okapi) - catches exact identifiers such as
"n78", "PCI 412", "O-DU", "Site-17" that dense embeddings often blur."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9]+(?:[\-\.][a-z0-9]+)*")
STOP = set("a an the of to in on for and or is are was were be by with at as from what which who how does do "
           "this that it its into than then there their these those can could would should".split())


def stem(t: str) -> str:
    """very light suffix stripping for plain words (identifiers are untouched)."""
    if not t.isalpha() or len(t) <= 4:
        return t
    for suf in ("ing", "ed", "es"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    if t.endswith("s") and not t.endswith(("ss", "is", "us")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    out = []
    for t in toks:
        if t in STOP:
            continue
        t = stem(t)
        out.append(t)
        if "-" in t:                     # also index the parts of "o-du" / "site-17"
            out.extend(p for p in t.split("-") if p and p not in STOP)
    return out


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])

    def query(self, text: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = tokenize(text)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        if allowed is not None:
            mask = np.array([i in allowed for i in self.ids])
            scores = np.where(mask, scores, -np.inf)
        order = np.lexsort((np.array(self.ids), -scores))[:k]
        return [(self.ids[i], float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)
