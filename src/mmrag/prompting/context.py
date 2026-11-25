"""Context assembly + compression (token optimisation, NFR2).

compressed (default): per retrieved chunk keep only the query-relevant part
    text  -> best matching sentences (original order)
    table -> caption + header + matching rows
    image -> caption + matching OCR lines
  each chunk capped at ``context.max_tokens_per_chunk``; whole context capped at
  ``context.max_context_tokens`` (lowest-ranked chunks dropped first).

naive (baseline): the WHOLE page / slide of every retrieved chunk (all text,
  full tables, OCR text) - and, for the token table, raw images are charged at
  ``evaluation.naive_image_tokens`` each as if sent to a vision model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..index import Index
from ..index.bm25 import tokenize
from ..ingest.chunker import split_sentences
from ..schemas import Hit
from ..utils.tokens import count_tokens, truncate_tokens


@dataclass
class ContextItem:
    idx: int                 # citation number [idx]
    chunk_id: str
    source: str
    page: int
    modality: str
    section: str
    doc_type: str
    topic: str
    text: str                # what is sent to the LLM
    full_text: str           # full chunk text (for grounding checks)

    def header(self) -> str:
        return f"[{self.idx}] {self.source} | page {self.page} | {self.modality} | {self.section[:60]}"


def _score(text: str, qt: set[str]) -> float:
    return sum(2.5 if any(c.isdigit() for c in t) or "-" in t else 1.0 for t in qt & set(tokenize(text)))


def compress(hit: Hit, query: str, max_tokens: int) -> str:
    c = hit.chunk
    qt = set(tokenize(query))
    # Tables and figures are short, fact-dense chunks: if the whole chunk fits the
    # budget, send it whole. Trimming them to query-matching rows/lines drops the
    # facts for broad questions ("details of Site-818" keeps only "Site-818").
    if c.modality in ("table", "image") and count_tokens(c.text) <= max_tokens:
        return c.text
    if c.modality == "table":
        lines = c.text.splitlines()
        cap = [l for l in lines if not l.startswith("|")]
        tbl = [l for l in lines if l.startswith("|")]
        if len(tbl) <= 2:
            return truncate_tokens(c.text, max_tokens)
        head, body = tbl[:2], tbl[2:]
        scored = sorted(((_score(r, qt), i) for i, r in enumerate(body)), reverse=True)
        keep = sorted(i for s, i in scored[:3] if s > 0) or list(range(min(4, len(body))))
        out = "\n".join(cap + head + [body[i] for i in keep])
    elif c.modality == "image":
        lines = [l for l in c.text.splitlines() if l.strip()]
        first, rest = lines[0], lines[1:]
        rest = [re.sub(r"^Text in image:\s*", "", l) for l in rest]
        scored = sorted(((_score(l, qt), i) for i, l in enumerate(rest)), reverse=True)
        keep = sorted(i for s, i in scored[:4] if s > 0) or list(range(min(4, len(rest))))
        out = first + ("\nText in image: " + " | ".join(rest[i] for i in keep) if keep else "")
    else:
        sents = split_sentences(c.text)
        scored = sorted(((_score(s, qt), i) for i, s in enumerate(sents)), reverse=True)
        keep = sorted(i for s, i in scored[:3] if s > 0) or list(range(min(2, len(sents))))
        out = " ".join(sents[i] for i in keep)
    return truncate_tokens(out, max_tokens)


def build_context(hits: list[Hit], query: str, cfg, mode: str = "compressed", index: Index | None = None) -> tuple[list[ContextItem], dict]:
    items: list[ContextItem] = []
    stats = {"mode": mode, "image_payload_tokens": 0}
    if mode == "naive":
        # whole pages of the retrieved chunks, deduplicated
        seen = set()
        for h in hits:
            key = (h.chunk.doc_id, h.chunk.page)
            if key in seen or index is None:
                continue
            seen.add(key)
            page_chunks = [c for c in index.chunks.values() if (c.doc_id, c.page) == key]
            page_chunks.sort(key=lambda c: c.chunk_id)
            text = "\n".join(c.text for c in page_chunks)
            stats["image_payload_tokens"] += cfg.get("evaluation.naive_image_tokens", 765) * sum(
                c.modality == "image" for c in page_chunks)
            items.append(ContextItem(len(items) + 1, h.chunk.chunk_id, h.chunk.source, h.chunk.page, "page",
                                     h.chunk.section, h.chunk.doc_type, h.chunk.topic, text, text))
        stats["tokens"] = sum(count_tokens(i.text) for i in items)
        return items, stats
    compress_on = cfg.get("context.compress", True)
    per_chunk = cfg.get("context.max_tokens_per_chunk", 140)
    budget = cfg.get("context.max_context_tokens", 700)
    used = 0
    for h in hits:
        text = compress(h, query, per_chunk) if compress_on else h.chunk.text
        t = count_tokens(text)
        if items and used + t > budget:
            break
        used += t
        c = h.chunk
        items.append(ContextItem(len(items) + 1, c.chunk_id, c.source, c.page, c.modality, c.section, c.doc_type,
                                 c.topic, text, c.text))
    stats["tokens"] = used
    return items, stats
