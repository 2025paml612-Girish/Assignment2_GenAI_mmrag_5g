"""Chunking strategies.

* recursive (default): text of a page is merged per section, then split on
  paragraph -> sentence boundaries into ~``chunk_tokens`` pieces with
  ``overlap_tokens`` overlap.
* page: one text chunk per page (the coarse baseline compared in E2).

Tables become their own chunks (header repeated when a big table is split).
Images become their own chunks (caption + OCR text). Modality is therefore an
explicit, filterable metadata field.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..schemas import Chunk, Element
from ..utils.tokens import count_tokens
from .metadata import OntologyTagger

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n{1,}", text):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return out


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur, cur_tok = [], [], 0
    for s in sentences:
        t = count_tokens(s)
        if cur and cur_tok + t > size:
            chunks.append(" ".join(cur))
            # overlap: carry trailing sentences
            keep, kt = [], 0
            for prev in reversed(cur):
                pt = count_tokens(prev)
                if kt + pt > overlap:
                    break
                keep.insert(0, prev)
                kt += pt
            cur, cur_tok = keep, kt
        cur.append(s)
        cur_tok += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_table(md: str, rows_per_chunk: int) -> list[str]:
    lines = md.splitlines()
    cap = [l for l in lines if not l.startswith("|")]
    tbl = [l for l in lines if l.startswith("|")]
    if len(tbl) <= rows_per_chunk + 2:
        return [md]
    head, body = tbl[:2], tbl[2:]
    out = []
    for i in range(0, len(body), rows_per_chunk):
        out.append("\n".join(cap + head + body[i:i + rows_per_chunk]))
    return out


def chunk_document(elements: list[Element], doc_meta: dict, tagger: OntologyTagger, strategy: str = "recursive",
                   chunk_tokens: int = 220, overlap_tokens: int = 40, rows_per_chunk: int = 12) -> list[Chunk]:
    chunks: list[Chunk] = []
    doc_id = doc_meta["doc_id"]

    def make(page: int, modality: str, section: str, text: str) -> Chunk:
        topic, topics = tagger.topics_for(f"{doc_meta['title']} {section} {text}")
        cid = f"{doc_id}-p{page}-{ {'text': 'x', 'table': 't', 'image': 'i'}[modality]}{sum(1 for c in chunks if c.page == page and c.modality == modality)}"
        return Chunk(chunk_id=cid, doc_id=doc_id, source=doc_meta["source"], doc_type=doc_meta["doc_type"],
                     title=doc_meta["title"], page=page, modality=modality, topic=topic, topics=topics,
                     section=section[:120], date=doc_meta.get("date", ""), sensitivity="public",
                     text=text, tokens=count_tokens(text))

    # text --------------------------------------------------------------
    by_page: dict[int, list[Element]] = defaultdict(list)
    for e in elements:
        if e.modality == "text":
            by_page[e.page].append(e)
    for page in sorted(by_page):
        els = by_page[page]
        if strategy == "page":
            txt = "\n".join(e.content for e in els)
            if txt.strip():
                chunks.append(make(page, "text", els[0].section, txt))
            continue
        # group consecutive elements of the same section
        groups: list[tuple[str, list[str]]] = []
        for e in els:
            if groups and groups[-1][0] == e.section:
                groups[-1][1].append(e.content)
            else:
                groups.append((e.section, [e.content]))
        # merge tiny groups (lone headings, captions) into their neighbour
        merged: list[tuple[str, list[str]]] = []
        for section, texts in groups:
            if merged and count_tokens(" ".join(merged[-1][1])) < 25:
                merged[-1] = (section or merged[-1][0], merged[-1][1] + texts)
            else:
                merged.append((section, texts))
        if len(merged) > 1 and count_tokens(" ".join(merged[-1][1])) < 25:
            last = merged.pop()
            merged[-1] = (merged[-1][0], merged[-1][1] + last[1])
        for section, texts in merged:
            for piece in _pack(split_sentences("\n".join(texts)), chunk_tokens, overlap_tokens):
                if count_tokens(piece) >= 4:
                    chunks.append(make(page, "text", section, piece))
    # tables & images -----------------------------------------------------
    for e in elements:
        if e.modality == "table":
            for part in _split_table(e.content, rows_per_chunk):
                chunks.append(make(e.page, "table", e.section, part))
        elif e.modality == "image" and len(e.content) > 12:
            chunks.append(make(e.page, "image", e.section, e.content))
    return chunks
