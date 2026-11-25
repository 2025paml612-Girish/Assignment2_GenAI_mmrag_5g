"""Hybrid, metadata-filtered, modality-aware retriever (Task A2).

Pipeline for one query
  1. Route (ontology): entities, topics, doc_type, preferred modality
  2. Metadata pre-filter: entity -> documents that mention it; else topic filter
  3. Dense search (vector store, cosine) and sparse search (BM25) on the same filter
  4. Reciprocal-rank fusion (RRF) of both rankings
  5. Modality routing boost (+ memory boost for documents used in the previous turn)
  6. Optional cross-encoder re-ranking of the fused top candidates
  7. Caption collapse: a text chunk that is only the caption of a table / figure
     on the same page is replaced by that table / figure chunk (the caption alone
     carries no facts but, being short, outranks the real table/figure in BM25)
  8. Modality guarantee: if the query asks for a table / image and none is in the
     top-k, the best chunk of that modality replaces the last slot
Results are sorted by (-score, chunk_id) so the output is deterministic (NFR1).

All modalities live in one index: tables are Markdown text and images are
caption + OCR text, so they share one embedding space; fusion therefore works
on a single ranked list and modality is handled through metadata.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..index import Index
from ..schemas import Hit
from ..utils.log import get_logger
from .router import QueryRouter, Route

log = get_logger("retriever")


class Retriever:
    def __init__(self, index: Index, cfg, router: QueryRouter | None = None):
        self.index = index
        self.cfg = cfg
        self.router = router or QueryRouter(cfg.ontology)
        self.reranker = None
        mode = str(cfg.get("retrieval.rerank", "auto")).lower()
        if mode in ("auto", "true"):
            try:
                if index.embedder.backend != "sentence-transformers" and mode == "auto":
                    raise RuntimeError("no neural backend")
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(cfg.get("retrieval.rerank_model"), device="cpu")
                log.info("Cross-encoder re-ranker enabled")
            except Exception as e:
                log.info("Re-ranker disabled (%s)", type(e).__name__)
        self._doc_text: dict[str, str] = {}
        self._doc_topics: dict[str, set[str]] = {}
        for c in index.chunks.values():
            self._doc_text[c.doc_id] = self._doc_text.get(c.doc_id, c.title.lower()) + " " + c.text.lower()
            self._doc_topics.setdefault(c.doc_id, set()).update(t for t in c.topics.split(",") if t)
        # (doc_id, page) -> [(caption, chunk_id)] of table / image chunks, for caption collapse
        self._rich: dict[tuple, list[tuple[str, str]]] = {}
        for c in index.chunks.values():
            if c.modality in ("table", "image") and c.text.strip():
                cap = re.sub(r"^\[Figure\]\s*", "", c.text.strip().splitlines()[0]).strip().lower()
                if len(cap) >= 8 and not cap.startswith("|"):
                    self._rich.setdefault((c.doc_id, c.page), []).append((cap, c.chunk_id))

    def _collapse_captions(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Swap caption-only text chunks for the table / figure chunk they caption."""
        out: dict[str, float] = {}
        for cid, sc in ranked:
            c = self.index.chunks[cid]
            target = cid
            if c.modality == "text":
                txt = c.text.strip().lower()
                for cap, rid in self._rich.get((c.doc_id, c.page), []):
                    if cap in txt and len(txt) <= len(cap) + 40:      # nothing beyond the caption (+ a heading)
                        target = rid
                        break
            if target not in out:
                out[target] = sc                                      # keep the best score seen
        return sorted(out.items(), key=lambda t: (-t[1], t[0]))

    # ------------------------------------------------------------------
    @lru_cache(maxsize=2048)
    def _qvec(self, text: str):
        return self.index.embedder.encode([text], is_query=True)[0]

    def entity_docs(self, entities: list[str]) -> set[str]:
        docs: set[str] | None = None
        for e in entities:
            pat = re.compile(r"(?<![\w-])" + re.escape(e.lower()).replace(r"\-", "[- ]?").replace(r"\ ", "[- ]?") + r"(?![\w])")
            hit = {d for d, t in self._doc_text.items() if pat.search(t)}
            if hit:
                docs = hit if docs is None else (docs & hit or docs)
        return docs or set()

    def search(self, query: str, route: Route | None = None, k: int | None = None, *, modalities: list[str] | None = None,
               use_filters: bool = True, use_sparse: bool | None = None, use_modality_routing: bool | None = None,
               boost_docs: list[str] | None = None) -> tuple[list[Hit], dict]:
        cfg = self.cfg
        k = k or cfg.get("retrieval.top_k", 4)
        ck = cfg.get("retrieval.candidate_k", 30)
        route = route or self.router.route(query)
        use_sparse = cfg.get("retrieval.hybrid", True) if use_sparse is None else use_sparse
        use_mod = cfg.get("retrieval.modality_routing", True) if use_modality_routing is None else use_modality_routing
        info = {"route": route.as_dict(), "filters": []}

        where: dict = {}
        if modalities:
            where["modality"] = list(modalities)
        if use_filters and cfg.get("retrieval.ontology_filter", True):
            docs = self.entity_docs(route.entities) if route.entities else set()
            if docs:
                where["doc_id"] = sorted(docs)
                info["filters"].append(f"entity->{len(docs)} docs")
            elif route.topics:
                tdocs = sorted(d for d, ts in self._doc_topics.items() if ts & set(route.topics))
                if tdocs:
                    where["doc_id"] = tdocs
                    where["_topic"] = True
                    info["filters"].append(f"topic={route.topics}->{len(tdocs)} docs")

        def run(wh: dict):
            dense = self.index.store.query(self._qvec(query), ck, wh or None)
            sparse = []
            if use_sparse:
                allowed = None
                if wh:
                    allowed = {cid for cid, c in self.index.chunks.items()
                               if all(getattr(c, f) in set(v) for f, v in wh.items())}
                sparse = self.index.bm25.query(query, ck, allowed)
            return dense, sparse

        topic_filter = where.pop("_topic", False)
        dense, sparse = run(where)
        if len(dense) < k and topic_filter:                # topic filter too narrow -> relax
            where.pop("doc_id")
            info["filters"].append("topic-relaxed")
            dense, sparse = run(where)
        info["max_dense"] = max((s for _, s in dense), default=0.0)

        # ---- reciprocal rank fusion ------------------------------------
        rrf_k = cfg.get("retrieval.rrf_k", 60)
        wd, ws = cfg.get("retrieval.dense_weight", 1.0), cfg.get("retrieval.sparse_weight", 1.0)
        fused: dict[str, float] = {}
        dmap, smap = dict(dense), dict(sparse)
        for r, (cid, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + wd / (rrf_k + r + 1)
        for r, (cid, _) in enumerate(sparse):
            fused[cid] = fused.get(cid, 0.0) + ws / (rrf_k + r + 1)
        if not fused:
            return [], info
        top = max(fused.values())
        scores = {cid: v / top for cid, v in fused.items()}          # normalise to [0, 1]
        boost = cfg.get("retrieval.modality_boost", 0.35)
        if use_mod and route.modality:
            for cid in scores:
                if self.index.chunks[cid].modality == route.modality:
                    scores[cid] += boost
        if boost_docs:
            for cid in scores:
                if self.index.chunks[cid].doc_id in boost_docs:
                    scores[cid] += 0.15
        ranked = sorted(scores.items(), key=lambda t: (-t[1], t[0]))

        # ---- optional cross-encoder re-rank ------------------------------
        if self.reranker is not None:
            cand = ranked[:max(12, 3 * k)]
            ce = self.reranker.predict([(query, self.index.chunks[c].text) for c, _ in cand])
            lo, hi = float(min(ce)), float(max(ce))
            ranked = sorted(((c, (float(s) - lo) / (hi - lo + 1e-9) + 0.5 * sc) for (c, sc), s in zip(cand, ce)),
                            key=lambda t: (-t[1], t[0]))
            info["reranked"] = True

        # ---- caption collapse ---------------------------------------------
        if cfg.get("retrieval.collapse_captions", True):
            ranked = self._collapse_captions(ranked)

        hits = [Hit(self.index.chunks[c], s, dmap.get(c, 0.0), smap.get(c, 0.0)) for c, s in ranked[:k]]
        # ---- modality guarantee -----------------------------------------
        if use_mod and route.modality and hits and not any(h.chunk.modality == route.modality for h in hits):
            for c, s in ranked[k:]:
                if self.index.chunks[c].modality == route.modality:
                    hits[-1] = Hit(self.index.chunks[c], s, dmap.get(c, 0.0), smap.get(c, 0.0))
                    info["modality_swap"] = True
                    break
        return hits, info
