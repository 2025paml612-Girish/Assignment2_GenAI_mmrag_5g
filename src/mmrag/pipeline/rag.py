"""End-to-end query path (Section 3.6 minimum workflow).

query -> DoS check -> input guardrail -> memory rewrite -> answer cache
      -> route (ontology) -> hybrid multimodal retrieval -> out-of-scope check
      -> context assembly/compression -> prompt (template + ontology) -> LLM
      -> output guardrail (grounded + cited, refusal) -> memory update -> Answer
"""
from __future__ import annotations

import re
import time

from ..config import Config
from ..guardrails import DoSGuard, InputGuard, OutputGuard
from ..guardrails.output_guard import CITE
from ..index import Index
from ..llm import make_llm
from ..memory import AnswerCache, ConversationMemory, Turn
from ..prompting.context import build_context
from ..prompting.variants import PromptBuilder
from ..retrieval.retriever import Retriever
from ..retrieval.router import QueryRouter
from ..schemas import Answer, Hit
from ..utils.log import get_logger

log = get_logger("rag")


class RAGSystem:
    def __init__(self, cfg: Config, index: Index | None = None, llm=None):
        self.cfg = cfg
        self.index = index or Index.load(cfg.path("index_dir"), cfg.get("vector_store.backend"),
                                         cfg.get("vector_store.collection"))
        self.router = QueryRouter(cfg.ontology)
        self.retriever = Retriever(self.index, cfg, self.router)
        self.llm = llm or make_llm(cfg)
        self.prompts = PromptBuilder(cfg)
        d = cfg.get("guardrails.dos", {})
        self.dos = DoSGuard(d.get("max_query_chars", 1000), d.get("max_requests_per_minute", 30),
                            d.get("max_concurrent", 4), d.get("burst_repeat_limit", 5))
        self.inguard = InputGuard(cfg)
        self.outguard = OutputGuard(cfg)
        self.memory = ConversationMemory(cfg.get("memory.short_term_turns", 4))
        self.cache = AnswerCache(cfg.path("cache_db")) if cfg.get("memory.answer_cache", True) else None

    # ------------------------------------------------------------------
    def _refuse(self, query, rewritten, variant, reason, t0, timings, hits=None) -> Answer:
        timings["total"] = time.perf_counter() - t0
        return Answer(query, rewritten, self.cfg.get("guardrails.output.refusal_text") if not reason.startswith("dos")
                      else "Request rejected: " + reason, [], hits or [], variant, True, reason, 0.0, 0, 0,
                      timings["total"], timings)

    def answer(self, query: str, session: str = "default", variant: str | None = None, *, use_guardrails: bool | None = None,
               use_cache: bool | None = None, use_memory: bool = True, context_mode: str = "compressed",
               k: int | None = None, instruction: str | None = None, demos: list[dict] | None = None,
               retrieval_kwargs: dict | None = None) -> Answer:
        cfg = self.cfg
        variant = variant or cfg.get("prompting.default_variant", "jinja2")
        guards = cfg.get("guardrails.enabled", True) if use_guardrails is None else use_guardrails
        use_cache = (self.cache is not None) if use_cache is None else (use_cache and self.cache is not None)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. DoS + input guardrails ---------------------------------------
        if guards:
            t = time.perf_counter()
            ok, reason = self.dos.check(session, query)
            timings["guard_dos"] = time.perf_counter() - t
            if not ok:
                return self._refuse(query, query, variant, reason, t0, timings)
            t = time.perf_counter()
            ok, reason, query_clean = self.inguard.check(query)
            timings["guard_input"] = time.perf_counter() - t
            if not ok:
                return self._refuse(query, query, variant, reason, t0, timings)
        else:
            query_clean = query

        # 2. memory: follow-up rewrite --------------------------------------
        t = time.perf_counter()
        route0 = self.router.route(query_clean)
        rewritten, boost_docs, followup = (query_clean, [], False)
        if use_memory:
            llm_rw = self.llm if cfg.get("memory.rewrite") == "llm" else None
            rewritten, boost_docs, followup = self.memory.rewrite(session, query_clean, route0.entities, llm_rw)
        route = self.router.route(rewritten) if followup else route0
        timings["memory_rewrite"] = time.perf_counter() - t

        # 3. long-term cache ------------------------------------------------
        ckey = None
        if use_cache:
            ckey = AnswerCache.key(rewritten, f"{variant}:{context_mode}:{instruction or ''}", self.index.version,
                                   self.llm.name)
            hit = self.cache.get(ckey)
            if hit:
                hits = [Hit(self.index.chunks[c], 0.0) for c in hit["retrieved"] if c in self.index.chunks]
                timings["total"] = time.perf_counter() - t0
                a = Answer(query, rewritten, hit["answer"], hit["citations"], hits, variant, hit["refused"],
                           hit["refusal_reason"], hit["grounding"], 0, 0, timings["total"], timings, from_cache=True)
                self._remember(session, query, rewritten, a, route)
                return a

        # 4. retrieval -------------------------------------------------------
        t = time.perf_counter()
        hits, rinfo = self.retriever.search(rewritten, route, k, boost_docs=boost_docs, **(retrieval_kwargs or {}))
        timings["retrieval"] = time.perf_counter() - t
        if guards:
            t = time.perf_counter()
            max_sparse = max((h.sparse for h in hits), default=0.0)
            oos = self.inguard.out_of_scope(route.domain_hits, rinfo.get("max_dense", 0.0), max_sparse)
            timings["guard_oos"] = time.perf_counter() - t
            if oos:
                return self._refuse(query, rewritten, variant, "input: out of scope for the corpus", t0, timings, hits)
        if not hits:
            return self._refuse(query, rewritten, variant, "retrieval: no evidence", t0, timings)

        # 5. context + prompt -------------------------------------------------
        t = time.perf_counter()
        contexts, cstats = build_context(hits, rewritten, cfg, context_mode, self.index)
        history = [{"query": h.rewritten, "answer": h.answer} for h in self.memory.history(session)[-2:]] if followup else []
        messages = self.prompts.build(variant, rewritten, contexts, route, history, instruction, demos)
        timings["context_prompt"] = time.perf_counter() - t

        # 6. LLM (ReAct runs a bounded tool loop) ---------------------------------
        t = time.perf_counter()
        in_tok = out_tok = 0
        max_tok, temp = cfg.get("llm.max_tokens", 220), cfg.get("llm.temperature", 0.0)
        ctx_meta = [(c.idx, c.text) for c in contexts]
        if variant == "react":
            text = ""
            for step in range(3):
                r = self.llm.generate(messages, max_tok, temp, stop=["Observation:"], question=rewritten, contexts=ctx_meta)
                in_tok += r.input_tokens
                out_tok += r.output_tokens
                fa = re.search(r"Final Answer:\s*(.*)", r.text, re.S)
                act = re.search(r"Action:\s*search\[(.*?)\]", r.text)
                if fa:
                    text = fa.group(1).strip()
                    break
                if act and step < 2:
                    more, _ = self.retriever.search(act.group(1), None, 2)
                    new = [h for h in more if h.chunk.chunk_id not in {c.chunk_id for c in contexts}]
                    extra, _ = build_context(new, act.group(1), cfg, "compressed")
                    for e in extra:
                        e.idx = len(contexts) + 1
                        contexts.append(e)
                    ctx_meta = [(c.idx, c.text) for c in contexts]
                    obs = "\n".join(f"{c.header()}\n{c.text}" for c in extra) or "(no new results)"
                    messages = messages + [{"role": "assistant", "content": r.text},
                                           {"role": "user", "content": f"Observation:\n{obs}"}]
                    continue
                text = re.sub(r"^\s*Thought:.*?\n", "", r.text, flags=re.S).strip()
                break
        else:
            try:
                r = self.llm.generate(messages, max_tok, temp, question=rewritten, contexts=ctx_meta)
            except Exception as e:
                log.error("LLM call failed: %s", e)
                return self._refuse(query, rewritten, variant, f"llm: {type(e).__name__}", t0, timings, hits)
            text, in_tok, out_tok = r.text, r.input_tokens, r.output_tokens
        timings["llm"] = time.perf_counter() - t
        if context_mode == "naive":
            in_tok += cstats.get("image_payload_tokens", 0)

        # 7. output guardrail ---------------------------------------------------
        t = time.perf_counter()
        if guards:
            g = self.outguard.check(text, contexts)
        else:
            by = {c.idx: c for c in contexts}
            cited = sorted({int(n) for n in CITE.findall(text) if int(n) in by})
            g = {"answer": text, "refused": False, "reason": "", "grounding": 0.0,
                 "citations": [{"n": i, "chunk_id": by[i].chunk_id, "source": by[i].source, "page": by[i].page,
                                "modality": by[i].modality} for i in cited]}
        timings["guard_output"] = time.perf_counter() - t
        timings["total"] = time.perf_counter() - t0
        a = Answer(query, rewritten, g["answer"], g["citations"], hits, variant, g["refused"], g["reason"],
                   g["grounding"], in_tok, out_tok, timings["total"], timings)
        a.timings["context_tokens"] = float(cstats.get("tokens", 0))
        if use_cache and ckey and not a.refused:
            self.cache.put(ckey, a.to_dict())
        self._remember(session, query, rewritten, a, route)
        return a

    def _remember(self, session, query, rewritten, a: Answer, route):
        ents = route.entities or (self.memory.history(session)[-1].entities if self.memory.history(session) else [])
        docs = list(dict.fromkeys(self.index.chunks[c["chunk_id"]].doc_id for c in a.citations
                                  if c.get("chunk_id") in self.index.chunks))
        self.memory.add(session, Turn(query, rewritten, a.answer, ents, docs))
