"""Experiments E1-E10 (Section 3.7). Every experiment writes CSV/JSON/Markdown to
``outputs/`` and ``build_report_tables`` collects them into REPORT_TABLES.md."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..index import Index, build_index
from ..ingest.pipeline import make_chunks, parse_corpus, run_ingest
from ..pipeline.rag import RAGSystem
from ..prompting.optimizer import PromptOptimizer, split_dev_test
from ..prompting.variants import PromptBuilder
from ..prompting.context import build_context
from ..retrieval.retriever import Retriever
from ..utils.log import get_logger
from ..utils.tokens import tokenizer_name
from .auto_eval import agreement, generate_synthetic, judge_answers
from .failure import failure_table
from .metrics import answer_correct, first_relevant_rank, pct, retrieval_summary
from .runner import by_modality, load_jsonl, run_questions, summarise

log = get_logger("experiments")
ABLATION_VARIANTS = ["zero_shot", "fstring", "three_shot", "jinja2", "create", "react"]


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _balanced_subset(golden: list[dict], n: int) -> list[dict]:
    per = max(1, n // 3)
    out = []
    for m in ("text", "table", "image"):
        out += [g for g in golden if g["modality"] == m][:per]
    return out


class Experiments:
    def __init__(self, cfg: Config, quick: bool = False):
        self.cfg = cfg
        self.quick = quick
        self.out = cfg.path("outputs_dir")
        self.out.mkdir(parents=True, exist_ok=True)
        self.golden = load_jsonl(cfg.path("golden_set"))
        self._rag: RAGSystem | None = None
        self.ks = tuple(cfg.get("evaluation.recall_k", [1, 3, 5]))

    @property
    def rag(self) -> RAGSystem:
        if self._rag is None:
            self._rag = RAGSystem(self.cfg)
        return self._rag

    def _save(self, name: str, obj) -> None:
        p = self.out / name
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(p, index=False)
        else:
            p.write_text(json.dumps(obj, indent=2, default=str))

    def _subset(self, n_full: int | None = None) -> list[dict]:
        return _balanced_subset(self.golden, 12) if self.quick else (self.golden if n_full is None else
                                                                       _balanced_subset(self.golden, n_full))

    # ------------------------------------------------------------------ E1
    def e1(self, force: bool = False) -> dict:
        idx_dir = self.cfg.path("index_dir")
        if force or not (idx_dir / "manifest.json").exists():
            _, stats = run_ingest(self.cfg, force=force)
        else:
            stats = json.loads((self.out / "e1_ingestion_stats.json").read_text())
        return stats

    # ------------------------------------------------------------------ retrieval-only helper
    def _retrieval_eval(self, retriever: Retriever, qs: list[dict], **kw) -> tuple[dict, pd.DataFrame]:
        rows = []
        k = max(self.ks)
        for q in qs:
            hits, _ = retriever.search(q["question"], k=k, **kw)
            rows.append({"qid": q["qid"], "modality": q["modality"], "rank": first_relevant_rank(hits, q["doc_id"], q["page"], q["modality"])})
        df = pd.DataFrame(rows)
        s = retrieval_summary(list(df["rank"].where(df["rank"].notna(), None)), self.ks)
        for m, g in df.groupby("modality"):
            s[f"recall@{k}_{m}"] = retrieval_summary(list(g["rank"].where(g["rank"].notna(), None)), (k,))[f"recall@{k}"]
        return {kk: round(v, 4) for kk, v in s.items()}, df

    # ------------------------------------------------------------------ E2
    def e2(self) -> pd.DataFrame:
        cfg = self.cfg
        by_doc, records = parse_corpus(cfg)
        from ..index.embedder import Embedder

        probe = Embedder(cfg.get("embedding.backend"), cfg.get("embedding.model"))
        if probe.backend == "sentence-transformers":
            configs = [("A", cfg.get("embedding.model"), "recursive", 256), ("B", cfg.get("embedding.model"), "page", 256),
                       ("C", cfg.get("embedding.alt_model"), "recursive", 256)]
            backend = "sentence-transformers"
        else:
            configs = [("A", "tfidf-lsa-256", "recursive", 256), ("B", "tfidf-lsa-256", "page", 256),
                       ("C", "tfidf-lsa-96", "recursive", 96)]
            backend = "tfidf"
        rows = []
        for name, model, strategy, dims in configs:
            t0 = time.time()
            chunks = make_chunks(cfg, by_doc, records, strategy)
            idx = build_index(chunks, self.out / "_tmp_e2", backend, model, "numpy", persist=False, dims=dims)
            r = Retriever(idx, cfg)
            for mode, kw in [("dense", {"use_sparse": False, "use_filters": False, "use_modality_routing": False}),
                             ("hybrid+filters", {})]:
                s, _ = self._retrieval_eval(r, self.golden, **kw)
                rows.append({"config": name, "embedding": idx.embedder.name, "chunking": strategy, "retrieval": mode,
                             "chunks": len(chunks), "avg_chunk_tokens": round(np.mean([c.tokens for c in chunks]), 1),
                             "build_s": round(time.time() - t0, 2), **s})
        df = pd.DataFrame(rows)
        self._save("e2_embedding_chunking.csv", df)
        shutil.rmtree(self.out / "_tmp_e2", ignore_errors=True)
        return df

    # ------------------------------------------------------------------ E3
    def e3(self) -> dict:
        r = self.rag.retriever
        setups = [
            ("text-only, dense (naive baseline)", {"modalities": ["text"], "use_filters": False, "use_sparse": False,
                                                   "use_modality_routing": False}),
            ("text-only, hybrid", {"modalities": ["text"], "use_filters": False, "use_modality_routing": False}),
            ("multimodal, dense", {"use_filters": False, "use_sparse": False, "use_modality_routing": False}),
            ("multimodal, hybrid", {"use_filters": False, "use_modality_routing": False}),
            ("multimodal, hybrid + modality routing", {"use_filters": False}),
            ("multimodal, hybrid + routing + metadata filter (final)", {}),
        ]
        rows = []
        for i, (name, kw) in enumerate(setups):
            s, _ = self._retrieval_eval(r, self.golden, **kw)
            row = {"setup": name, **s}
            if i in (0, len(setups) - 1):            # end-to-end answer accuracy for baseline vs final
                d = run_questions(self.rag, self._subset(), tag=f"e3-{i}", retrieval_kwargs=kw)
                row["answer_accuracy"] = round(float(d.correct.mean()), 4)
                for m, g in d.groupby("modality"):
                    row[f"answer_acc_{m}"] = round(float(g.correct.mean()), 4)
            rows.append(row)
        df = pd.DataFrame(rows)
        self._save("e3_retrieval_design.csv", df)
        # FR2 worked examples: one per modality
        ex = []
        for m in ("text", "table", "image"):
            for q in [g for g in self.golden if g["modality"] == m]:
                a = self.rag.answer(q["question"], session=f"e3-{q['qid']}", use_cache=False, use_memory=False)
                if answer_correct(a.answer, q["key"]) or q is [g for g in self.golden if g["modality"] == m][-1]:
                    ex.append({"modality": m, "question": q["question"], "answer": a.answer,
                               "citations": a.citations, "gold_source": f"{q['source']} p.{q['page']}"})
                    break
        self._save("e3_modality_examples.json", ex)
        return {"table": df, "examples": ex}

    # ------------------------------------------------------------------ E4
    def e4(self) -> dict:
        p = self.cfg.path("golden_set").parent / "multiturn_dialog.json"
        dialog = json.loads(p.read_text()) if p.exists() else []
        res = {}
        for mode, use_mem in (("with_memory", True), ("without_memory", False)):
            sess = f"e4-{mode}-{time.time()}"
            turns = []
            for t in dialog:
                a = self.rag.answer(t["query"], session=sess, use_cache=False, use_memory=use_mem)
                turns.append({"turn": t["turn"], "user": t["query"], "standalone_query": a.rewritten_query,
                              "answer": a.answer, "citations": [f"{c['source']} p.{c['page']} ({c['modality']})"
                                                                for c in a.citations],
                              "correct": int(answer_correct(a.answer, t["expect"]) and not a.refused)})
            res[mode] = turns
        lines = ["# E4 - Multi-turn conversation with short-term memory", "",
                 "Memory design: last N turns per session (query, standalone query, answer, entities, cited docs). "
                 "Follow-ups are detected (anaphora / leading 'and' / no own entity) and rewritten by carrying over "
                 "the previous entities; documents cited in the previous turn get a retrieval boost. A SQLite answer "
                 "cache (long-term memory) returns repeated questions with zero LLM tokens.", ""]
        for mode, turns in res.items():
            lines += [f"## {mode.replace('_', ' ')}", ""]
            for t in turns:
                lines += [f"**Turn {t['turn']} - user:** {t['user']}", f"- standalone query: `{t['standalone_query']}`",
                          f"- answer: {t['answer']}", f"- sources: {', '.join(t['citations']) or '-'}",
                          f"- correct: {'yes' if t['correct'] else 'no'}", ""]
        (self.out / "e4_memory_transcript.md").write_text("\n".join(lines), encoding="utf-8")
        self._save("e4_memory.json", res)
        return res

    # ------------------------------------------------------------------ E5
    def e5(self) -> dict:
        df = run_questions(self.rag, self.golden, tag="e5")
        self._save("e5_golden_results.csv", df)
        summ = summarise(df, self.ks)
        bym = by_modality(df, self.ks)
        self._save("e5_golden_by_modality.csv", bym)
        # automatic approach 1: synthetic questions from the corpus
        synth = generate_synthetic(self.rag.index, {g["doc_id"] for g in self.golden},
                                   n=12 if self.quick else 30, llm=self.rag.llm)
        (self.out / "e5_synthetic_questions.jsonl").write_text("\n".join(json.dumps(s) for s in synth), encoding="utf-8")
        sdf = run_questions(self.rag, synth, tag="e5s")
        self._save("e5_synthetic_results.csv", sdf)
        ssum = summarise(sdf, self.ks)
        # automatic approach 2: reference-free judge vs golden labels
        jdf = judge_answers(self.rag, self.golden)
        self._save("e5_judge_vs_golden.csv", jdf)
        agr = agreement(jdf)
        comp = pd.DataFrame([
            {"method": "Golden set (hand labels)", "n": summ["n"], "recall@5": summ.get("recall@5"), "mrr": summ["mrr"],
             "answer_accuracy": summ["answer_accuracy"], "faithfulness": summ["faithfulness"]},
            {"method": f"Synthetic QG ({synth[0]['generator'] if synth else '-'})", "n": ssum["n"],
             "recall@5": ssum.get("recall@5"), "mrr": ssum["mrr"], "answer_accuracy": ssum["answer_accuracy"],
             "faithfulness": ssum["faithfulness"]},
            {"method": f"Reference-free judge ({jdf.judge.iloc[0] if len(jdf) else '-'}) on golden questions",
             "n": agr["n"], "recall@5": None, "mrr": None, "answer_accuracy": agr["judge_pass_rate"], "faithfulness": None},
        ])
        self._save("e5_auto_vs_golden.csv", comp)
        self._save("e5_judge_agreement.json", agr)
        # failure analysis (>= 3 cases)
        ft = failure_table(df, self.rag.index, self.golden)
        if len(ft) < 3 and len(sdf):
            extra = sdf[sdf.correct == 0].head(3 - len(ft))
            for _, r in extra.iterrows():
                ft = pd.concat([ft, pd.DataFrame([{"qid": r.qid, "modality": r.modality, "question": r.question,
                                                   "gold": r.gold, "system_answer": r.answer, "gold_source": "synthetic",
                                                   "gold_rank": r["rank"], "cause": "synthetic-set miss",
                                                   "proposed_fix": "see cause column of golden cases"}])])
        self._save("e5_failure_analysis.csv", ft)
        md = ["# E5 - Failure analysis", ""]
        for _, r in ft.iterrows():
            md += [f"## {r.qid} ({r.modality}) - cause: `{r.cause}`", f"- question: {r.question}",
                   f"- gold: {r.gold} ({r.gold_source}, gold rank {r.gold_rank})", f"- system: {r.system_answer}",
                   f"- proposed fix: {r.proposed_fix}", ""]
        (self.out / "e5_failure_analysis.md").write_text("\n".join(md), encoding="utf-8")
        res = {"golden": summ, "synthetic": ssum, "judge_agreement": agr, "n_failures": int(len(ft))}
        self._save("e5_summary.json", res)
        return res

    # ------------------------------------------------------------------ E6
    def e6(self) -> dict:
        rag = self.rag
        qs = self.golden
        res: dict = {}
        # latency over all golden queries (full pipeline, guards on)
        g = pd.read_csv(self.out / "e5_golden_results.csv") if (self.out / "e5_golden_results.csv").exists() \
            else run_questions(rag, qs, tag="e6")
        res["latency"] = {"n_queries": int(len(g)), "p50_s": round(pct(g.latency_s, 50), 4),
                          "p95_s": round(pct(g.latency_s, 95), 4), "max_s": round(float(g.latency_s.max()), 4),
                          "fr3_p95_under_90s": bool(pct(g.latency_s, 95) < 90)}
        # consistency (NFR1): >=5 runs per query, cache and memory off
        runs = self.cfg.get("evaluation.consistency_runs", 5)
        cq = self._subset()
        crow = []
        for q in cq:
            rets, ans = [], []
            for i in range(runs):
                a = rag.answer(q["question"], session=f"cons-{q['qid']}-{i}", use_cache=False, use_memory=False)
                rets.append(tuple(h.chunk.chunk_id for h in a.hits))
                ans.append(a.answer.strip())
            crow.append({"qid": q["qid"], "modality": q["modality"], "runs": runs,
                         "identical_retrieval": int(len(set(rets)) == 1), "identical_answer": int(len(set(ans)) == 1),
                         "distinct_answers": len(set(ans))})
        cdf = pd.DataFrame(crow)
        self._save("e6_consistency.csv", cdf)
        res["consistency"] = {"queries": int(len(cdf)), "runs_per_query": runs,
                              "identical_retrieval_share": round(float(cdf.identical_retrieval.mean()), 4),
                              "identical_answer_share": round(float(cdf.identical_answer.mean()), 4)}
        # DoS + guardrail overhead (FR4)
        orow = []
        for q in cq:
            a_on = rag.answer(q["question"], session=f"ov-on-{q['qid']}", use_cache=False, use_memory=False,
                              use_guardrails=True)
            a_off = rag.answer(q["question"], session=f"ov-off-{q['qid']}", use_cache=False, use_memory=False,
                               use_guardrails=False)
            guard_t = sum(v for k, v in a_on.timings.items() if k.startswith("guard_"))
            orow.append({"qid": q["qid"], "latency_with_guards_s": a_on.latency_s, "latency_without_guards_s": a_off.latency_s,
                         "guard_time_s": guard_t, "difference_s": a_on.latency_s - a_off.latency_s})
        odf = pd.DataFrame(orow)
        self._save("e6_guardrail_overhead.csv", odf)
        res["guardrail_overhead"] = {"mean_guard_time_ms": round(1000 * float(odf.guard_time_s.mean()), 4),
                                     "p95_guard_time_ms": round(1000 * pct(odf.guard_time_s, 95), 4),
                                     "mean_latency_with_s": round(float(odf.latency_with_guards_s.mean()), 4),
                                     "mean_latency_without_s": round(float(odf.latency_without_guards_s.mean()), 4),
                                     "fr4_under_6s": bool(pct(odf.guard_time_s, 95) < 6)}
        # DoS behaviour
        sess = f"dos-{time.time()}"
        blocked = sum(rag.answer(f"What PCI is assigned to cell C{100 + i}-1?", session=sess, use_cache=False,
                                 use_memory=False).refusal_reason.startswith("dos") for i in range(40))
        t0 = time.perf_counter()
        big = rag.answer("x" * 5000, session="dos-big")
        res["dos"] = {"burst_requests": 40, "rate_limit_rpm": self.cfg.get("guardrails.dos.max_requests_per_minute"),
                      "blocked_by_rate_limit": int(blocked), "oversized_blocked": bool(big.refused),
                      "oversized_reject_ms": round((time.perf_counter() - t0) * 1000, 3)}
        # tokens: naive baseline vs optimised design (NFR2)
        trow = []
        for label, kw, ctxmode in [
            ("naive: whole pages + raw images (zero-shot)", {"variant": "zero_shot"}, "naive"),
            ("chunks, no compression (jinja2)", {"variant": "jinja2", "nocompress": True}, "compressed"),
            ("final: top-k compressed chunks + OCR text (jinja2)", {"variant": "jinja2"}, "compressed"),
        ]:
            cfg2 = self.cfg.clone(context__compress=not kw.get("nocompress", False))
            rag2 = RAGSystem(cfg2, rag.index, rag.llm)
            d = run_questions(rag2, cq, kw["variant"], tag=f"tok-{label[:6]}", context_mode=ctxmode)
            s = summarise(d, self.ks)
            trow.append({"design": label, "avg_input_tokens": s["avg_input_tokens"], "avg_output_tokens": s["avg_output_tokens"],
                         "answer_accuracy": s["answer_accuracy"], "faithfulness": s["faithfulness"],
                         "latency_p50_s": s["latency_p50_s"]})
        # repeated query served from cache
        q0 = cq[0]["question"]
        rag.answer(q0, session="cache-a", use_memory=False)
        a2 = rag.answer(q0, session="cache-b", use_memory=False)
        trow.append({"design": "repeated query served from answer cache", "avg_input_tokens": a2.input_tokens,
                     "avg_output_tokens": a2.output_tokens, "answer_accuracy": None, "faithfulness": None,
                     "latency_p50_s": a2.latency_s})
        tdf = pd.DataFrame(trow)
        base = tdf.avg_input_tokens.iloc[0]
        tdf["token_reduction_vs_naive"] = [f"{(1 - v / base) * 100:.1f}%" if base else "-" for v in tdf.avg_input_tokens]
        self._save("e6_token_comparison.csv", tdf)
        res["tokens"] = tdf.to_dict(orient="records")
        res["tokenizer"] = tokenizer_name()
        self._save("e6_nonfunctional.json", res)
        return res

    # ------------------------------------------------------------------ E7
    def e7(self) -> pd.DataFrame:
        qs = self._subset()
        rows, per_q = [], []
        cons_q = _balanced_subset(qs, 6)
        for v in ABLATION_VARIANTS + ["naive-baseline"]:
            variant, mode = ("zero_shot", "naive") if v == "naive-baseline" else (v, "compressed")
            d = run_questions(self.rag, qs, variant, tag=f"e7-{v}", context_mode=mode)
            d["variant"] = v
            per_q.append(d)
            s = summarise(d, self.ks)
            # per-variant consistency (5 runs on a balanced subset)
            same = []
            for q in cons_q:
                outs = {self.rag.answer(q["question"], session=f"e7c-{v}-{q['qid']}-{i}", variant=variant, use_cache=False,
                                        use_memory=False, context_mode=mode).answer for i in range(5)}
                same.append(len(outs) == 1)
            rows.append({"variant": v, "answer_accuracy": s["answer_accuracy"], "faithfulness": s["faithfulness"],
                         "citation_accuracy": s["citation_accuracy"], "avg_input_tokens": s["avg_input_tokens"],
                         "avg_output_tokens": s["avg_output_tokens"], "latency_p50_s": s["latency_p50_s"],
                         "latency_p95_s": s["latency_p95_s"], "consistency": round(float(np.mean(same)), 3)})
        df = pd.DataFrame(rows)
        self._save("e7_prompt_ablation.csv", df)
        self._save("e7_prompt_ablation_per_question.csv", pd.concat(per_q))
        return df

    # ------------------------------------------------------------------ E8
    def e8(self) -> dict:
        qs = self._subset()
        dev, test = split_dev_test(qs, self.cfg.get("evaluation.dev_fraction", 0.5), self.cfg.get("project.seed", 42))
        opt = PromptOptimizer(self.rag, dev, test, self.cfg.get("evaluation.optimizer_iterations", 4),
                              self.cfg.get("evaluation.token_budget_penalty", 0.0005))
        res = opt.run(self.out)
        self.rag.prompts = PromptBuilder(self.cfg)               # reload optimised prompt
        md = ["# E8 - Self-improving prompt log", "",
              f"LLM: `{res['llm']}`  |  dev = {res['dev_size']} questions, held-out test = {res['test_size']} questions",
              f"Objective: `{res['objective']}`", ""]
        if not res["generative"]:
            md += ["> Note: the extractive fallback ignores prompt wording, so instruction edits cannot change its "
                   "answers; only token cost changes. Run with an Ollama / OpenAI-compatible model to see real gains.", ""]
        tab = []
        for r in res["iterations"]:
            tab.append({"iter": r["iteration"], "change": r["change"], "accepted": "yes" if r["accepted"] else "no",
                        "dev_accuracy": r["dev"]["answer_accuracy"], "dev_faithfulness": r["dev"]["faithfulness"],
                        "dev_input_tokens": r["dev"]["avg_input_tokens"], "objective": r["dev"]["objective"]})
        md += [_md(pd.DataFrame(tab)), "", "## Instruction per iteration", ""]
        for r in res["iterations"]:
            md += [f"- iter {r['iteration']} ({'accepted' if r['accepted'] else 'rejected'}): `{r['instruction']}` "
                   f"demos={r['demos']}"]
        tb, ta = res["test_before"], res["test_after"]
        md += ["", "## Held-out test split", "",
               _md(pd.DataFrame([{"prompt": "initial", **{k: tb[k] for k in ("answer_accuracy", "faithfulness", "avg_input_tokens", "objective")}},
                                 {"prompt": "optimised", **{k: ta[k] for k in ("answer_accuracy", "faithfulness", "avg_input_tokens", "objective")}}])),
               "", "## Final prompt", "", f"Instruction: `{res['final_prompt']['instruction']}`",
               f"Demos: {[d.get('qid') for d in res['final_prompt']['demos']]}"]
        (self.out / "e8_selfimprove_log.md").write_text("\n".join(md), encoding="utf-8")
        return res

    # ------------------------------------------------------------------ E9
    def e9(self) -> dict:
        tests = load_jsonl(self.cfg.path("guard_tests"))
        rows = []
        for t in tests:
            a = self.rag.answer(t["query"], session=f"e9-{t['tid']}", use_cache=False, use_memory=False)
            got = "block" if a.refused else "allow"
            rows.append({"tid": t["tid"], "category": t["category"], "query": t["query"][:70] + ("..." if len(t["query"]) > 70 else ""),
                         "expected": t["expected"], "result": got, "pass": int(got == t["expected"]),
                         "reason": a.refusal_reason, "guard_ms": round(1000 * sum(v for k, v in a.timings.items()
                                                                                  if k.startswith("guard_")), 3)})
        df = pd.DataFrame(rows)
        self._save("e9_guardrail_tests.csv", df)
        shutil.copy(self.cfg.root / "config" / "ontology.yaml", self.out / "e9_ontology.yaml")
        # how prompts consume the ontology
        q = next(g for g in self.golden if g["modality"] == "table")
        route = self.rag.router.route(q["question"])
        hits, info = self.rag.retriever.search(q["question"], route)
        ctx, _ = build_context(hits, q["question"], self.cfg)
        prompt = self.rag.prompts.build("jinja2", q["question"], ctx, route)[0]["content"]
        (self.out / "e9_rendered_prompt_example.txt").write_text(
            f"# Query: {q['question']}\n# Route (ontology): {json.dumps(route.as_dict())}\n# Filters: {info['filters']}\n\n"
            + prompt, encoding="utf-8")
        routes = []
        for g in self.golden[::6]:
            r = self.rag.router.route(g["question"])
            _, inf = self.rag.retriever.search(g["question"], r)
            routes.append({"query": g["question"][:70], "topics": ",".join(r.topics), "modality": r.modality,
                           "doc_type": r.doc_type, "entities": ",".join(r.entities), "filter": "; ".join(inf["filters"])})
        rdf = pd.DataFrame(routes)
        self._save("e9_ontology_routing.csv", rdf)
        return {"tests": df, "pass_rate": float(df["pass"].mean()), "routing": rdf}

    # ------------------------------------------------------------------ E10
    def e10(self) -> pd.DataFrame:
        abl = pd.read_csv(self.out / "e7_prompt_ablation.csv") if (self.out / "e7_prompt_ablation.csv").exists() else self.e7()
        qs = self._subset()
        d = run_questions(self.rag, qs, "optimized", tag="e10-opt")
        s = summarise(d, self.ks)
        cons_q = _balanced_subset(qs, 6)
        same = [len({self.rag.answer(q["question"], session=f"e10c-{q['qid']}-{i}", variant="optimized", use_cache=False,
                                     use_memory=False).answer for i in range(5)}) == 1 for q in cons_q]
        row = {"variant": "optimized (E8)", "answer_accuracy": s["answer_accuracy"], "faithfulness": s["faithfulness"],
               "citation_accuracy": s["citation_accuracy"], "avg_input_tokens": s["avg_input_tokens"],
               "avg_output_tokens": s["avg_output_tokens"], "latency_p50_s": s["latency_p50_s"],
               "latency_p95_s": s["latency_p95_s"], "consistency": round(float(np.mean(same)), 3)}
        bench = pd.concat([abl, pd.DataFrame([row])], ignore_index=True)
        base = bench.loc[bench.variant == "naive-baseline", "avg_input_tokens"]
        if len(base):
            bench["input_tokens_vs_naive"] = [f"{(v / base.iloc[0] - 1) * 100:+.1f}%" for v in bench.avg_input_tokens]
        self._save("e10_benchmark.csv", bench)
        # integrated demo transcript
        demo = []
        for q in _balanced_subset(self.golden, 3):
            a = self.rag.answer(q["question"], session="e10-demo", use_cache=False)
            demo.append(a.to_dict())
        (self.out / "e10_demo_answers.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
        return bench

    # ------------------------------------------------------------------ run all
    def run_all(self, force_ingest: bool = False, only: list[str] | None = None) -> None:
        steps = [("E1", lambda: self.e1(force_ingest)), ("E2", self.e2), ("E3", self.e3), ("E4", self.e4),
                 ("E5", self.e5), ("E6", self.e6), ("E7", self.e7), ("E8", self.e8), ("E9", self.e9), ("E10", self.e10)]
        status = {}
        for name, fn in steps:
            if only and name not in only:
                continue
            t0 = time.time()
            log.info("===== %s =====", name)
            try:
                fn()
                status[name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
            except Exception as e:  # keep going, report at the end
                log.exception("%s failed", name)
                status[name] = {"status": f"error: {type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        self._save("run_status.json", status)
        build_report_tables(self.cfg, self.rag)
        log.info("All results in %s (see REPORT_TABLES.md)", self.out)


# ----------------------------------------------------------------------------
def build_report_tables(cfg: Config, rag: RAGSystem | None = None) -> Path:
    out = cfg.path("outputs_dir")

    def csv(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else None

    def js(name):
        p = out / name
        return json.loads(p.read_text()) if p.exists() else None

    L = ["# Part A - Results tables (auto-generated)", "",
         f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  tokenizer: {tokenizer_name()}"]
    if rag is not None:
        L.append(f"LLM: `{rag.llm.name}`  |  embedder: `{rag.index.embedder.name}`  |  vector store: `{rag.index.manifest.get('store')}`"
                 f"  |  re-ranker: `{'on' if rag.retriever.reranker is not None else 'off'}`")
    L.append("")
    st = js("e1_ingestion_stats.json")
    if st:
        keys = ["documents_parsed", "pdf", "pptx_ppt", "pages_or_slides", "text_blocks", "tables", "images",
                "ocr_characters", "chunks_total", "chunks_text", "chunks_table", "chunks_image", "avg_chunk_tokens",
                "parse_seconds_total", "ocr_engine", "fr1_satisfied"]
        L += ["## E1 - Corpus and ingestion", "", _md(pd.DataFrame([{"metric": k, "value": st.get(k)} for k in keys])), ""]
    for title, name in [("E2 - Embedding and chunking configurations", "e2_embedding_chunking.csv"),
                        ("E3 - Retrieval design (text-only baseline vs multimodal / metadata-aware)", "e3_retrieval_design.csv"),
                        ("E5 - Golden set by modality", "e5_golden_by_modality.csv"),
                        ("E5 - Automatic evaluation vs golden set", "e5_auto_vs_golden.csv"),
                        ("E5 - Failure analysis", "e5_failure_analysis.csv"),
                        ("E6 - Consistency (NFR1)", "e6_consistency.csv"),
                        ("E6 - Token consumption vs naive baseline (NFR2)", "e6_token_comparison.csv"),
                        ("E7 - Prompt ablation", "e7_prompt_ablation.csv"),
                        ("E9 - Guardrail tests", "e9_guardrail_tests.csv"),
                        ("E9 - Ontology routing examples", "e9_ontology_routing.csv"),
                        ("E10 - Final benchmark (quality, tokens, latency, consistency)", "e10_benchmark.csv")]:
        df = csv(name)
        if df is not None:
            if name == "e5_failure_analysis.csv":
                df = df[["qid", "modality", "gold", "system_answer", "cause"]]
            if name == "e6_consistency.csv":
                df = pd.DataFrame([{"queries": len(df), "runs_per_query": int(df.runs.iloc[0]) if len(df) else 0,
                                    "identical_retrieval_share": df.identical_retrieval.mean(),
                                    "identical_answer_share": df.identical_answer.mean()}])
            L += [f"## {title}", "", _md(df), ""]
    e5 = js("e5_summary.json")
    if e5:
        L += ["## E5 - Golden set summary", "", _md(pd.DataFrame([e5["golden"]])), "",
              "Judge agreement with golden labels:", "", _md(pd.DataFrame([e5["judge_agreement"]])), ""]
    e6 = js("e6_nonfunctional.json")
    if e6:
        L += ["## E6 - Latency (FR3), guardrail overhead (FR4), DoS", "",
              _md(pd.DataFrame([e6["latency"]])), "", _md(pd.DataFrame([e6["guardrail_overhead"]])), "",
              _md(pd.DataFrame([e6["dos"]])), ""]
    e8 = out / "e8_selfimprove_log.md"
    if e8.exists():
        L += [e8.read_text(encoding="utf-8").replace("# E8", "## E8"), ""]
    e4 = out / "e4_memory_transcript.md"
    if e4.exists():
        L += [e4.read_text(encoding="utf-8").replace("# E4", "## E4"), ""]
    p = out / "REPORT_TABLES.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
