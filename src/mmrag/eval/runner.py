"""Run a question set through the system and collect per-question metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import answer_correct, first_relevant_rank, pct, retrieval_summary


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def run_questions(rag, questions: list[dict], variant: str | None = None, *, tag: str = "", use_guardrails=None,
                  context_mode: str = "compressed", instruction=None, demos=None, retrieval_kwargs=None,
                  recall_k: int = 5) -> pd.DataFrame:
    rows = []
    for q in questions:
        # retrieval metrics are measured on the top-``recall_k`` list
        hits_k, _ = rag.retriever.search(q["question"], k=recall_k, **(retrieval_kwargs or {}))
        rank = first_relevant_rank(hits_k, q["doc_id"], q["page"], q.get("modality"))
        doc_rank = next((i for i, h in enumerate(hits_k, 1) if h.chunk.doc_id == q["doc_id"]), None)
        a = rag.answer(q["question"], session=f"eval-{tag}-{q['qid']}", variant=variant, use_cache=False,
                       use_memory=False, use_guardrails=use_guardrails, context_mode=context_mode,
                       instruction=instruction, demos=demos, retrieval_kwargs=retrieval_kwargs)
        correct = (not a.refused) and answer_correct(a.answer, q["key"], q.get("aliases"))
        cited_ok = any(c.get("chunk_id", "").startswith(q["doc_id"] + "-p" + str(q["page"]) + "-") for c in a.citations)
        rows.append({
            "qid": q["qid"], "modality": q["modality"], "difficulty": q.get("difficulty", ""),
            "question": q["question"], "gold": q["answer"], "answer": a.answer,
            "rank": rank, "doc_rank": doc_rank, "correct": int(correct),
            "faithfulness": round(a.grounding, 3) if not a.refused else 0.0,
            "citation_correct": int(cited_ok), "refused": int(a.refused), "refusal_reason": a.refusal_reason,
            "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "latency_s": a.latency_s,
            "guard_s": sum(v for k, v in a.timings.items() if k.startswith("guard_")),
            "retrieved": "|".join(h.chunk.chunk_id for h in a.hits),
            "cited": "|".join(c.get("chunk_id", "") for c in a.citations),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, ks=(1, 3, 5)) -> dict:
    ranks = [None if (r is None or (isinstance(r, float) and np.isnan(r))) else int(r) for r in df["rank"]]
    s = retrieval_summary(ranks, ks)
    s.update({
        "n": int(len(df)),
        "answer_accuracy": float(df.correct.mean()),
        "faithfulness": float(df.faithfulness.mean()),
        "citation_accuracy": float(df.citation_correct.mean()),
        "refusal_rate": float(df.refused.mean()),
        "avg_input_tokens": float(df.input_tokens.mean()),
        "avg_output_tokens": float(df.output_tokens.mean()),
        "latency_p50_s": pct(df.latency_s, 50),
        "latency_p95_s": pct(df.latency_s, 95),
    })
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


def by_modality(df: pd.DataFrame, ks=(1, 3, 5)) -> pd.DataFrame:
    rows = []
    for m, g in df.groupby("modality"):
        s = summarise(g, ks)
        s["modality"] = m
        rows.append(s)
    return pd.DataFrame(rows)
