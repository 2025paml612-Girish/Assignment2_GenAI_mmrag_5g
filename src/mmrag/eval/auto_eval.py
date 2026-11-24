"""Automatic evaluation without hand labels (Task A4, second bullet).

Approach 1 - Synthetic question generation from the corpus
    Sample chunks that are NOT used by the golden set; from each, pick a "fact
    unit" (a table row cell, a sentence or an OCR line that contains a number /
    identifier) and turn it into a question whose answer is the masked value.
    With a generative LLM the question is written by the LLM; otherwise a
    deterministic template is used. The source chunk is the label, so
    Recall@k, MRR and answer accuracy are computed with no human effort.

Approach 2 - Reference-free judge (LLM-as-judge with a rubric)
    The judge sees only question, retrieved context and answer (no gold answer)
    and scores: grounded (answer supported by context), cited, responsive
    (answers the asked type, e.g. a number when the question asks a value).
    With a generative LLM the rubric is applied by the LLM; otherwise a
    deterministic proxy (grounding score + citation validity + type check) is used.
    Its verdicts on the golden questions are compared with the golden labels
    (agreement, Cohen's kappa, confusion matrix).
"""
from __future__ import annotations

import random
import re

import pandas as pd

from ..guardrails.output_guard import CITE, grounding_score
from ..index.bm25 import tokenize
from ..ingest.chunker import split_sentences
from .metrics import answer_correct

NUM = re.compile(r"\b\d[\d.:]*\b")


# ----------------------------------------------------------------------------
# approach 1: synthetic questions
# ----------------------------------------------------------------------------
def _template_question(chunk) -> tuple[str, str] | None:
    title = chunk.title
    if chunk.modality == "table":
        rows = [l for l in chunk.text.splitlines() if l.startswith("|") and not set(l) <= set("|-: ")]
        if len(rows) < 2:
            return None
        header = [c.strip() for c in rows[0].strip("|").split("|")]
        r = rows[1 + (len(chunk.chunk_id) % (len(rows) - 1))]
        cells = [c.strip() for c in r.strip("|").split("|")]
        for j in range(len(header) - 1, 0, -1):
            if j < len(cells) and NUM.search(cells[j]):
                return f"In the {title}, what is the {header[j]} for {cells[0]}?", cells[j]
        return None
    if chunk.modality == "image":
        lines = [l for l in re.sub(r"Text in image:\s*", "", chunk.text).splitlines()[1:] if NUM.search(l)]
        for l in lines:
            m = list(NUM.finditer(l))[-1]
            stem = l[:m.start()].strip(" :=-")
            if len(stem) >= 4:
                return f"In the figure of the {title}, what value is shown for '{stem}'?", m.group(0)
        return None
    for s in split_sentences(chunk.text):
        m = re.search(r"\b(?:R\d{2}\.\d+\.\d+|DEF-\d+|\d+(?:\.\d+)?\s?(?:ms|dB|bytes|s|ns|us|MHz|GB|%))\b", s)
        if m and len(s) < 220:
            masked = s.replace(m.group(0), "____", 1)
            return f"Fill the blank from the {title}: \"{masked}\"", m.group(0)
    return None


def generate_synthetic(index, exclude_docs: set[str], n: int = 30, seed: int = 42, llm=None) -> list[dict]:
    rnd = random.Random(seed)
    pool = [c for c in index.chunks.values() if c.doc_id not in exclude_docs]
    rnd.shuffle(pool)
    out, per_mod = [], {"text": 0, "table": 0, "image": 0}
    cap = (n + 2) // 3
    for c in pool:
        if per_mod[c.modality] >= cap:
            continue
        qa = None
        if llm is not None and getattr(llm, "generative", False):
            prompt = ("Write ONE question that can be answered only from this passage, and its short answer.\n"
                      f"Passage ({c.title}, {c.modality}):\n{c.text[:900]}\n"
                      "Mention the document identifier in the question. Format:\nQ: <question>\nA: <answer>")
            try:
                t = llm.generate([{"role": "user", "content": prompt}], max_tokens=90).text
                q = re.search(r"Q:\s*(.+)", t)
                a = re.search(r"A:\s*(.+)", t)
                if q and a and a.group(1).strip().lower() in c.text.lower():
                    qa = (q.group(1).strip(), a.group(1).strip())
            except Exception:
                qa = None
        qa = qa or _template_question(c)
        if not qa:
            continue
        per_mod[c.modality] += 1
        out.append({"qid": f"S{len(out) + 1:03d}", "question": qa[0], "answer": qa[1], "key": qa[1],
                    "modality": c.modality, "doc_id": c.doc_id, "page": c.page, "source": c.source,
                    "source_chunk": c.chunk_id, "generator": "llm" if llm is not None and getattr(llm, "generative", False) else "template"})
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------
# approach 2: reference-free judge
# ----------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading an answer produced by a document QA system. You do NOT know the true answer.
Question: {question}
Retrieved context:
{context}
Answer: {answer}
Rubric:
- GROUNDED: every fact in the answer is stated in the context (yes/no)
- CITED: the answer cites the context items it uses as [n] (yes/no)
- RESPONSIVE: the answer gives the specific item the question asks for (yes/no)
Reply exactly in the form: GROUNDED=yes|no; CITED=yes|no; RESPONSIVE=yes|no"""


ATTR = re.compile(r"^\s*(?:what|which|how many|how much|how often|who|when|where)\s+(.*?)\s+(?:is|was|were|are|does|do|did|has|"
                  r"have|appears?|carries|runs?|shown|annotated|assigned|allocated|configured|measured|recorded|used|"
                  r"contains?|uses?)\b", re.I)
GENERIC = {"value", "the", "a", "an", "of", "in", "for", "maximum", "max", "average", "actual", "type", "id"}


def asked_attribute(question: str) -> set[str]:
    """Words naming what the question asks for, e.g. 'how much memory is ...' -> {memory}."""
    m = ATTR.search(question)
    words = tokenize(m.group(1)) if m else tokenize(question)[:3]
    out = {w for w in words if w not in GENERIC and not any(c.isdigit() for c in w)}
    return out or set(words[:2])


def heuristic_judge(question: str, context_texts: list[str], answer: str, refused: bool) -> dict:
    """Deterministic rubric proxy (no gold answer):
    grounded   - >= 60 % of answer tokens occur in the retrieved context
    cited      - the answer carries a [n] citation
    responsive - the answer supplies a new value AND the evidence line holding that value
                 talks about the attribute the question asks for (>= 50 % word coverage)."""
    if refused:
        return {"grounded": 0, "cited": 0, "responsive": 0, "pass": 0}
    clean = CITE.sub("", answer)
    grounded = int(grounding_score(answer, context_texts) >= 0.6)
    cited = int(bool(CITE.search(answer)))
    q_toks = set(tokenize(question))
    new_vals = [t for t in tokenize(clean) if any(c.isdigit() for c in t) and t not in q_toks]
    attr = asked_attribute(question)
    evidence = set(tokenize(clean))
    for line in (l for t in context_texts for l in t.splitlines()):
        lt = set(tokenize(line))
        if any(v in lt for v in new_vals):
            evidence |= lt
    cover = len(attr & evidence) / max(1, len(attr))
    responsive = int(bool(new_vals or len(clean.split()) >= 4) and cover >= 0.5)
    return {"grounded": grounded, "cited": cited, "responsive": responsive,
            "pass": int(grounded and cited and responsive)}


def llm_judge(llm, question, context_texts, answer, refused) -> dict:
    if refused:
        return {"grounded": 0, "cited": 0, "responsive": 0, "pass": 0}
    ctx = "\n".join(f"[{i + 1}] {t[:400]}" for i, t in enumerate(context_texts))
    t = llm.generate([{"role": "user", "content": JUDGE_PROMPT.format(question=question, context=ctx, answer=answer)}],
                     max_tokens=30).text.lower()
    g = int("grounded=yes" in t.replace(" ", ""))
    c = int("cited=yes" in t.replace(" ", ""))
    r = int("responsive=yes" in t.replace(" ", ""))
    return {"grounded": g, "cited": c, "responsive": r, "pass": int(g and c and r)}


def judge_answers(rag, questions: list[dict]) -> pd.DataFrame:
    use_llm = getattr(rag.llm, "generative", False)
    rows = []
    for q in questions:
        a = rag.answer(q["question"], session=f"judge-{q['qid']}", use_cache=False, use_memory=False)
        ctx = [h.chunk.text for h in a.hits]
        j = llm_judge(rag.llm, q["question"], ctx, a.answer, a.refused) if use_llm else \
            heuristic_judge(q["question"], ctx, a.answer, a.refused)
        gold = int((not a.refused) and answer_correct(a.answer, q["key"], q.get("aliases")))
        rows.append({"qid": q["qid"], "modality": q["modality"], "answer": a.answer, "gold_correct": gold,
                     "judge_pass": j["pass"], "judge_grounded": j["grounded"], "judge_cited": j["cited"],
                     "judge_responsive": j["responsive"], "judge": "llm-rubric" if use_llm else "heuristic-rubric"})
    return pd.DataFrame(rows)


def agreement(df: pd.DataFrame) -> dict:
    a, b = df.gold_correct.astype(int), df.judge_pass.astype(int)
    n = len(df)
    po = float((a == b).mean()) if n else 0.0
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 4), "cohen_kappa": round(float(kappa), 4),
            "tp_both_correct": int(((a == 1) & (b == 1)).sum()), "tn_both_wrong": int(((a == 0) & (b == 0)).sum()),
            "judge_false_pass": int(((a == 0) & (b == 1)).sum()), "judge_false_fail": int(((a == 1) & (b == 0)).sum()),
            "golden_accuracy": round(float(pa), 4), "judge_pass_rate": round(float(pb), 4)}
