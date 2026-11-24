"""Automatic failure diagnosis for golden-set misses (Task A4, failure analysis)."""
from __future__ import annotations

import pandas as pd

from .metrics import answer_correct

FIXES = {
    "retrieval_miss": "Gold page not in top-k. Add the identifier to the entity regex / ontology so the metadata "
                      "filter fires, raise candidate_k, or enable the cross-encoder re-ranker.",
    "ocr_miss": "The value is not in the OCR text of the figure. Raise render DPI / upscale crops, switch OCR engine, "
                "or add a VLM caption for diagrams with small fonts.",
    "wrong_evidence": "The right chunk was retrieved but the answer used another unit (caption, other row or other "
                      "modality). Strengthen modality routing, compress context harder, or add a table/figure "
                      "few-shot demo to the prompt.",
    "false_refusal": "Evidence was present but the output guard refused (low grounding / no citation). Lower "
                     "min_grounding slightly or repair citations; check that the model copies values verbatim.",
    "generation_error": "Context contained the value but the generator paraphrased or picked a neighbouring value. "
                        "Add an instruction to copy values with units exactly; use the optimised prompt.",
}


def diagnose(row: pd.Series, index, golden: dict) -> str:
    if row["refused"] and not pd.isna(row["rank"]):
        return "false_refusal"
    if row["rank"] is None or pd.isna(row["rank"]):
        return "retrieval_miss"
    if golden["modality"] == "image":
        img = [c for c in index.chunks.values() if c.doc_id == golden["doc_id"] and c.page == golden["page"]
               and c.modality == "image"]
        if img and not any(answer_correct(c.text, golden["key"]) for c in img):
            return "ocr_miss"
    gold_prefix = f"{golden['doc_id']}-p{golden['page']}-"
    if row.cited and not any(c.startswith(gold_prefix) for c in str(row.cited).split("|")):
        return "wrong_evidence"
    return "generation_error"


def failure_table(df: pd.DataFrame, index, golden_list: list[dict]) -> pd.DataFrame:
    gl = {g["qid"]: g for g in golden_list}
    rows = []
    for _, r in df[df.correct == 0].iterrows():
        g = gl[r.qid]
        cause = diagnose(r, index, g)
        rows.append({"qid": r.qid, "modality": r.modality, "question": r.question, "gold": r.gold,
                     "system_answer": r.answer, "gold_source": f"{g['source']} p.{g['page']}",
                     "gold_rank": r["rank"], "cause": cause, "proposed_fix": FIXES[cause]})
    return pd.DataFrame(rows)
