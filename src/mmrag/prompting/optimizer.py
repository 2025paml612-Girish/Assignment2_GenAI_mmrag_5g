"""Self-improving prompt (Task A5 / E8, FR5).

An equivalent of DSPy-style optimisation without the dependency, combining:
  1. LLM critique-and-rewrite : the LLM sees the current instruction and the dev
     failures and proposes a rewritten instruction (generative backends only);
  2. failure-driven instruction edits : a library of targeted edits chosen from the
     failure modalities / refusal pattern of the current iteration;
  3. bootstrapped few-shot selection : correctly answered dev examples (one per
     modality, compact context) are tried as demonstrations.

Each iteration evaluates every candidate on the DEV split and keeps the best one
only if the objective improves:
        objective = 0.7 * accuracy + 0.3 * faithfulness - penalty * avg_input_tokens
The final prompt is reported on the held-out TEST split (never used to optimise).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..eval.runner import run_questions, summarise
from ..utils.log import get_logger
from .variants import DEFAULT_INSTRUCTION

log = get_logger("optimizer")

EDITS = {
    "table": "For table evidence, find the row whose first cell matches the identifier in the question and return "
             "the value of the column the question asks for, with its unit.",
    "image": "Values from figures appear after 'Text in image:'; copy the value next to the label that the question "
             "names.",
    "text": "Copy identifiers, versions and numbers exactly as written in the context.",
    "refusal": "Refuse only if no context item mentions the identifier from the question.",
    "brevity": "Answer in at most 20 words.",
}


def split_dev_test(golden: list[dict], dev_fraction: float = 0.5, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rnd = random.Random(seed)
    dev, test = [], []
    for mod in ("text", "table", "image"):
        g = [q for q in golden if q["modality"] == mod]
        rnd.shuffle(g)
        cut = int(round(len(g) * dev_fraction))
        dev += g[:cut]
        test += g[cut:]
    return dev, test


class PromptOptimizer:
    def __init__(self, rag, dev: list[dict], test: list[dict], iterations: int = 4, penalty: float = 0.0005):
        self.rag, self.dev, self.test = rag, dev, test
        self.iterations, self.penalty = iterations, penalty

    def objective(self, s: dict) -> float:
        return 0.7 * s["answer_accuracy"] + 0.3 * s["faithfulness"] - self.penalty * s["avg_input_tokens"]

    def evaluate(self, instruction: str, demos: list[dict], qs: list[dict], tag: str):
        df = run_questions(self.rag, qs, "jinja2", tag=tag, instruction=instruction, demos=demos)
        s = summarise(df)
        s["objective"] = round(self.objective(s), 4)
        return s, df

    def _critique(self, instruction: str, df) -> str | None:
        if not getattr(self.rag.llm, "generative", False):
            return None
        fails = df[df.correct == 0].head(5)
        if fails.empty:
            return None
        ex = "\n".join(f"- Q: {r.question}\n  system answer: {r.answer}\n  expected: {r.gold}" for r in fails.itertuples())
        prompt = ("You improve instructions for a document question-answering prompt.\n"
                  f"Current instruction: \"{instruction}\"\nFailures on development questions:\n{ex}\n"
                  "Write an improved instruction (max 60 words) that would fix these failures without mentioning "
                  "the specific questions. Return only the instruction text.")
        try:
            out = self.rag.llm.generate([{"role": "user", "content": prompt}], max_tokens=120).text.strip().strip('"')
            return out if 10 < len(out) < 600 else None
        except Exception:
            return None

    @staticmethod
    def _bootstrap_demos(df, rag) -> list[dict]:
        demos = []
        for mod in ("table", "image", "text"):
            ok = df[(df.correct == 1) & (df.modality == mod)]
            if ok.empty:
                continue
            r = ok.iloc[0]
            cid = (r.cited or r.retrieved).split("|")[0]
            c = rag.index.chunks.get(cid)
            if c is None:
                continue
            ctx = f"[1] {c.source} | page {c.page} | {c.modality}\n{c.text[:260]}"
            demos.append({"context": ctx, "question": r.question, "answer": r.answer.split("[")[0].strip() + " [1].",
                          "qid": r.qid})
        return demos[:2]

    def run(self, out_dir: Path) -> dict:
        log_rows = []
        cur_inst, cur_demos = DEFAULT_INSTRUCTION, []
        cur_s, cur_df = self.evaluate(cur_inst, cur_demos, self.dev, "opt0")
        log_rows.append({"iteration": 0, "change": "baseline (default Jinja2 instruction, no demos)",
                         "instruction": cur_inst, "demos": [], "dev": cur_s, "accepted": True})
        applied: set[str] = set()
        for it in range(1, self.iterations + 1):
            cands = []
            crit = self._critique(cur_inst, cur_df)
            if crit:
                cands.append(("LLM critique-and-rewrite of the instruction", crit, cur_demos, "critique"))
            fails = cur_df[cur_df.correct == 0]
            order = list(fails.modality.value_counts().index) + ["refusal" if fails.refused.sum() else "brevity",
                                                                 "text", "table", "image", "brevity"]
            for key in order:
                if key in EDITS and key not in applied:
                    cands.append((f"failure-driven edit: +'{key}' rule", cur_inst + " " + EDITS[key], cur_demos, key))
                    break
            if not cur_demos and "demos" not in applied:
                demos = self._bootstrap_demos(cur_df, self.rag)
                if demos:
                    cands.append((f"bootstrapped few-shot: +{len(demos)} demos from correct dev answers "
                                  f"({', '.join(d['qid'] for d in demos)})", cur_inst, demos, "demos"))
            if not cands:
                log_rows.append({"iteration": it, "change": "no new candidate", "instruction": cur_inst,
                                 "demos": [d["qid"] for d in cur_demos], "dev": cur_s, "accepted": False})
                continue
            best = None
            applied.update(k for *_, k in cands if k != "critique")
            for desc, inst, demos, key in cands:
                s, df = self.evaluate(inst, demos, self.dev, f"opt{it}")
                log_rows.append({"iteration": it, "change": desc, "instruction": inst,
                                 "demos": [d["qid"] for d in demos], "dev": s, "accepted": False})
                if best is None or s["objective"] > best[0]["objective"]:
                    best = (s, df, inst, demos, key, len(log_rows) - 1)
            s, df, inst, demos, key, li = best
            if s["objective"] > cur_s["objective"] + 1e-9:
                cur_inst, cur_demos, cur_s, cur_df = inst, demos, s, df
                log_rows[li]["accepted"] = True
            log.info("iteration %d: best candidate objective %.4f (current %.4f)", it, s["objective"], cur_s["objective"])
        base_test, _ = self.evaluate(DEFAULT_INSTRUCTION, [], self.test, "opt-test-base")
        final_test, _ = self.evaluate(cur_inst, cur_demos, self.test, "opt-test-final")
        result = {"llm": self.rag.llm.name, "generative": getattr(self.rag.llm, "generative", False),
                  "dev_size": len(self.dev), "test_size": len(self.test), "dev_qids": [q["qid"] for q in self.dev],
                  "test_qids": [q["qid"] for q in self.test], "objective": "0.7*accuracy + 0.3*faithfulness - "
                  f"{self.penalty}*input_tokens", "iterations": log_rows,
                  "final_prompt": {"instruction": cur_inst, "demos": cur_demos},
                  "test_before": base_test, "test_after": final_test}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "e8_selfimprove_log.json").write_text(json.dumps(result, indent=2))
        (out_dir / "e8_optimized_prompt.json").write_text(json.dumps({"instruction": cur_inst, "demos": cur_demos}, indent=2))
        return result
