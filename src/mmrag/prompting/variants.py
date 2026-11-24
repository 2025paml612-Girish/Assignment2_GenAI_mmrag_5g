"""Prompt variants for the ablation study (Task A5 / E7).

zero_shot  : bare context + question
fstring    : Python f-string with role, grounding and citation rules
three_shot : f-string + 3 fixed demonstrations (prompts/three_shot_examples.json,
             written for documents that are NOT in the corpus -> no test leakage)
jinja2     : prompts/answer_jinja2.j2, filled from the metadata ontology
             (topic labels, routed modality, per-chunk source/page/modality, memory)
create     : prompts/create.j2  - CREATE framework
react      : prompts/react.j2   - ReAct loop with a search[...] tool (pipeline runs the loop)
optimized  : jinja2 template with the instruction + demos found by the
             self-improving optimiser (outputs/e8_optimized_prompt.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import ROOT

VARIANTS = ["zero_shot", "fstring", "three_shot", "jinja2", "create", "react", "optimized"]
DEFAULT_INSTRUCTION = "Answer the question precisely and briefly."
DOMAIN = "5G NR / O-RAN"
USER_ROLE = "RAN test and automation engineers"


class PromptBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        labels = {k: v.get("label", k) for k, v in cfg.ontology.get("topics", {}).items()}
        self.env = Environment(loader=FileSystemLoader(str(ROOT / "prompts")), undefined=StrictUndefined,
                               trim_blocks=False, lstrip_blocks=False)
        self.env.filters["topic_label"] = lambda t: labels.get(t, t)
        self.examples = json.loads((ROOT / "prompts" / "three_shot_examples.json").read_text(encoding="utf-8"))
        self.refusal = cfg.get("guardrails.output.refusal_text")
        self.optimized = None
        p = Path(cfg.get("prompting.optimized_prompt_file", ""))
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            self.optimized = json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def ctx_block(contexts) -> str:
        return "\n".join(f"{c.header()}\n{c.text}" for c in contexts)

    def build(self, variant: str, question: str, contexts, route=None, history=None, instruction: str | None = None,
              demos: list[dict] | None = None) -> list[dict]:
        ctx = self.ctx_block(contexts)
        rd = route.as_dict() if route is not None else {"topics": [], "modality": None, "doc_type": None}
        if variant == "zero_shot":
            text = f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        elif variant == "fstring":
            text = (f"You are a {DOMAIN} documentation assistant. Answer using only the context and cite sources as [n]. "
                    f"If the answer is not in the context, say \"{self.refusal}\"\n\nContext:\n{ctx}\n\n"
                    f"Question: {question}\nAnswer:")
        elif variant == "three_shot":
            shots = "\n\n".join(f"Context:\n{e['context']}\nQuestion: {e['question']}\nAnswer: {e['answer']}"
                                for e in self.examples)
            text = (f"You are a {DOMAIN} documentation assistant. Answer using only the context and cite sources as [n]. "
                    f"If the answer is not in the context, say \"{self.refusal}\"\n\n{shots}\n\n"
                    f"Context:\n{ctx}\nQuestion: {question}\nAnswer:")
        elif variant in ("jinja2", "optimized"):
            if variant == "optimized" and self.optimized and instruction is None:
                instruction = self.optimized.get("instruction")
                demos = self.optimized.get("demos", [])
            text = self.env.get_template("answer_jinja2.j2").render(
                domain=DOMAIN, user_role=USER_ROLE, instruction=instruction or DEFAULT_INSTRUCTION,
                refusal=self.refusal, route=_Obj(rd), history=history or [], demos=demos or [],
                contexts=contexts, question=question)
        elif variant == "create":
            text = self.env.get_template("create.j2").render(domain=DOMAIN, user_role=USER_ROLE, refusal=self.refusal,
                                                             route=_Obj(rd), contexts=contexts, question=question)
        elif variant == "react":
            text = self.env.get_template("react.j2").render(domain=DOMAIN, refusal=self.refusal, contexts=contexts,
                                                            question=question, max_steps=2)
        else:
            raise ValueError(f"unknown prompt variant {variant}; choose from {VARIANTS}")
        return [{"role": "user", "content": text.strip()}]


class _Obj(dict):
    """dict with attribute access for Jinja2 (route.modality)."""
    __getattr__ = dict.get
