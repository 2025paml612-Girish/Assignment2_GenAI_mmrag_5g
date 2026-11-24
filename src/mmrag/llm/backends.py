from __future__ import annotations

import re
import time

import requests

from ..index.bm25 import tokenize
from ..utils.log import get_logger
from ..utils.tokens import count_tokens
from .base import LLM, LLMResponse

log = get_logger("llm")


def _msgs_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) + 4 for m in messages)


class OllamaLLM(LLM):
    def __init__(self, host: str, model: str, seed: int = 42, timeout: float = 80):
        self.host, self.model, self.seed, self.timeout = host.rstrip("/"), model, seed, timeout
        self.name = f"ollama:{model}"

    @staticmethod
    def available(host: str) -> bool:
        try:
            return requests.get(host.rstrip("/") + "/api/tags", timeout=2).ok
        except Exception:
            return False

    def generate(self, messages, max_tokens=256, temperature=0.0, stop=None, **meta) -> LLMResponse:
        t0 = time.time()
        opts = {"temperature": temperature, "seed": self.seed, "num_predict": max_tokens, "top_k": 1}
        if stop:
            opts["stop"] = stop
        r = requests.post(f"{self.host}/api/chat", json={"model": self.model, "messages": messages, "stream": False,
                                                         "options": opts}, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        text = d.get("message", {}).get("content", "")
        return LLMResponse(text.strip(), d.get("prompt_eval_count") or _msgs_tokens(messages),
                           d.get("eval_count") or count_tokens(text), time.time() - t0, self.name)


class OpenAICompatLLM(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, seed: int = 42, timeout: float = 80):
        self.base_url, self.api_key, self.model, self.seed, self.timeout = base_url.rstrip("/"), api_key, model, seed, timeout
        self.name = f"openai_compat:{model}"

    def generate(self, messages, max_tokens=256, temperature=0.0, stop=None, **meta) -> LLMResponse:
        t0 = time.time()
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "seed": self.seed}
        if stop:
            body["stop"] = stop
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        d = r.json()
        text = d["choices"][0]["message"]["content"] or ""
        u = d.get("usage") or {}
        return LLMResponse(text.strip(), u.get("prompt_tokens") or _msgs_tokens(messages),
                           u.get("completion_tokens") or count_tokens(text), time.time() - t0, self.name)


class ExtractiveLLM(LLM):
    """Deterministic evidence selector used when no generative model is available.

    It reads the structured ``contexts`` passed by the pipeline (not the prompt
    text), scores sentences / table rows / OCR lines by weighted overlap with the
    question, and for tables returns the cell of the column named in the question.
    Its input-token count is still the real prompt size, so token metrics stay valid.
    """
    name = "extractive"
    generative = False

    def generate(self, messages, max_tokens=256, temperature=0.0, stop=None, **meta) -> LLMResponse:
        t0 = time.time()
        q = meta.get("question", messages[-1]["content"])
        ctxs: list[tuple[int, str]] = meta.get("contexts", [])
        text = self.answer(q, ctxs)
        return LLMResponse(text, _msgs_tokens(messages), count_tokens(text), time.time() - t0, self.name)

    @staticmethod
    def _w(tok: str) -> float:
        return 2.5 if any(ch.isdigit() for ch in tok) or "-" in tok else 1.0

    CAPTION = re.compile(r"^\s*(\[figure\]|\[chart\]|figure\b|fig\.|table\b|chart\b|results\b|document date|illustration\b|summary\b)", re.I)
    QSTOP = {"what", "which", "value", "shown", "figure", "table", "according", "annotated", "appears", "chart",
             "diagram", "measured", "recorded", "used", "how", "much", "many"}

    SYN = {"maximum": "max", "minimum": "min", "average": "avg", "avg": "avg", "max": "max", "min": "min"}

    def _toks(self, text: str) -> set[str]:
        return {self.SYN.get(t, t) for t in tokenize(text)}

    def answer(self, question: str, ctxs: list[tuple[int, str]]) -> str:
        qt = self._toks(question) - self.QSTOP
        # ---- candidate units: (text, ctx idx, kind, payload) -------------
        cands = []
        for idx, text in ctxs:
            lines = [l for l in text.splitlines() if l.strip()]
            rows = [l for l in lines if l.startswith("|") and not set(l) <= set("|-: ")]
            caps = [l for l in lines if not l.startswith("|")]
            cap = caps[0] if caps else ""
            if len(rows) >= 2:
                header = [c.strip() for c in rows[0].strip("|").split("|")]
                for row in rows[1:]:
                    cands.append((row, idx, "row", (header, [c.strip() for c in row.strip("|").split("|")], cap)))
                continue
            img = bool(lines) and lines[0].lstrip().startswith(("[Figure]", "[Chart]"))
            for l in lines:
                l = re.sub(r"^Text in image:\s*", "", l)
                for part in re.split(r" \| |, (?=[A-Z][a-z]{2}=)", l):
                    for u in re.split(r"(?<=[.!?])\s+", part):
                        u = u.strip()
                        if len(u) > 3 and not self.CAPTION.match(u):
                            cands.append((u, idx, "img" if img else "text", (None, None, cap if img else "")))
        if not cands:
            return "I could not find evidence for this in the indexed documents."
        # ---- context IDF: tokens present in most contexts (e.g. the site id) carry little signal
        ctx_toks = [self._toks(t) for _, t in ctxs]
        n = max(1, len(ctx_toks))

        def w(t):
            df = sum(t in ct for ct in ctx_toks)
            base = 2.5 if any(ch.isdigit() for ch in t) or "-" in t else 1.0
            return base * (0.25 if n >= 2 and df >= max(2, 0.75 * n) else 1.0)

        def score(unit, cap):
            ut = self._toks(unit)
            s = sum(w(t) for t in qt & ut)
            if s == 0:
                return 0.0
            s += 0.4 * sum(w(t) for t in qt & self._toks(cap)) if cap else 0.0
            new = [t for t in ut - qt if any(c.isdigit() for c in t) and not set(t.split("-")) <= qt]
            return s + (1.0 if new else -1.0)

        best = (0.0, "", 0)
        for unit, idx, kind, (header, cells, cap) in cands:
            sc = score(unit, cap)
            if sc <= best[0]:
                continue
            ans = unit
            if kind == "row":
                cols = []
                for j, h in enumerate(header):
                    ht = self._toks(re.sub(r"\(.*?\)", "", h))
                    ct = self._toks(cells[j]) if j < len(cells) else set()
                    if ht and not (ct & qt) and j < len(cells):
                        cols.append((len(ht & qt) / len(ht), -j, j))
                cols.sort(reverse=True)
                if cols and cols[0][0] > 0:
                    j = cols[0][2]
                    ans = f"{header[j]} for {cells[0]} is {cells[j]}"
                else:
                    ans = "; ".join(f"{h}: {c}" for h, c in zip(header, cells))
            best = (sc, ans, idx)
        if best[0] < 1.5:
            return "I could not find evidence for this in the indexed documents."
        return f"{best[1].rstrip('.')} [{best[2]}]."


def make_llm(cfg) -> LLM:
    b = cfg.get("llm.backend", "auto")
    seed, timeout = cfg.get("llm.seed", 42), cfg.get("llm.timeout_s", 80)
    host = cfg.get("llm.ollama.host")
    if b == "ollama" or (b == "auto" and OllamaLLM.available(host)):
        llm = OllamaLLM(host, cfg.get("llm.ollama.model"), seed, timeout)
    elif b == "openai_compat" or (b == "auto" and cfg.get("llm.openai_compat.base_url")):
        llm = OpenAICompatLLM(cfg.get("llm.openai_compat.base_url"), cfg.get("llm.openai_compat.api_key"),
                              cfg.get("llm.openai_compat.model"), seed, timeout)
    else:
        if b == "auto":
            log.warning("No Ollama server / OpenAI-compatible endpoint found -> extractive fallback LLM")
        llm = ExtractiveLLM()
    log.info("LLM backend: %s", llm.name)
    return llm
