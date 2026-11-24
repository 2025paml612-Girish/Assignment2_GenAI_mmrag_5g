"""Input guardrails: prompt injection, personal data (PII) and out-of-scope checks."""
from __future__ import annotations

import re

INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (the |all |your )?(rules|instructions|guidelines|context)",
    r"forget (your|all|the) (rules|guidelines|instructions)",
    r"(reveal|print|show|repeat) (your |the )?(system|hidden|initial) (prompt|instructions|configuration)",
    r"\byou are now\b", r"\bact as (an? )?(unrestricted|jailbroken|different)", r"\bdan\b", r"developer mode",
    r"jailbreak", r"</?\s*(context|system|instructions)\s*>", r"^\s*system\s*:", r"\bsystem:\s",
    r"(answer|respond) without (citations|sources)", r"invent (numbers|facts|data)",
]
PII = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_in": r"(?:\+91[\s-]?)?(?<!\d)[6-9]\d{4}[\s-]?\d{5}(?!\d)",
    "aadhaar": r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "card": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
}


def _luhn(num: str) -> bool:
    d = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(d) <= 16:
        return False
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x * 2 > 9 else x * 2
        s += x
    return s % 10 == 0


class InputGuard:
    def __init__(self, cfg):
        g = cfg.get("guardrails.input", {})
        self.block_injection = g.get("block_injection", True)
        self.block_pii = g.get("block_pii", True)
        self.oos = g.get("out_of_scope_check", True)
        self.min_hits = g.get("oos_min_domain_hits", 1)
        self.min_score = g.get("oos_min_retrieval_score", 0.12)
        self.inj = [re.compile(p, re.I | re.M) for p in INJECTION]
        self.pii = {k: re.compile(v) for k, v in PII.items()}

    def injection(self, q: str) -> str | None:
        for p in self.inj:
            if p.search(q):
                return p.pattern
        return None

    def find_pii(self, q: str) -> list[str]:
        found = []
        for k, p in self.pii.items():
            for m in p.finditer(q):
                if k == "card" and not _luhn(m.group(0)):
                    continue
                found.append(k)
        return sorted(set(found))

    def redact(self, q: str) -> str:
        for k, p in self.pii.items():
            q = p.sub(f"<{k.upper()}>", q)
        return q

    def check(self, q: str) -> tuple[bool, str, str]:
        """returns (allowed, reason, possibly-redacted query)"""
        if self.block_injection:
            pat = self.injection(q)
            if pat:
                return False, "input: prompt injection detected", q
        pii = self.find_pii(q)
        if pii:
            if self.block_pii:
                return False, f"input: personal data detected ({', '.join(pii)})", q
            q = self.redact(q)
        return True, "", q

    def out_of_scope(self, domain_hits: int, max_dense: float, max_sparse: float) -> bool:
        """Out of scope = no domain vocabulary AND weak retrieval evidence."""
        if not self.oos:
            return False
        return domain_hits < self.min_hits and (max_dense < self.min_score or max_sparse < 2.0)
