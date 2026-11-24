"""Token counting used for every token metric in the project.

Uses tiktoken ``cl100k_base`` when it can be loaded (it downloads a small file
on first use). If that is impossible (offline machine) it falls back to a
regex approximation that is within ~10 % of BPE counts for English technical
text. The same counter is used for the naive baseline and for the optimised
system, so comparisons stay fair either way.
"""
from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / not installed
        return None


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _encoder() is not None else "regex-approx"


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # approximation: words + punctuation, long words split every 6 chars
    n = 0
    for tok in _WORD.findall(text):
        n += max(1, (len(tok) + 5) // 6) if tok.isalnum() else 1
    return n


def truncate_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is not None:
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    words = text.split()
    out: list[str] = []
    for w in words:
        out.append(w)
        if count_tokens(" ".join(out)) > max_tokens:
            out.pop()
            break
    return " ".join(out)
