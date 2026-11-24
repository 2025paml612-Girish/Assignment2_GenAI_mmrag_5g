"""Configuration loader.

Reads ``config/config.yaml`` and ``config/ontology.yaml``, expands ``${ENV_VARS}``
and exposes a small dict-like object with dotted access: ``cfg.get("retrieval.top_k")``.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]          # project root (folder with pyproject.toml)
_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (no extra dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    return obj


class Config:
    def __init__(self, data: dict, ontology: dict, root: Path = ROOT):
        self.data = data
        self.ontology = ontology
        self.root = root

    # dotted getter -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self.data
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def path(self, key: str) -> Path:
        """Resolve a key under ``paths`` relative to the project root."""
        p = Path(self.get(f"paths.{key}"))
        return p if p.is_absolute() else self.root / p

    def clone(self, **overrides: Any) -> "Config":
        c = Config(copy.deepcopy(self.data), self.ontology, self.root)
        for k, v in overrides.items():
            c.set(k.replace("__", "."), v)
        return c


def load_config(config_path: str | Path | None = None) -> Config:
    _load_dotenv(ROOT / ".env")
    cpath = Path(config_path) if config_path else ROOT / "config" / "config.yaml"
    data = _expand(yaml.safe_load(cpath.read_text(encoding="utf-8")))
    onto = yaml.safe_load((ROOT / "config" / "ontology.yaml").read_text(encoding="utf-8"))
    return Config(data, onto)
