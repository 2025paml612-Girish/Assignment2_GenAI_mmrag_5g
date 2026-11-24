"""Long-term answer cache (SQLite).

Key = sha1(normalised standalone query | prompt variant | index version | llm name).
A repeated question returns the stored grounded answer with zero LLM tokens.
The index version is part of the key, so re-ingesting invalidates old entries.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path


class AnswerCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, payload TEXT, created REAL, hits INTEGER)")
        self.lock = threading.Lock()

    @staticmethod
    def key(query: str, variant: str, index_version: str, llm: str) -> str:
        q = re.sub(r"\s+", " ", query.lower()).strip(" ?.")
        return hashlib.sha1(f"{q}|{variant}|{index_version}|{llm}".encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT payload FROM answers WHERE key=?", (key,)).fetchone()
            if row:
                self.conn.execute("UPDATE answers SET hits = hits + 1 WHERE key=?", (key,))
                self.conn.commit()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO answers VALUES (?,?,?,0)", (key, json.dumps(payload), time.time()))
            self.conn.commit()

    def clear(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM answers")
            self.conn.commit()
