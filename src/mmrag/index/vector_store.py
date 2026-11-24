"""Vector stores with one interface: add / query(vector, k, where) / count.

* ChromaStore : persistent ChromaDB collection, cosine HNSW, metadata ``where`` filters.
* NumpyStore  : exact cosine search over a float32 matrix + metadata list
                (no extra dependency, fully deterministic). Used when ChromaDB is
                not installed, and for fast in-memory E2 experiments.

``where`` is a backend-neutral dict ``{field: [allowed values]}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils.log import get_logger

log = get_logger("vector_store")


class NumpyStore:
    name = "numpy"

    def __init__(self, path: Path | None = None):
        self.path = path
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.mat: np.ndarray | None = None

    def add(self, ids, vectors, metadatas):
        self.ids.extend(ids)
        self.meta.extend(metadatas)
        v = np.asarray(vectors, dtype=np.float32)
        self.mat = v if self.mat is None else np.vstack([self.mat, v])

    def count(self) -> int:
        return len(self.ids)

    def _mask(self, where: dict | None) -> np.ndarray:
        m = np.ones(len(self.ids), dtype=bool)
        if where:
            for field, allowed in where.items():
                allowed = set(allowed)
                m &= np.array([md.get(field) in allowed for md in self.meta])
        return m

    def query(self, vector, k: int, where: dict | None = None) -> list[tuple[str, float]]:
        if self.mat is None or not len(self.ids):
            return []
        sims = self.mat @ np.asarray(vector, dtype=np.float32)
        mask = self._mask(where)
        sims = np.where(mask, sims, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        res = sorted(((self.ids[i], float(sims[i])) for i in idx), key=lambda t: (-t[1], t[0]))
        return res

    def persist(self):
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self.mat)
        (self.path / "vector_ids.json").write_text(json.dumps({"ids": self.ids, "meta": self.meta}))

    @classmethod
    def load(cls, path: Path) -> "NumpyStore":
        s = cls(path)
        d = json.loads((path / "vector_ids.json").read_text())
        s.ids, s.meta = d["ids"], d["meta"]
        s.mat = np.load(path / "vectors.npy")
        return s


class ChromaStore:
    name = "chroma"

    def __init__(self, path: Path, collection: str, reset: bool = False):
        import chromadb
        from chromadb.config import Settings

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path / "chroma"),
                                                settings=Settings(anonymized_telemetry=False))
        if reset:
            try:
                self.client.delete_collection(collection)
            except Exception:
                pass
        self.col = self.client.get_or_create_collection(collection, embedding_function=None,
                                                        metadata={"hnsw:space": "cosine"})

    def add(self, ids, vectors, metadatas):
        vectors = np.asarray(vectors, dtype=np.float32)
        clean = [{k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in m.items()} for m in metadatas]
        B = 500
        for i in range(0, len(ids), B):
            self.col.add(ids=ids[i:i + B], embeddings=vectors[i:i + B].tolist(), metadatas=clean[i:i + B])

    def count(self) -> int:
        return self.col.count()

    @staticmethod
    def _where(where: dict | None):
        if not where:
            return None
        conds = [{f: {"$in": list(v)}} for f, v in where.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def query(self, vector, k: int, where: dict | None = None) -> list[tuple[str, float]]:
        n = self.count()
        if n == 0:
            return []
        try:
            r = self.col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                               n_results=min(k, n), where=self._where(where), include=["distances"])
        except Exception as e:      # e.g. filter matches nothing
            log.debug("chroma query failed: %s", e)
            return []
        ids, dists = r["ids"][0], r["distances"][0]
        return sorted(((i, 1.0 - float(d)) for i, d in zip(ids, dists)), key=lambda t: (-t[1], t[0]))

    def persist(self):  # PersistentClient writes through
        pass


def make_store(backend: str, path: Path, collection: str, reset: bool = False):
    if backend in ("auto", "chroma"):
        try:
            return ChromaStore(path, collection, reset=reset)
        except Exception as e:
            if backend == "chroma":
                raise
            log.warning("ChromaDB unavailable (%s) -> numpy vector store", e)
    return NumpyStore(path)
