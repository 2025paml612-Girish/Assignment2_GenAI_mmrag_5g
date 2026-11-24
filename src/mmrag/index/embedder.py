"""Embedding models.

* ``sentence-transformers`` : BAAI/bge-small-en-v1.5 (default, 384-d, strong on
  retrieval benchmarks, fast on CPU) or all-MiniLM-L6-v2 (E2 comparison).
* ``tfidf`` : offline fallback - TF-IDF (word 1-2 grams) + TruncatedSVD (LSA).
  Used automatically when sentence-transformers is missing or the model
  cannot be downloaded, so the pipeline always runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..utils.log import get_logger

log = get_logger("embedder")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, backend: str = "auto", model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32,
                 normalize: bool = True):
        self.requested = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.backend = None
        self.model = None
        self._tfidf = None
        self._svd = None
        if backend in ("auto", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model, device="cpu")
                self.backend = "sentence-transformers"
            except Exception as e:
                if backend == "sentence-transformers":
                    raise
                log.warning("sentence-transformers model '%s' unavailable (%s: %s) -> TF-IDF/LSA fallback",
                            model, type(e).__name__, str(e)[:80])
        if self.backend is None:
            self.backend = "tfidf"
        log.info("Embedding backend: %s", self.name)

    @property
    def name(self) -> str:
        return f"st:{self.requested}" if self.backend == "sentence-transformers" else f"tfidf-lsa({self.requested_dims()})"

    def requested_dims(self) -> int:
        return getattr(self._svd, "n_components", 256) if self._svd is not None else 256

    # ------------------------------------------------------------------
    def fit(self, texts: list[str], dims: int = 256) -> None:
        """Only the TF-IDF backend needs fitting on the corpus."""
        if self.backend != "tfidf":
            return
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                      token_pattern=r"(?u)\b[\w\-\.]+\b", lowercase=True)
        X = self._tfidf.fit_transform(texts)
        n = max(2, min(dims, X.shape[1] - 1, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=42)
        self._svd.fit(X)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v.astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (v / n).astype(np.float32)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self.backend == "sentence-transformers":
            if is_query and "bge" in self.requested.lower():
                texts = [BGE_QUERY_PREFIX + t for t in texts]
            v = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=False)
            return self._norm(v)
        if self._tfidf is None:
            raise RuntimeError("TF-IDF embedder not fitted")
        return self._norm(self._svd.transform(self._tfidf.transform(texts)))

    # persistence --------------------------------------------------------
    def save(self, path: Path) -> None:
        state = {"backend": self.backend, "model": self.requested}
        if self.backend == "tfidf":
            state.update(tfidf=self._tfidf, svd=self._svd)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path, batch_size: int = 32) -> "Embedder":
        with open(path, "rb") as f:
            st = pickle.load(f)
        if st["backend"] == "tfidf":
            e = cls.__new__(cls)
            e.requested, e.batch_size, e.normalize = st["model"], batch_size, True
            e.backend, e.model = "tfidf", None
            e._tfidf, e._svd = st["tfidf"], st["svd"]
            return e
        return cls("sentence-transformers", st["model"], batch_size)
