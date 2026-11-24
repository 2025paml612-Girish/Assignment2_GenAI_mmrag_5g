"""Index = chunk store + embedder + vector store + BM25 (+ manifest)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..schemas import Chunk
from ..utils.log import get_logger
from .bm25 import BM25Index
from .embedder import Embedder
from .vector_store import NumpyStore, make_store

log = get_logger("index")


def embed_text(c: Chunk) -> str:
    """Text that is embedded: contextual header (title/section/modality) + content."""
    return f"{c.title} | {c.section} | {c.modality}: {c.text}"


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, store, bm25: BM25Index, manifest: dict):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.order = [c.chunk_id for c in chunks]
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.manifest = manifest

    @property
    def version(self) -> str:
        return self.manifest.get("version", "0")

    @classmethod
    def load(cls, index_dir: Path, store_backend: str = "auto", collection: str = "mmrag_5g") -> "Index":
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index at {index_dir}. Run `uv run mmrag ingest` first.")
        chunks = [Chunk.from_dict(json.loads(l)) for l in (index_dir / "chunks.jsonl").open(encoding="utf-8")]
        manifest = json.loads((index_dir / "manifest.json").read_text())
        emb = Embedder.load(index_dir / "embedder.pkl")
        if manifest.get("store") == "chroma" and store_backend in ("auto", "chroma"):
            store = make_store("chroma", index_dir, collection)
        else:
            store = NumpyStore.load(index_dir)
        return cls(chunks, emb, store, BM25Index.load(index_dir / "bm25.pkl"), manifest)


def build_index(chunks: list[Chunk], index_dir: Path, embed_backend: str = "auto", model: str = "BAAI/bge-small-en-v1.5",
                store_backend: str = "auto", collection: str = "mmrag_5g", batch_size: int = 32,
                persist: bool = True, dims: int = 256) -> Index:
    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [embed_text(c) for c in chunks]
    emb = Embedder(embed_backend, model, batch_size)
    emb.fit(texts, dims)
    vecs = emb.encode(texts)
    ids = [c.chunk_id for c in chunks]
    metas = [c.meta() for c in chunks]
    if persist:
        store = make_store(store_backend, index_dir, collection, reset=True)
    else:
        store = NumpyStore(None)
    store.add(ids, vecs, metas)
    # numpy copy is always kept: deterministic fallback + E2/E3 experiments
    np_store = store if isinstance(store, NumpyStore) else NumpyStore(index_dir)
    if np_store is not store:
        np_store.add(ids, vecs, metas)
    bm25 = BM25Index(ids, [c.text + " " + c.title + " " + c.section for c in chunks])
    version = hashlib.md5(("|".join(ids) + emb.name).encode()).hexdigest()[:10]
    manifest = {"version": version, "n_chunks": len(chunks), "embedder": emb.name, "store": store.name,
                "dims": int(vecs.shape[1]), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "build_seconds": round(time.time() - t0, 2)}
    if persist:
        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        emb.save(index_dir / "embedder.pkl")
        np_store.persist()
        bm25.save(index_dir / "bm25.pkl")
        (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Index built: %d chunks, %s, store=%s, %.1fs", len(chunks), emb.name, store.name, manifest["build_seconds"])
    return Index(chunks, emb, store, bm25, manifest)
