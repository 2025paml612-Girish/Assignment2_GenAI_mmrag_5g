"""Ingestion orchestrator (Task A1 / Experiment E1).

corpus dir --> parse (PDF / PPTX / PPT) --> elements.jsonl (cached)
           --> chunk + ontology metadata --> build index
Also writes the preprocessing record and the corpus inventory.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..config import Config
from ..schemas import Chunk, Element
from ..utils.log import get_logger
from .chunker import chunk_document
from .layout import LayoutDetector
from .metadata import OntologyTagger
from .ocr import get_ocr
from .pdf_parser import parse_pdf
from .pptx_parser import convert_ppt, parse_pptx

log = get_logger("ingest")
SUPPORTED = {".pdf", ".pptx", ".ppt"}


def _doc_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path.stem)[:48].strip("_")


def parse_corpus(cfg: Config, force: bool = False) -> tuple[dict[str, list[Element]], list[dict]]:
    """Parse every document once; results cached in work_dir/elements.jsonl."""
    corpus = cfg.path("corpus_dir")
    work = cfg.path("work_dir")
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "elements.jsonl"
    record_file = work / "preprocessing_record.jsonl"
    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in SUPPORTED and not p.name.startswith("~$"))
    sig = str(len(files)) + "|" + "|".join(f"{p.name}:{p.stat().st_size}" for p in files)
    sig_file = work / "corpus_signature.txt"
    if not force and cache.exists() and sig_file.exists() and sig_file.read_text() == sig:
        log.info("Using cached parse of %d documents (%s)", len(files), cache)
        by_doc: dict[str, list[Element]] = {}
        for line in cache.open(encoding="utf-8"):
            d = json.loads(line)
            d["bbox"] = tuple(d["bbox"]) if d["bbox"] else None
            by_doc.setdefault(d["doc_id"], []).append(Element(**d))
        records = [json.loads(l) for l in record_file.open(encoding="utf-8")]
        return by_doc, records

    if len(files) < cfg.get("ingest.min_documents", 100):
        log.warning("Only %d documents found in %s (FR1 requires >= %d)", len(files), corpus,
                    cfg.get("ingest.min_documents"))
    ocr = get_ocr(cfg.get("ingest.ocr.engine", "auto"), cfg.get("ingest.ocr.min_confidence", 0.5))
    layout = LayoutDetector(cfg.get("ingest.layout.detector", "heuristic"), cfg.get("ingest.layout.yolo_weights"),
                            cfg.get("ingest.layout.render_dpi", 150))
    min_px = cfg.get("ingest.ocr.min_image_px", 80)
    by_doc, records = {}, []
    for i, path in enumerate(files, 1):
        doc_id = _doc_id(path)
        t0 = time.time()
        rec = {"doc_id": doc_id, "file": str(path.relative_to(corpus)), "type": path.suffix.lower().lstrip("."),
               "bytes": path.stat().st_size, "status": "ok", "error": "", "ocr_engine": ocr.name,
               "layout": layout.detector}
        try:
            src = path
            if path.suffix.lower() == ".ppt":
                src = convert_ppt(path, work / "converted") if cfg.get("ingest.convert_legacy_ppt") else None
                if src is None:
                    raise RuntimeError("legacy .ppt could not be converted")
            if src.suffix.lower() == ".pdf":
                els, st = parse_pdf(src, doc_id, ocr, layout, min_px)
            else:
                els, st = parse_pptx(src, doc_id, ocr, min_px)
            rec.update(st)
            by_doc[doc_id] = els
        except Exception as e:  # one bad file must not stop ingestion
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
            log.error("Failed to parse %s: %s", path.name, e)
        rec["parse_seconds"] = round(time.time() - t0, 3)
        records.append(rec)
        if i % 20 == 0 or i == len(files):
            log.info("Parsed %d/%d documents", i, len(files))
    with open(cache, "w", encoding="utf-8") as f:
        for els in by_doc.values():
            for e in els:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    with open(record_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sig_file.write_text(sig)
    return by_doc, records


def make_chunks(cfg: Config, by_doc: dict[str, list[Element]], records: list[dict], strategy: str | None = None) -> list[Chunk]:
    tagger = OntologyTagger(cfg.ontology)
    strategy = strategy or cfg.get("chunking.strategy", "recursive")
    rec_by_id = {r["doc_id"]: r for r in records}
    chunks: list[Chunk] = []
    for doc_id, els in by_doc.items():
        r = rec_by_id.get(doc_id, {})
        title = r.get("title") or doc_id.replace("_", " ")
        full = " ".join(e.content for e in els[:60])
        m = re.search(r"(?:document )?date:?\s*(\d{4}-\d{2}-\d{2})", full, re.I)
        if m:
            r["date"] = m.group(1)
        doc_meta = {"doc_id": doc_id, "source": Path(r.get("file", doc_id)).name, "title": title,
                    "date": r.get("date", ""), "doc_type": tagger.doc_type(title, full)}
        r["doc_type"] = doc_meta["doc_type"]
        chunks.extend(chunk_document(els, doc_meta, tagger, strategy, cfg.get("chunking.chunk_tokens", 220),
                                     cfg.get("chunking.overlap_tokens", 40), cfg.get("chunking.table_rows_per_chunk", 12)))
    return chunks


def write_inventory(cfg: Config, records: list[dict], chunks: list[Chunk]) -> dict:
    out = cfg.path("outputs_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    per_doc = Counter(c.doc_id for c in chunks)
    df["chunks"] = df["doc_id"].map(per_doc).fillna(0).astype(int)
    df.to_csv(out / "e1_corpus_inventory.csv", index=False)
    ok = df[df.status == "ok"]
    mod = Counter(c.modality for c in chunks)
    stats = {
        "documents_total": int(len(df)),
        "documents_parsed": int(len(ok)),
        "documents_failed": int((df.status != "ok").sum()),
        "pdf": int((ok.type == "pdf").sum()),
        "pptx_ppt": int(ok.type.isin(["pptx", "ppt"]).sum()),
        "pages_or_slides": int(ok.get("pages", pd.Series(dtype=int)).fillna(0).sum()),
        "text_blocks": int(ok.get("text_blocks", pd.Series(dtype=int)).fillna(0).sum()),
        "tables": int(ok.get("tables", pd.Series(dtype=int)).fillna(0).sum()),
        "images": int(ok.get("images", pd.Series(dtype=int)).fillna(0).sum()),
        "images_skipped_small_or_repeated": int(ok.get("images_skipped", pd.Series(dtype=int)).fillna(0).sum()),
        "ocr_characters": int(ok.get("images_ocr_chars", pd.Series(dtype=int)).fillna(0).sum()),
        "docs_with_tables": int((ok.get("tables", 0) > 0).sum()),
        "docs_with_images": int((ok.get("images", 0) > 0).sum()),
        "chunks_total": len(chunks),
        "chunks_text": mod.get("text", 0), "chunks_table": mod.get("table", 0), "chunks_image": mod.get("image", 0),
        "avg_chunk_tokens": round(sum(c.tokens for c in chunks) / max(1, len(chunks)), 1),
        "doc_types": dict(Counter(df.get("doc_type", pd.Series(dtype=str)).dropna())),
        "parse_seconds_total": round(float(df.parse_seconds.sum()), 1),
        "ocr_engine": records[0].get("ocr_engine") if records else "",
        "fr1_satisfied": bool(len(ok) >= cfg.get("ingest.min_documents", 100)),
    }
    (out / "e1_ingestion_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def run_ingest(cfg: Config, force: bool = False):
    from ..index import build_index

    by_doc, records = parse_corpus(cfg, force=force)
    chunks = make_chunks(cfg, by_doc, records)
    stats = write_inventory(cfg, records, chunks)
    idx = build_index(chunks, cfg.path("index_dir"), cfg.get("embedding.backend"), cfg.get("embedding.model"),
                      cfg.get("vector_store.backend"), cfg.get("vector_store.collection"),
                      cfg.get("embedding.batch_size", 32))
    stats["index"] = idx.manifest
    (cfg.path("outputs_dir") / "e1_ingestion_stats.json").write_text(json.dumps(stats, indent=2))
    return idx, stats
