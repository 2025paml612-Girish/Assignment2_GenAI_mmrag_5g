"""PDF parser: text blocks, tables and images per page.

* Text  : PyMuPDF text blocks in reading order; headings detected by font size;
          blocks lying inside a table are skipped (no duplication).
* Tables: PyMuPDF ``find_tables`` -> rows -> Markdown table (header kept).
* Images: LayoutDetector regions -> original raster bytes (or a clip render for
          vector diagrams) -> OCR text + caption found next to the figure.
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pymupdf
from PIL import Image

from ..schemas import Element
from .layout import LayoutDetector
from .ocr import OCREngine

CAPTION = re.compile(r"^\s*(fig(ure)?|chart|diagram|image)\s*[\d.:-]", re.I)
TABLE_CAPTION = re.compile(r"^\s*table\s*[\d.:-]", re.I)


def rows_to_markdown(rows: list[list]) -> str:
    rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def _table_rows(t, words) -> list[list[str]]:
    """Rebuild cell text from word boxes (keeps spaces that ``Table.extract`` may drop)."""
    rows = []
    for row in t.rows:
        cells = []
        for c in row.cells:
            if c is None:
                cells.append("")
                continue
            r = pymupdf.Rect(c)
            ws = [w for w in words if r.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))]
            ws.sort(key=lambda w: (round(w[1]), w[0]))
            cells.append(" ".join(w[4] for w in ws))
        rows.append(cells)
    if getattr(t.header, "external", False):
        rows.insert(0, list(t.header.names))
    return rows


def _body_font_size(page_dict: dict) -> float:
    sizes: dict[float, int] = {}
    for b in page_dict.get("blocks", []):
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                k = round(s["size"], 1)
                sizes[k] = sizes.get(k, 0) + len(s["text"])
    return max(sizes, key=sizes.get) if sizes else 10.0


def parse_pdf(path: Path, doc_id: str, ocr: OCREngine, layout: LayoutDetector, min_px: int = 80) -> tuple[list[Element], dict]:
    elements: list[Element] = []
    stats = {"pages": 0, "text_blocks": 0, "tables": 0, "images": 0, "images_ocr_chars": 0,
             "images_skipped": 0, "title": "", "date": ""}
    doc = pymupdf.open(path)
    meta = doc.metadata or {}
    stats["title"] = (meta.get("title") or "").strip()
    cd = meta.get("creationDate") or ""
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", cd)
    if m:
        stats["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    seen_images: dict[str, int] = {}
    section = ""
    for pno, page in enumerate(doc, start=1):
        stats["pages"] += 1
        # ---------------- tables ----------------
        table_rects: list[pymupdf.Rect] = []
        try:
            tabs = page.find_tables()
            tables = list(tabs.tables) if hasattr(tabs, "tables") else list(tabs)
        except Exception:
            tables = []
        page_dict = page.get_text("dict")
        blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
        body = _body_font_size(page_dict)

        def block_text(b):
            return " ".join(s["text"] for ln in b["lines"] for s in ln["spans"]).strip()

        words = page.get_text("words")
        for t in tables:
            rows = _table_rows(t, words)
            md = rows_to_markdown(rows)
            if not md or len(rows) < 2:
                continue
            r = pymupdf.Rect(t.bbox)
            table_rects.append(r)
            # caption: text block just above the table
            cap = ""
            for b in blocks:
                br = pymupdf.Rect(b["bbox"])
                if br.y1 <= r.y0 + 2 and r.y0 - br.y1 < 40 and TABLE_CAPTION.match(block_text(b)):
                    cap = block_text(b)
            elements.append(Element(doc_id, path.name, pno, "table",
                                    (cap + "\n" if cap else "") + md, section, tuple(r),
                                    {"n_rows": len(rows) - 1, "caption": cap}))
            stats["tables"] += 1

        # ---------------- figures ----------------
        regions = layout.figure_regions(page, min_px)
        region_rects = [rg.rect for rg in regions]
        for rg in regions:
            img = None
            if rg.kind == "raster" and rg.xref:
                try:
                    raw = doc.extract_image(rg.xref)["image"]
                    h = hashlib.md5(raw).hexdigest()
                    seen_images[h] = seen_images.get(h, 0) + 1
                    if seen_images[h] > 2:        # repeated logo / watermark
                        stats["images_skipped"] += 1
                        continue
                    img = Image.open(io.BytesIO(raw))
                except Exception:
                    img = None
            if img is None:
                img = layout.crop(page, rg)
            if min(img.size) < min_px:
                stats["images_skipped"] += 1
                continue
            if max(img.size) < 900:                 # upscale small images for OCR
                f = 900 / max(img.size)
                img = img.resize((int(img.width * f), int(img.height * f)))
            ocr_text, conf = ocr.read(img)
            caption = _find_caption(blocks, rg.rect, block_text)
            content = f"[Figure] {caption}".strip() + ("\nText in image: " + ocr_text if ocr_text else "")
            elements.append(Element(doc_id, path.name, pno, "image", content, section, tuple(rg.rect),
                                    {"ocr_conf": round(conf, 3), "caption": caption, "region": rg.kind,
                                     "size": img.size}))
            stats["images"] += 1
            stats["images_ocr_chars"] += len(ocr_text)

        # ---------------- text ----------------
        for b in blocks:
            br = pymupdf.Rect(b["bbox"])
            if any((br & tr).get_area() > 0.5 * max(br.get_area(), 1) for tr in table_rects):
                continue
            if any(br in rr for rr in region_rects):      # text drawn inside a figure
                continue
            txt = block_text(b)
            if not txt or len(txt) < 3:
                continue
            max_size = max((s["size"] for ln in b["lines"] for s in ln["spans"]), default=body)
            is_heading = max_size >= body * 1.15 and len(txt) < 120
            if is_heading:
                section = txt
                if not stats["title"] and pno == 1:
                    stats["title"] = txt
            elements.append(Element(doc_id, path.name, pno, "text", txt, section, tuple(br),
                                    {"heading": is_heading}))
            stats["text_blocks"] += 1
    doc.close()
    return elements, stats


def _find_caption(blocks, rect: pymupdf.Rect, block_text) -> str:
    best, best_d = "", 1e9
    for b in blocks:
        br = pymupdf.Rect(b["bbox"])
        t = block_text(b)
        if not t:
            continue
        d_below = br.y0 - rect.y1
        d_above = rect.y0 - br.y1
        d = min(abs(d_below) if d_below >= -2 else 1e9, abs(d_above) if d_above >= -2 else 1e9)
        if d < 45 and (CAPTION.match(t) or d < best_d):
            if CAPTION.match(t):
                return t
            best, best_d = t, d
    return best if best_d < 25 else ""
