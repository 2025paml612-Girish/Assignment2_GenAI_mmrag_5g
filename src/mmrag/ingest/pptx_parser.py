"""PPTX parser: slide title, text frames, tables, pictures, charts and notes.

* Charts are read from their embedded data (categories + series values), which is
  exact and costs far fewer tokens than OCR-ing a rendered chart.
* Pictures are OCR-ed; the slide title is used as the caption.
* Legacy .ppt files are converted to .pptx with LibreOffice (if installed).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image
import io

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from ..schemas import Element
from ..utils.log import get_logger
from .ocr import OCREngine
from .pdf_parser import rows_to_markdown

log = get_logger("pptx")


def convert_ppt(path: Path, out_dir: Path) -> Path | None:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        log.warning("Cannot convert %s: LibreOffice not found", path.name)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "--headless", "--convert-to", "pptx", "--outdir", str(out_dir), str(path)],
                   capture_output=True, timeout=180)
    out = out_dir / (path.stem + ".pptx")
    return out if out.exists() else None


def _iter_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(sh.shapes)
        else:
            yield sh


def parse_pptx(path: Path, doc_id: str, ocr: OCREngine, min_px: int = 80) -> tuple[list[Element], dict]:
    prs = Presentation(str(path))
    stats = {"pages": 0, "text_blocks": 0, "tables": 0, "images": 0, "images_ocr_chars": 0,
             "images_skipped": 0, "charts": 0, "title": "", "date": ""}
    cp = prs.core_properties
    stats["title"] = (cp.title or "").strip()
    if cp.created:
        stats["date"] = cp.created.strftime("%Y-%m-%d")
    elements: list[Element] = []
    seen: dict[str, int] = {}
    for sno, slide in enumerate(prs.slides, start=1):
        stats["pages"] += 1
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip()
        if sno == 1 and not stats["title"]:
            stats["title"] = title
        section = title
        for sh in _iter_shapes(slide.shapes):
            if getattr(sh, "has_table", False) and sh.has_table:
                rows = [[c.text for c in r.cells] for r in sh.table.rows]
                md = rows_to_markdown(rows)
                if md:
                    elements.append(Element(doc_id, path.name, sno, "table", (title if title.lower().startswith("table") else f"Table: {title}") + "\n" + md, section,
                                            None, {"n_rows": len(rows) - 1, "caption": title}))
                    stats["tables"] += 1
            elif getattr(sh, "has_chart", False) and sh.has_chart:
                txt = _chart_to_text(sh.chart, title)
                elements.append(Element(doc_id, path.name, sno, "image", txt, section, None,
                                        {"region": "chart-data", "caption": title}))
                stats["images"] += 1
                stats["charts"] += 1
            elif sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                blob = sh.image.blob
                h = hashlib.md5(blob).hexdigest()
                seen[h] = seen.get(h, 0) + 1
                if seen[h] > 2:
                    stats["images_skipped"] += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(blob)).convert("RGB")
                except Exception:
                    stats["images_skipped"] += 1
                    continue
                if min(img.size) < min_px:
                    stats["images_skipped"] += 1
                    continue
                if max(img.size) < 900:
                    f = 900 / max(img.size)
                    img = img.resize((int(img.width * f), int(img.height * f)))
                text, conf = ocr.read(img)
                alt = ""
                try:
                    alt = sh._element.nvPicPr.cNvPr.get("descr") or ""
                except Exception:
                    pass
                cap = title if (not alt or re.search(r"\.(png|jpe?g|gif|bmp|emf|wmf)$", alt, re.I)) else alt
                content = f"[Figure] {cap}".strip() + ("\nText in image: " + text if text else "")
                elements.append(Element(doc_id, path.name, sno, "image", content, section, None,
                                        {"ocr_conf": round(conf, 3), "caption": cap, "region": "picture"}))
                stats["images"] += 1
                stats["images_ocr_chars"] += len(text)
            elif sh.has_text_frame:
                txt = "\n".join(p.text.strip() for p in sh.text_frame.paragraphs if p.text.strip())
                if txt and txt != title:
                    elements.append(Element(doc_id, path.name, sno, "text", txt, section))
                    stats["text_blocks"] += 1
        if title:
            elements.append(Element(doc_id, path.name, sno, "text", title, section, None, {"heading": True}))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                elements.append(Element(doc_id, path.name, sno, "text", "Speaker notes: " + notes, section))
                stats["text_blocks"] += 1
    return elements, stats


def _chart_to_text(chart, title: str) -> str:
    lines = [f"[Chart] {title}".strip()]
    try:
        if chart.has_title and chart.chart_title.has_text_frame:
            lines.append("Chart title: " + chart.chart_title.text_frame.text)
    except Exception:
        pass
    try:
        cats = list(chart.plots[0].categories)
        for s in chart.plots[0].series:
            vals = ", ".join(f"{c}={v}" for c, v in zip(cats, s.values))
            lines.append(f"Series {s.name}: {vals}")
    except Exception:
        pass
    return "\n".join(lines)
