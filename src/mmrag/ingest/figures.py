"""Re-render the figure behind an *image* chunk so the UI can show it.

Ingestion only keeps the OCR text + caption of a figure (that is what gets
embedded), so the pixels are not in the index. This module goes back to the
source PDF / PPTX, finds the same figure region the parser used and returns it
as PNG bytes. No re-ingestion is needed.

Chunk ids look like ``<doc_id>-p<page>-i<k>``: ``page`` is 1-based and ``k`` is
the k-th image chunk on that page, which matches the k-th figure region the
parser kept on that page.
"""
from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from ..utils.log import get_logger

log = get_logger("figures")

_IDX = re.compile(r"-i(\d+)$")


def _find_file(corpus_dir: Path, name: str) -> Path | None:
    p = corpus_dir / name
    if p.exists():
        return p
    hits = list(corpus_dir.rglob(name))          # corpus may have sub-folders
    return hits[0] if hits else None


def _image_index(chunk_id: str) -> int:
    m = _IDX.search(chunk_id)
    return int(m.group(1)) if m else 0


def _pdf_figure(path: Path, page_no: int, k: int, min_px: int, dpi: int, detector: str,
                yolo_weights: str | None) -> bytes | None:
    import pymupdf

    from .layout import LayoutDetector

    doc = pymupdf.open(path)
    if not 1 <= page_no <= doc.page_count:
        return None
    page = doc[page_no - 1]
    layout = LayoutDetector(detector, yolo_weights, dpi)
    regions = layout.figure_regions(page, min_px)
    if regions:
        rg = regions[min(k, len(regions) - 1)]
        pix = page.get_pixmap(dpi=dpi, clip=rg.rect)      # render region exactly as it looks on the page
        if min(pix.width, pix.height) >= 20:
            return pix.tobytes("png")
    # fallback: no region found -> whole page, so the user still sees the figure
    return page.get_pixmap(dpi=dpi).tobytes("png")


def _pptx_figure(path: Path, slide_no: int, k: int, min_px: int) -> bytes | None:
    from PIL import Image
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    from .pptx_parser import _iter_shapes

    prs = Presentation(str(path))
    if not 1 <= slide_no <= len(prs.slides):
        return None
    seen: dict[str, int] = {}
    # count earlier slides' pictures too, so the "repeated logo" rule matches the parser
    for sno, slide in enumerate(prs.slides, start=1):
        pics = []
        for sh in _iter_shapes(slide.shapes):
            if sh.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            blob = sh.image.blob
            h = hashlib.md5(blob).hexdigest()
            seen[h] = seen.get(h, 0) + 1
            if seen[h] > 2:
                continue
            try:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception:
                continue
            if min(img.size) < min_px:
                continue
            buf = io.BytesIO()
            img.save(buf, "PNG")
            pics.append(buf.getvalue())
        if sno == slide_no:
            return pics[min(k, len(pics) - 1)] if pics else None   # charts have no picture -> None
    return None


def figure_png(cfg, chunk) -> bytes | None:
    """PNG bytes of the figure for an image chunk, or None if it cannot be found."""
    if getattr(chunk, "modality", "") != "image":
        return None
    path = _find_file(cfg.path("corpus_dir"), chunk.source)
    if path is None:
        log.warning("figure source not found: %s", chunk.source)
        return None
    k = _image_index(chunk.chunk_id)
    min_px = cfg.get("ingest.ocr.min_image_px", 80)
    try:
        if path.suffix.lower() == ".pdf":
            return _pdf_figure(path, chunk.page, k, min_px, cfg.get("ingest.layout.render_dpi", 150),
                               cfg.get("ingest.layout.detector", "heuristic"), cfg.get("ingest.layout.yolo_weights"))
        if path.suffix.lower() == ".pptx":
            return _pptx_figure(path, chunk.page, k, min_px)
        if path.suffix.lower() == ".ppt":
            conv = cfg.path("work_dir") / "converted" / (path.stem + ".pptx")
            return _pptx_figure(conv, chunk.page, k, min_px) if conv.exists() else None
    except Exception as e:  # never break the UI because of a figure
        log.warning("could not render figure for %s: %s", chunk.chunk_id, e)
    return None
