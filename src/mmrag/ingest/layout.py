"""Layout / region detection.

Finds the regions of a page that hold figures so that only those regions are
cropped and OCR-ed (instead of OCR-ing or sending whole pages).

* ``heuristic`` (default): raster image rectangles reported by PyMuPDF, plus
  clusters of vector drawings (diagrams drawn with lines/boxes) that cover a
  meaningful part of the page.
* ``yolo``: a YOLO document-layout model (e.g. DocLayout-YOLO or a YOLOv8 model
  fine-tuned on DocLayNet). Boxes with class figure/picture/chart are returned.
  Enable with ``ingest.layout.detector: yolo`` and ``ingest.layout.yolo_weights``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from ..utils.log import get_logger

log = get_logger("layout")

FIGURE_CLASSES = {"figure", "picture", "image", "chart", "diagram"}


@dataclass
class Region:
    rect: pymupdf.Rect
    kind: str            # raster | vector | yolo
    xref: int = 0        # image xref for raster images (0 otherwise)
    score: float = 1.0


class LayoutDetector:
    def __init__(self, detector: str = "heuristic", yolo_weights: str | None = None, dpi: int = 150):
        self.detector = detector
        self.dpi = dpi
        self.model = None
        if detector == "yolo":
            try:
                from ultralytics import YOLO

                if not yolo_weights:
                    raise ValueError("ingest.layout.yolo_weights is not set")
                self.model = YOLO(yolo_weights)
                log.info("YOLO layout model loaded: %s", yolo_weights)
            except Exception as e:
                log.warning("YOLO unavailable (%s) - falling back to heuristic layout", e)
                self.detector = "heuristic"

    def figure_regions(self, page: pymupdf.Page, min_px: int = 80) -> list[Region]:
        if self.detector == "yolo" and self.model is not None:
            return self._yolo(page)
        return self._heuristic(page, min_px)

    # ------------------------------------------------------------------
    def _heuristic(self, page: pymupdf.Page, min_px: int) -> list[Region]:
        regions: list[Region] = []
        page_area = abs(page.rect)
        for info in page.get_images(full=True):
            xref = info[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for r in rects:
                if r.width < min_px * 0.5 or r.height < min_px * 0.5:   # icons / bullets
                    continue
                regions.append(Region(rect=r, kind="raster", xref=xref))
        # vector diagrams: union of drawing rects not overlapping a raster image
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        boxes = [d["rect"] for d in drawings if d.get("rect") and d["rect"].width > 2 and d["rect"].height > 2]
        if len(boxes) >= 6:
            u = pymupdf.Rect(boxes[0])
            for b in boxes[1:]:
                u |= b
            covered = any((u & r.rect).get_area() > 0.5 * u.get_area() for r in regions)
            if not covered and 0.08 * page_area < u.get_area() < 0.9 * page_area:
                regions.append(Region(rect=u, kind="vector"))
        return regions

    def _yolo(self, page: pymupdf.Page) -> list[Region]:
        from PIL import Image

        pix = page.get_pixmap(dpi=self.dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        scale = 72.0 / self.dpi
        res = self.model.predict(img, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            name = str(res.names[int(cls)]).lower()
            if name in FIGURE_CLASSES and conf >= 0.3:
                x0, y0, x1, y1 = (v * scale for v in box)
                out.append(Region(rect=pymupdf.Rect(x0, y0, x1, y1), kind="yolo", score=float(conf)))
        return out

    def crop(self, page: pymupdf.Page, region: Region):
        """Render only the region (cheaper and cleaner than the full page)."""
        from PIL import Image

        pix = page.get_pixmap(dpi=self.dpi, clip=region.rect)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
