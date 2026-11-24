"""OCR engine wrapper.

Order of preference (``ingest.ocr.engine: auto``):
  1. RapidOCR (onnxruntime, pip-only, works on Windows/Linux/macOS without system installs)
  2. Tesseract through pytesseract (needs the tesseract binary on PATH)
  3. none  -> images are indexed by caption / surrounding text only

OCR turns text inside images into compact text, so the LLM never needs the raw
image (token optimisation, NFR2).
"""
from __future__ import annotations

import io
import re
import shutil
from functools import lru_cache

import numpy as np
from PIL import Image

from ..utils.log import get_logger

log = get_logger("ocr")


class OCREngine:
    def __init__(self, engine: str = "auto", min_confidence: float = 0.5):
        self.min_conf = min_confidence
        self.name = "none"
        self._impl = None
        if engine in ("auto", "rapidocr"):
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._impl = RapidOCR()
                self.name = "rapidocr"
            except Exception as e:  # pragma: no cover - depends on env
                if engine == "rapidocr":
                    log.warning("RapidOCR unavailable (%s)", e)
        if self._impl is None and engine in ("auto", "tesseract"):
            if shutil.which("tesseract"):
                try:
                    import pytesseract  # noqa: F401

                    self.name = "tesseract"
                except Exception:
                    pass
        log.info("OCR engine: %s", self.name)

    # ------------------------------------------------------------------
    def read(self, img: Image.Image | bytes) -> tuple[str, float]:
        """Return (text, mean_confidence). Lines are ordered top-to-bottom, left-to-right."""
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        if self.name == "rapidocr":
            return self._rapid(img)
        if self.name == "tesseract":
            return self._tess(img)
        return "", 0.0

    def _rapid(self, img: Image.Image) -> tuple[str, float]:
        result, _ = self._impl(np.array(img))
        if not result:
            return "", 0.0
        items = []
        for box, text, conf in result:
            conf = float(conf)
            if conf < self.min_conf or not text.strip():
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((sum(ys) / 4, min(xs), max(ys) - min(ys), text.strip(), conf))
        return _join_lines(items)

    def _tess(self, img: Image.Image) -> tuple[str, float]:
        import pytesseract

        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        items = []
        for i, word in enumerate(d["text"]):
            conf = float(d["conf"][i]) / 100.0 if d["conf"][i] not in ("-1", -1) else 0.0
            if not word.strip() or conf < self.min_conf:
                continue
            items.append((d["top"][i] + d["height"][i] / 2, d["left"][i], d["height"][i], word, conf))
        return _join_lines(items)


def repair_spacing(text: str) -> str:
    """OCR often glues words: 'FronthaulVLAN2180' -> 'Fronthaul VLAN 2180'.
    Short identifiers such as n78, 180A or R24.3 are left untouched."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z]{3})(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z]{3})", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _join_lines(items: list[tuple]) -> tuple[str, float]:
    """Group OCR boxes into reading-order lines."""
    if not items:
        return "", 0.0
    items.sort(key=lambda t: (t[0], t[1]))
    lines: list[list[tuple]] = []
    for it in items:
        if lines and abs(it[0] - lines[-1][0][0]) < max(6.0, 0.5 * it[2]):
            lines[-1].append(it)
        else:
            lines.append([it])
    text = "\n".join(" ".join(t[3] for t in sorted(line, key=lambda t: t[1])) for line in lines)
    text = repair_spacing(text)
    conf = float(np.mean([t[4] for t in items]))
    return text, conf


@lru_cache(maxsize=4)
def get_ocr(engine: str = "auto", min_confidence: float = 0.5) -> OCREngine:
    return OCREngine(engine, min_confidence)
