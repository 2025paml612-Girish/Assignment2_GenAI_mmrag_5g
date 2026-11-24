"""Small reportlab helpers shared by build_pdfs.py."""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "docs" / "img"
NAVY = colors.HexColor("#1f3b5c")
LIGHT = colors.HexColor("#eef3f8")

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=NAVY),
    "subtitle": ParagraphStyle("st", parent=_ss["Normal"], fontSize=12, leading=16, alignment=TA_CENTER,
                               textColor=colors.HexColor("#444444")),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19,
                         textColor=NAVY, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                         textColor=NAVY, spaceBefore=8, spaceAfter=4),
    "h3": ParagraphStyle("h3", parent=_ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                         spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("b", parent=_ss["BodyText"], fontName="Helvetica", fontSize=10, leading=13.6, spaceAfter=4),
    "small": ParagraphStyle("sm", parent=_ss["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11),
    "cell": ParagraphStyle("c", parent=_ss["BodyText"], fontName="Helvetica", fontSize=8, leading=10),
    "cellb": ParagraphStyle("cb", parent=_ss["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10,
                            textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=9.4, backColor=colors.HexColor("#f5f5f5"),
                           borderColor=colors.HexColor("#dddddd"), borderWidth=0.5, borderPadding=5, spaceBefore=3,
                           spaceAfter=7, leftIndent=4, rightIndent=4),
    "caption": ParagraphStyle("cap", parent=_ss["Italic"], fontSize=8.5, leading=11, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555555"), spaceAfter=8),
    "note": ParagraphStyle("n", parent=_ss["BodyText"], fontSize=9, leading=12, backColor=colors.HexColor("#fff7e0"),
                           borderColor=colors.HexColor("#e0c068"), borderWidth=0.5, borderPadding=5, spaceBefore=4,
                           spaceAfter=8),
}
S["toc1"] = ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10, leading=14, leftIndent=0)
S["toc2"] = ParagraphStyle("toc2", fontName="Helvetica", fontSize=9, leading=12, leftIndent=14)


class Doc(BaseDocTemplate):
    def __init__(self, path: Path, title: str):
        super().__init__(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.8 * cm,
                         bottomMargin=1.7 * cm, title=title, author="mmrag-5g project")
        self.doc_title = title
        pw, ph = A4
        fp = Frame(self.leftMargin, self.bottomMargin, pw - 3.6 * cm, ph - 3.5 * cm, id="p")
        lw, lh = landscape(A4)
        fl = Frame(1.5 * cm, 1.5 * cm, lw - 3 * cm, lh - 3.2 * cm, id="l")
        self.addPageTemplates([PageTemplate("portrait", [fp], onPage=self._deco, pagesize=A4),
                               PageTemplate("landscape", [fl], onPage=self._deco, pagesize=landscape(A4))])

    def _deco(self, canv, doc):
        w, h = canv._pagesize
        canv.saveState()
        canv.setStrokeColor(NAVY)
        canv.setLineWidth(0.6)
        canv.line(1.5 * cm, h - 1.3 * cm, w - 1.5 * cm, h - 1.3 * cm)
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(colors.HexColor("#555555"))
        canv.drawString(1.5 * cm, h - 1.15 * cm, "PCAM ZG516 - Assignment 2 - Part A: Multimodal Document Retrieval System")
        canv.drawRightString(w - 1.5 * cm, h - 1.15 * cm, self.doc_title)
        canv.drawRightString(w - 1.5 * cm, 0.9 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            lvl = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = f"k{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=lvl, closed=lvl > 0)
            self.notify("TOCEntry", (lvl, text, self.page, key))


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(t):
    return [CondPageBreak(5 * cm), Paragraph(t, S["h1"])]


def H2(t):
    return [CondPageBreak(3.5 * cm), Paragraph(t, S["h2"])]


def H3(t):
    return Paragraph(t, S["h3"])


def bullets(items, style="body"):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", ParagraphStyle("bl", parent=S[style], leftIndent=12, firstLineIndent=-9))
            for i in items]


def code(text: str, width: int = 108):
    lines = []
    for ln in textwrap.dedent(text).strip("\n").splitlines():
        if len(ln) <= width:
            lines.append(ln)
        else:
            ind = len(ln) - len(ln.lstrip())
            parts = textwrap.wrap(ln.strip(), max(30, width - ind), subsequent_indent="    ") or [ln.strip()]
            lines += [" " * ind + p for p in parts]
    return XPreformatted(escape("\n".join(lines)), S["code"])


def note(t):
    return Paragraph(t, S["note"])


def table(rows, widths=None, font=8, header=True, zebra=True):
    st_c = ParagraphStyle("cx", parent=S["cell"], fontSize=font, leading=font + 2)
    st_h = ParagraphStyle("chx", parent=S["cellb"], fontSize=font, leading=font + 2)
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c) if isinstance(c, str) and c.startswith("<") else escape(str(c)),
                               st_h if (header and i == 0) else st_c) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if header else 0, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def df_table(df: pd.DataFrame, cols=None, rename=None, widths=None, font=7.5, fmt="{:.3f}"):
    df = df[cols] if cols else df
    rows = [[(rename or {}).get(c, c) for c in df.columns]]
    for _, r in df.iterrows():
        row = []
        for v in r:
            if isinstance(v, float):
                row.append("-" if v != v else fmt.format(v))
            else:
                row.append(v)
        rows.append(row)
    return table(rows, widths, font)


def image(name: str, max_w: float, max_h: float, caption: str | None = None):
    p = IMG / name if not str(name).startswith("/") else Path(name)
    w, h = PILImage.open(p).size
    s = min(max_w / w, max_h / h)
    out = [Image(str(p), width=w * s, height=h * s)]
    if caption:
        out.append(Paragraph(caption, S["caption"]))
    return KeepTogether(out)


def toc():
    t = TableOfContents()
    t.levelStyles = [S["toc1"], S["toc2"]]
    return t


def snippet(rel: str, name: str | None = None, max_lines: int = 40, start: str | None = None) -> str:
    """Source of a function / class / method (``Class.method``) from a project file, via AST."""
    path = ROOT / rel
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    if start:
        i = next(i for i, l in enumerate(lines) if start in l)
        seg = lines[i:i + max_lines]
    elif name is None:
        seg = lines[:max_lines]
    else:
        tree = ast.parse(src)
        parts = name.split(".")
        node = tree
        for part in parts:
            node = next(n for n in ast.iter_child_nodes(node)
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.name == part)
        first = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        seg = lines[first - 1:node.end_lineno]
    if len(seg) > max_lines:
        seg = seg[:max_lines] + ["    ...  (truncated - see the file for the rest)"]
    return "\n".join(seg)


def load_csv(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def load_json(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


__all__ = ["Doc", "P", "H1", "H2", "H3", "bullets", "code", "note", "table", "df_table", "image", "toc", "snippet",
           "load_csv", "load_json", "S", "cm", "Spacer", "PageBreak", "NextPageTemplate", "KeepTogether", "ROOT", "OUT"]
