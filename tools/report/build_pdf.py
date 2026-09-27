"""report/report.md -> report/report.pdf with reportlab (platypus).
Handles the subset of markdown the report uses: #/## headings, paragraphs,
bullet lists with **bold** and *italic*, images, and italic caption
paragraphs. Run from the repo root with the repo's .venv."""
import re, sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, KeepTogether, Table, TableStyle, CondPageBreak, PageBreak
from reportlab.lib import colors
from PIL import Image as PILImage

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("report/report.md")  # any markdown file; images resolve beside it
OUT = SRC.with_suffix(".pdf")
COMPACT = "compact" in sys.argv[2:]
from reportlab.lib.pagesizes import letter as _letter
SIDE = 1.7 if COMPACT else 2.1  # side margins, cm; the brief uses the narrower setting
FRAME = _letter[0] - 2 * SIDE * cm  # the text frame width; tables and figures are sized from it  # the brief: smaller type, tighter margins, narrower figures
base = getSampleStyleSheet()
body = ParagraphStyle("body", parent=base["Normal"], fontName="Times-Roman", fontSize=10 if COMPACT else 10.5, leading=11.6 if COMPACT else 13.2, spaceAfter=3 if COMPACT else 4, alignment=TA_JUSTIFY)
cap = ParagraphStyle("cap", parent=body, fontName="Times-Italic", fontSize=9.5, leading=12, textColor="#333333", spaceAfter=8, alignment=TA_LEFT)
capnext = ParagraphStyle("capnext", parent=cap, keepWithNext=1)
small = ParagraphStyle("small", parent=body, fontSize=9, leading=10.8, spaceAfter=2)
h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=16, leading=19, spaceAfter=4, spaceBefore=0)
h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=6 if COMPACT else 9, spaceAfter=3, keepWithNext=1)
byline = ParagraphStyle("byline", parent=body, textColor="#444444", spaceAfter=8)
cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11, spaceAfter=0, alignment=TA_LEFT)  # table cells are never justified
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")

table_rows = []
def flush_table():
    """Markdown table -> reportlab Table. First column left, the rest centred."""
    global table_rows
    if not table_rows:
        return
    marked = [i for i, r in enumerate(table_rows) if i > 0 and r[0].startswith("**")]
    header = table_rows[0]; grouped = any(": " in h for h in header)
    spans, extra = [], []
    if grouped:
        groups = [h.split(": ")[0] if ": " in h else "" for h in header]
        subs = [h.split(": ")[1] if ": " in h else h for h in header]
        top = []
        i = 0
        while i < len(groups):
            j = i
            while j + 1 < len(groups) and groups[j + 1] == groups[i] and groups[i]: j += 1
            top.append(groups[i] if groups[i] else "")
            if j > i: spans.append(("SPAN", (i, 0), (j, 0)))
            top.extend([""] * (j - i)); i = j + 1
        # a plain header (no group) spans both header rows
        for c, g in enumerate(groups):
            if not g: spans.append(("SPAN", (c, 0), (c, 1))); top[c] = subs[c]; subs[c] = ""
        table_rows = [top, subs] + table_rows[1:]; marked = [m + 1 for m in marked]
        extra = [("ALIGN", (1, 0), (-1, 1), "CENTER"), ("LINEBELOW", (0, 1), (-1, 1), 0.6, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white)]
        cellc0 = ParagraphStyle("cellc0", parent=cellb, alignment=1)
        rows = [[Paragraph(inline(c), (cellc0 if (i < 2 and k > 0) else cellb) if i < 2 else cell) for k, c in enumerate(r)] for i, r in enumerate(table_rows)]
    else:
        rows = [[Paragraph(inline(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(table_rows)]
    ncol = len(rows[0])
    # Column widths follow the longest body cell; a long header wraps onto lines rather than widening its column.
    nh = 2 if grouped else 1
    longest = [max([len(r[i]) for r in table_rows[nh:]] + [min(len(table_rows[k][i]), 14) for k in range(nh)]) for i in range(ncol)]
    weights = [max(12, min(n, 34)) for n in longest]
    widths = [FRAME * w / sum(weights) for w in weights]
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=nh)  # a long table may split; the header repeats
    t.setStyle(TableStyle([("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
                          + [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e8eef5")) for i in marked] + spans + extra))
    for r in rows[nh:]:
        for c in r[1:]:
            c.style = ParagraphStyle("cellc", parent=cell, alignment=1)
    table_rows = []
    # A short table stays with its caption; a long one may split with its header repeated.
    if len(rows) <= 7 and flow and isinstance(flow[-1], Paragraph) and flow[-1].style.name == "cap":
        flow.append(KeepTogether([flow.pop(), t, Spacer(1, 8)]))
    else:
        flow.append(t); flow.append(Spacer(1, 8))

def inline(md):
    """**bold** and *italic* to reportlab's mini-HTML; escape ampersands."""
    md = md.replace("&", "&amp;")
    md = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)
    md = re.sub(r"\*(.+?)\*", r"<i>\1</i>", md)
    md = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', md)
    return md

def image(path, width=16.5 * cm):
    im = PILImage.open(path); w, h = im.size
    img = Image(str(path), width=width, height=width * h / w); img.hAlign = "CENTER"; return img

flow, para, bullets = [], [], []
def flush_para():
    global para
    if para:
        text = " ".join(para).strip()
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            caption = Paragraph(inline(text[1:-1]), cap)
            if flow and isinstance(flow[-1], Image):
                flow.append(KeepTogether([flow.pop(), caption]))
            else:
                flow.append(caption)
        else:
            flow.append(Paragraph(inline(text), small if (COMPACT and re.match(r"^[A-Z][A-Za-z]+, [A-Z]\. ", text)) else body))
        para = []
numbered = False
def flush_bullets():
    global bullets, numbered
    if bullets:
        items = [ListItem(Paragraph(inline(" ".join(b)), body), leftIndent=14) for b in bullets]
        if numbered:
            flow.append(ListFlowable(items, bulletType="1", bulletFormat="%s.", leftIndent=16, bulletFontName="Times-Roman", bulletFontSize=10.5, spaceAfter=4))
        else:
            flow.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9, spaceAfter=4))
        bullets = []; numbered = False

first_para = True
para_style_override = []  # positions of reference paragraphs, set smaller in compact mode
for line in SRC.read_text().splitlines():
    if line.strip() == "<!-- pagebreak -->":
        flush_table(); flush_para(); flush_bullets(); flow.append(PageBreak())
    elif line.startswith("# "):
        flush_table(); flush_para(); flush_bullets(); flow.append(Paragraph(inline(line[2:]), h1))
    elif line.startswith("## "):
        flush_table(); flush_para(); flush_bullets()
        flow.append(CondPageBreak(3.5 * cm))  # a heading with under 3.5 cm left below it moves to the next page
        flow.append(Paragraph(inline(line[3:]), h2))
    elif line.startswith("!["):
        flush_para(); flush_bullets()
        path = SRC.parent / re.search(r"\((.+?)\)", line).group(1)
        # Figures are drawn about as wide as the page, so full width prints their text near its real size.
        width = FRAME * (0.85 if "classification" in path.name else 1.0)
        flow.append(Spacer(1, 4)); flow.append(image(path, width=width))
    elif line.startswith("|"):
        flush_para(); flush_bullets()
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not all(set(c) <= set("-: ") for c in cells):  # skip the |---| separator row
            table_rows.append(cells)
    elif line.startswith("- "):
        flush_table(); flush_para(); bullets.append([line[2:]])
    elif re.match(r"^\d+\. ", line):
        flush_table(); flush_para(); numbered = True; bullets.append([re.sub(r"^\d+\. ", "", line)])
    elif line.startswith("  ") and bullets:
        bullets[-1].append(line.strip())
    elif not line.strip():
        flush_table(); flush_para(); flush_bullets()
    else:
        if first_para and line.startswith("Wesley"):
            flow.append(Paragraph(inline(line), byline)); first_para = False
        else:
            para.append(line)
    if re.match(r"^[A-Z][A-Za-z]+, [A-Z]\.", line) and len(para) == 1 and COMPACT:
        para_style_override.append(len(flow))
flush_table(); flush_para(); flush_bullets()

m = 1.5 if COMPACT else 1.9
doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=SIDE * cm, rightMargin=SIDE * cm, topMargin=m * cm, bottomMargin=m * cm,
                        title="Forecasting daily store demand", author="Wesley Gates")
doc.build(flow)
from pypdf import PdfReader
print("pages:", len(PdfReader(str(OUT)).pages))
