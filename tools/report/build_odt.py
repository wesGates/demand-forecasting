"""Markdown -> .odt built directly with odfpy, full-resolution figures sized to
the text width, so an export from Writer is as crisp as the reportlab PDF.
Handles the report's markdown subset: # and ## headings, paragraphs with
**bold** and *italic*, bullet and numbered lists, images, italic captions,
and tables (a "group: sub" header becomes a two-row header; a row whose
first cell is bold is shaded).
Usage, from the fork root: md_to_odt2.py <source.md> <out.odt>"""
import re, sys
from pathlib import Path
from PIL import Image as PILImage
from odf.opendocument import OpenDocumentText
from odf.style import (Style, TextProperties, ParagraphProperties, TableProperties, TableColumnProperties,
                       TableCellProperties, GraphicProperties, ListLevelProperties, PageLayout, PageLayoutProperties, MasterPage)
from odf.text import H, P, Span, List, ListItem, ListStyle, ListLevelStyleBullet, ListLevelStyleNumber
from odf.table import Table, TableColumn, TableRow, TableCell, CoveredTableCell
from odf.draw import Frame, Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
COMPACT = "compact" in sys.argv[3:]  # the brief: narrower margins
SIDE, TOPBOT = (1.7, 1.5) if COMPACT else (2.1, 1.9)
TEXT_WIDTH_CM = round(21.59 - 2 * SIDE, 2)

doc = OpenDocumentText()
# page layout
pl = PageLayout(name="pm1"); pl.addElement(PageLayoutProperties(pagewidth="21.59cm", pageheight="27.94cm", margintop=f"{TOPBOT}cm", marginbottom=f"{TOPBOT}cm", marginleft=f"{SIDE}cm", marginright=f"{SIDE}cm"))
doc.automaticstyles.addElement(pl); doc.masterstyles.addElement(MasterPage(name="Standard", pagelayoutname=pl))

def pstyle(name, size="10.5pt", bold=False, italic=False, color=None, align="justify", before="0cm", after="0.15cm", family="paragraph", parent=None):
    st = Style(name=name, family=family, parentstylename=parent)
    tp = dict(fontname="Liberation Serif", fontsize=size)
    if bold: tp["fontweight"] = "bold"
    if italic: tp["fontstyle"] = "italic"
    if color: tp["color"] = color
    st.addElement(TextProperties(**tp))
    if family == "paragraph":
        st.addElement(ParagraphProperties(textalign=align, margintop=before, marginbottom=after, keepwithnext="always" if name.startswith("H") else "auto"))
    doc.styles.addElement(st); return st
body = pstyle("Body"); cap = pstyle("Caption", size="9.5pt", italic=True, color="#333333", align="start", after="0.25cm")
h1 = pstyle("H1", size="16pt", bold=True, align="start", after="0.1cm"); h2 = pstyle("H2", size="12pt", bold=True, align="start", before="0.35cm", after="0.1cm")
cellp = pstyle("Cell", size="9pt", align="start", after="0cm"); cellc = pstyle("CellC", size="9pt", align="center", after="0cm")
cellh = pstyle("CellH", size="9pt", bold=True, align="center", after="0cm"); cellh0 = pstyle("CellH0", size="9pt", bold=True, align="start", after="0cm")
figp = pstyle("Figure", align="center", after="0.1cm")  # figures sit centred in their own paragraph
h2_break = pstyle("H2Break", size="12pt", bold=True, align="start", before="0cm", after="0.1cm")  # a heading that starts a new page
h2_break.addElement(ParagraphProperties(breakbefore="page"))
cap_break = pstyle("CaptionBreak", size="9.5pt", italic=True, color="#333333", align="start", after="0.25cm")
cap_break.addElement(ParagraphProperties(breakbefore="page"))
body_break = pstyle("BodyBreak"); body_break.addElement(ParagraphProperties(breakbefore="page"))
bold = Style(name="B", family="text"); bold.addElement(TextProperties(fontweight="bold")); doc.styles.addElement(bold)
ital = Style(name="I", family="text"); ital.addElement(TextProperties(fontstyle="italic")); doc.styles.addElement(ital)
mono = Style(name="M", family="text"); mono.addElement(TextProperties(fontname="Liberation Mono", fontsize="9pt")); doc.styles.addElement(mono)
# lists
ls_b = ListStyle(name="Bullets"); lvl = ListLevelStyleBullet(level="1", bulletchar="•"); lvl.addElement(ListLevelProperties(spacebefore="0.3cm", minlabelwidth="0.5cm")); ls_b.addElement(lvl); doc.styles.addElement(ls_b)
ls_n = ListStyle(name="Numbers"); lvn = ListLevelStyleNumber(level="1", numformat="1", numsuffix="."); lvn.addElement(ListLevelProperties(spacebefore="0.3cm", minlabelwidth="0.6cm")); ls_n.addElement(lvn); doc.styles.addElement(ls_n)
# table cell styles
def cellstyle(name, shaded=False, top=False, bottom=False):
    st = Style(name=name, family="table-cell")
    props = dict(padding="0.06cm", border="none")
    if top: props["bordertop"] = "0.6pt solid #000000"
    if bottom: props["borderbottom"] = "0.6pt solid #000000"
    if shaded: props["backgroundcolor"] = "#e8eef5"
    st.addElement(TableCellProperties(**props)); doc.automaticstyles.addElement(st); return st
cs_plain, cs_shade = cellstyle("c0"), cellstyle("c1", shaded=True)
cs_head, cs_last, cs_last_sh = cellstyle("ch", bottom=True), cellstyle("cl", bottom=True), cellstyle("cls", shaded=True, bottom=True)
fr = Style(name="fr1", family="graphic"); fr.addElement(GraphicProperties(anchortype="as-char")); doc.automaticstyles.addElement(fr)

def inline(text, para):
    """**bold**, *italic*, `code` into text spans."""
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`", text):
        if m.start() > pos: para.addText(text[pos:m.start()])
        if m.group(1): para.addElement(Span(stylename=bold, text=m.group(1)))
        elif m.group(2): para.addElement(Span(stylename=ital, text=m.group(2)))
        else: para.addElement(Span(stylename=mono, text=m.group(3)))
        pos = m.end()
    if pos < len(text): para.addText(text[pos:])
    return para

def add_table(rows):
    header, grouped = rows[0], any(": " in h for h in rows[0])
    ncol = len(header); nh = 2 if grouped else 1
    marked = {i for i, r in enumerate(rows) if i > 0 and r[0].startswith("**")}
    t = Table(name=f"T{tcount[0]}"); tcount[0] += 1
    tst = Style(name=f"ts{tcount[0]}", family="table"); tst.addElement(TableProperties(width=f"{TEXT_WIDTH_CM}cm", align="left", margintop="0.1cm", marginbottom="0.3cm")); doc.automaticstyles.addElement(tst); t.setAttribute("stylename", tst)
    longest = [max([len(r[i]) for r in rows[1:]] + [min(len(rows[0][i]), 14)]) for i in range(ncol)]
    weights = [max(12, min(n, 34)) for n in longest]
    for i, w in enumerate(weights):
        cst = Style(name=f"tc{tcount[0]}_{i}", family="table-column"); cst.addElement(TableColumnProperties(columnwidth=f"{TEXT_WIDTH_CM * w / sum(weights):.2f}cm")); doc.automaticstyles.addElement(cst)
        t.addElement(TableColumn(stylename=cst))
    def cell(text, pst, cst, span=1):
        c = TableCell(stylename=cst, valuetype="string")
        if span > 1: c.setAttribute("numbercolumnsspanned", str(span))
        c.addElement(inline(text, P(stylename=pst))); return c
    if grouped:
        groups = [h.split(": ")[0] if ": " in h else "" for h in header]; subs = [h.split(": ")[1] if ": " in h else "" for h in header]
        r1, r2 = TableRow(), TableRow()
        i = 0
        while i < ncol:
            j = i
            while j + 1 < ncol and groups[i] and groups[j + 1] == groups[i]: j += 1
            if groups[i]:
                r1.addElement(cell(groups[i], cellh, cs_plain, span=j - i + 1))
                for k in range(i + 1, j + 1): r1.addElement(CoveredTableCell())  # the columns under the span
            else:
                r1.addElement(cell(header[i], cellh0 if i == 0 else cellh, cs_plain))
            i = j + 1
        for i in range(ncol): r2.addElement(cell(subs[i] if groups[i] else "", cellh0 if i == 0 else cellh, cs_head))
        t.addElement(r1); t.addElement(r2)
    else:
        r1 = TableRow()
        for i, h in enumerate(header): r1.addElement(cell(h, cellh0 if i == 0 else cellh, cs_head))
        t.addElement(r1)
    for ri, r in enumerate(rows[1:], start=1):
        last = ri == len(rows) - 1
        cst = (cs_last_sh if ri in marked else cs_last) if last else (cs_shade if ri in marked else cs_plain)
        tr = TableRow()
        for i, c in enumerate(r): tr.addElement(cell(c, cellp if i == 0 else cellc, cst))
        t.addElement(tr)
    doc.text.addElement(t)

def add_image(path):
    im = PILImage.open(path); w_cm = TEXT_WIDTH_CM * (1.0 if "final_results" in path.name else 0.91 if "progression" in path.name else 0.85 if "weekly_error" in path.name else 0.6 if "classification" in path.name else 0.94)
    h_cm = w_cm * im.height / im.width
    href = doc.addPicture(str(path))
    p = P(stylename=figp)
    frame = Frame(stylename=fr, width=f"{w_cm:.2f}cm", height=f"{h_cm:.2f}cm", anchortype="as-char")
    frame.addElement(Image(href=href)); p.addElement(frame); doc.text.addElement(p)

tcount = [0]
para, bullets, numbered, table_rows = [], [], False, []
page_break_pending = False
def flush_para():
    global para, page_break_pending
    if para:
        text = " ".join(para).strip()
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            doc.text.addElement(inline(text[1:-1], P(stylename=cap_break if page_break_pending else cap)))
        else:
            doc.text.addElement(inline(text, P(stylename=body_break if page_break_pending else body)))
        para = []; page_break_pending = False
def flush_bullets():
    global bullets, numbered
    if bullets:
        lst = List(stylename=ls_n if numbered else ls_b)
        for b in bullets:
            li = ListItem(); li.addElement(inline(" ".join(b), P(stylename=body))); lst.addElement(li)
        doc.text.addElement(lst); bullets = []; numbered = False
def flush_table():
    global table_rows
    if table_rows: add_table(table_rows); table_rows = []
def flush_all(): flush_table(); flush_para(); flush_bullets()

for line in SRC.read_text().splitlines():
    if line.strip() == "<!-- pagebreak -->":
        flush_all(); page_break_pending = True; continue
    if line.startswith("# "):
        flush_all(); doc.text.addElement(H(outlinelevel=1, stylename=h1, text=line[2:]))
    elif line.startswith("## "):
        flush_all()
        doc.text.addElement(H(outlinelevel=2, stylename=h2_break if page_break_pending else h2, text=line[3:])); page_break_pending = False
    elif line.startswith("!["):
        flush_all(); add_image(SRC.parent / re.search(r"\((.+?)\)", line).group(1))
    elif line.startswith("|"):
        flush_para(); flush_bullets()
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not all(set(c) <= set("-: ") for c in cells): table_rows.append(cells)
    elif line.startswith("- "):
        flush_table(); flush_para(); bullets.append([line[2:]])
    elif re.match(r"^\d+\. ", line):
        flush_table(); flush_para(); numbered = True; bullets.append([re.sub(r"^\d+\. ", "", line)])
    elif line.startswith("  ") and bullets:
        bullets[-1].append(line.strip())
    elif not line.strip():
        flush_all()
    else:
        para.append(line)
flush_all()
doc.save(str(OUT)); print("wrote", OUT)
