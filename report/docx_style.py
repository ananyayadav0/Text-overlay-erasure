"""Make the .docx look like the PDF (report/style.css).

make_reference()  builds report/reference.docx: page size, fonts, sizes, colours, spacing and the
                  horizontal-rule table style copied from style.css.
postprocess()     fixes what a reference document cannot express: content-based column widths for
                  short tables (the browser's auto layout), centred header cells, vertically centred
                  cells, table text style, the gap after tables and list indents.

Word adds the space after one paragraph to the space before the next, whereas CSS collapses the two
margins; the "before" values below are therefore the CSS value minus the 6 pt after a body paragraph.
"""
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK2, RULE = "0B0B0B", "52514E", "D9D8D3"
SERIF, SANS = "Charter", "Helvetica Neue"
TEXT_WIDTH_TW = 10092                    # A4 (210 mm) minus 2 x 16 mm margins, in twips
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def pt(v):            # points -> twentieths of a point
    return str(round(v * 20))


def half(v):          # points -> half-points (w:sz)
    return str(round(v * 2))


def _q(tag):
    return f"{{{W}}}{tag}"


def _set_rpr(style_el, font=None, size=None, color=None, bold=None, italic=False):
    """Rewrite a style's run properties from scratch (drops theme fonts and theme colours)."""
    from lxml import etree
    rpr = style_el.find(_q("rPr"))
    if rpr is None:
        rpr = etree.SubElement(style_el, _q("rPr"))
    for child in list(rpr):
        if child.tag in {_q(t) for t in ("rFonts", "b", "bCs", "i", "iCs", "color", "sz", "szCs")}:
            rpr.remove(child)
    order = []
    if font:
        f = etree.Element(_q("rFonts"))
        for a in ("ascii", "hAnsi", "cs", "eastAsia"):
            f.set(_q(a), font)
        order.append(f)
    if bold is not None:
        for t in ("b", "bCs"):
            e = etree.Element(_q(t))
            e.set(_q("val"), "1" if bold else "0")
            order.append(e)
    for t in ("i", "iCs"):
        e = etree.Element(_q(t))
        e.set(_q("val"), "1" if italic else "0")
        order.append(e)
    if color:
        e = etree.Element(_q("color"))
        e.set(_q("val"), color)
        order.append(e)
    if size:
        for t in ("sz", "szCs"):
            e = etree.Element(_q(t))
            e.set(_q("val"), half(size))
            order.append(e)
    for i, e in enumerate(order):
        rpr.insert(i, e)


def _set_ppr(style_el, align=None, before=0, after=0, line=None, keep_next=False, left=None, right=None,
             hanging=None, border_bottom=None):
    """Rewrite a style's paragraph properties (element order follows the OOXML schema)."""
    from lxml import etree
    ppr = style_el.find(_q("pPr"))
    if ppr is not None:
        style_el.remove(ppr)
    ppr = etree.Element(_q("pPr"))
    rpr = style_el.find(_q("rPr"))
    (rpr.addprevious(ppr) if rpr is not None else style_el.append(ppr))
    if keep_next:
        etree.SubElement(ppr, _q("keepNext"))
    if border_bottom:
        bdr = etree.SubElement(ppr, _q("pBdr"))
        b = etree.SubElement(bdr, _q("bottom"))
        for k, v in border_bottom.items():
            b.set(_q(k), v)
    sp = etree.SubElement(ppr, _q("spacing"))
    sp.set(_q("before"), pt(before))
    sp.set(_q("after"), pt(after))
    if line:
        sp.set(_q("line"), pt(line))
        sp.set(_q("lineRule"), "atLeast")
    if left is not None or right is not None or hanging is not None:
        ind = etree.SubElement(ppr, _q("ind"))
        if left is not None:
            ind.set(_q("left"), pt(left))
        if right is not None:
            ind.set(_q("right"), pt(right))
        if hanging is not None:
            ind.set(_q("hanging"), pt(hanging))
    if align:
        jc = etree.SubElement(ppr, _q("jc"))
        jc.set(_q("val"), align)


def _style(styles, style_id, name=None, based_on="Normal", kind="paragraph"):
    """Find a style by id, or create it."""
    from lxml import etree
    for s in styles.findall(_q("style")):
        if s.get(_q("styleId")) == style_id:
            return s
    s = etree.SubElement(styles, _q("style"))
    s.set(_q("type"), kind)
    s.set(_q("customStyle"), "1")
    s.set(_q("styleId"), style_id)
    n = etree.SubElement(s, _q("name"))
    n.set(_q("val"), name or style_id)
    if based_on:
        b = etree.SubElement(s, _q("basedOn"))
        b.set(_q("val"), based_on)
    etree.SubElement(s, _q("qFormat"))
    return s


TABLE_STYLE = f"""<w:style xmlns:w="{W}" w:type="table" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="TableNormal"/><w:uiPriority w:val="59"/>
<w:tblPr><w:tblBorders>
<w:top w:val="single" w:sz="9" w:space="0" w:color="{INK}"/><w:left w:val="nil"/>
<w:bottom w:val="single" w:sz="9" w:space="0" w:color="{INK}"/><w:right w:val="nil"/>
<w:insideH w:val="single" w:sz="6" w:space="0" w:color="{RULE}"/><w:insideV w:val="nil"/></w:tblBorders>
<w:tblCellMar><w:top w:w="50" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="50" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>
<w:tcPr><w:vAlign w:val="center"/></w:tcPr>
<w:tblStylePr w:type="firstRow"><w:rPr><w:b/><w:bCs/></w:rPr><w:tcPr><w:tcBorders>
<w:top w:val="single" w:sz="9" w:space="0" w:color="{INK}"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{INK}"/></w:tcBorders></w:tcPr></w:tblStylePr>
</w:style>"""


def make_reference(path=HERE / "reference.docx"):
    from lxml import etree
    base = path.with_suffix(".base.docx")
    subprocess.run(["pandoc", "-o", str(base), "--print-default-data-file", "reference.docx"], check=True)
    work = path.parent / ".reference_build"
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(base) as z:
        z.extractall(work)
    base.unlink()

    st_path = work / "word" / "styles.xml"
    tree = etree.parse(str(st_path))
    styles = tree.getroot()

    # document defaults: Charter 10.5 pt, ink, British English (hyphenation dictionary)
    rpr_def = styles.find(f"{_q('docDefaults')}/{_q('rPrDefault')}/{_q('rPr')}")
    for child in list(rpr_def):
        rpr_def.remove(child)
    rpr_def.append(etree.fromstring(f'<w:rFonts xmlns:w="{W}" w:ascii="{SERIF}" w:hAnsi="{SERIF}" w:cs="{SERIF}" w:eastAsia="{SERIF}"/>'))
    rpr_def.append(etree.fromstring(f'<w:color xmlns:w="{W}" w:val="{INK}"/>'))
    rpr_def.append(etree.fromstring(f'<w:sz xmlns:w="{W}" w:val="{half(10.5)}"/>'))
    rpr_def.append(etree.fromstring(f'<w:szCs xmlns:w="{W}" w:val="{half(10.5)}"/>'))
    rpr_def.append(etree.fromstring(f'<w:lang xmlns:w="{W}" w:val="en-GB" w:eastAsia="en-GB" w:bidi="ar-SA"/>'))

    body = dict(font=SERIF, size=10.5, color=INK)
    for sid in ("Normal", "BodyText", "FirstParagraph"):
        s = _style(styles, sid)
        _set_rpr(s, **body)
        _set_ppr(s, align="both", before=0, after=6 if sid != "Normal" else 0, line=10.5 * 1.42)
    s = _style(styles, "Compact")                         # tight list items
    _set_rpr(s, **body)
    _set_ppr(s, align="left", before=0, after=2, line=10.5 * 1.42)

    s = _style(styles, "Heading1")                        # the report title
    _set_rpr(s, font=SERIF, size=17, color=INK, bold=True)
    _set_ppr(s, align="center", before=0, after=4, line=17 * 1.2, keep_next=True)
    s = _style(styles, "Heading2")                        # section, with the thin rule underneath
    _set_rpr(s, font=SERIF, size=12.5, color=INK, bold=True)
    _set_ppr(s, align="left", before=10, after=5, line=12.5 * 1.42, keep_next=True,
             border_bottom={"val": "single", "sz": "6", "space": "2", "color": RULE})
    s = _style(styles, "Heading3")
    _set_rpr(s, font=SERIF, size=11, color=INK, bold=True)
    _set_ppr(s, align="left", before=5, after=4, line=11 * 1.42, keep_next=True)

    s = _style(styles, "Authors", "Authors")
    _set_rpr(s, font=SERIF, size=10, color=INK2)
    _set_ppr(s, align="center", before=0, after=12, line=10 * 1.42)
    s = _style(styles, "Abstract")                        # CSS: 10 pt below, collapsing into the next heading's 16 pt
    _set_rpr(s, font=SERIF, size=10, color=INK)
    _set_ppr(s, align="both", before=0, after=6, line=10 * 1.42, left=24, right=24)

    s = _style(styles, "TableCaption")
    _set_rpr(s, font=SERIF, size=9, color=INK2)
    _set_ppr(s, align="both", before=2, after=4, line=9 * 1.42, keep_next=True)
    s = _style(styles, "ImageCaption")
    _set_rpr(s, font=SERIF, size=9, color=INK2)
    _set_ppr(s, align="both", before=3, after=10, line=9 * 1.42)
    for sid in ("Figure", "CaptionedFigure"):
        s = _style(styles, sid)
        _set_ppr(s, align="center", before=2, after=0, keep_next=True)
    s = _style(styles, "Bibliography")
    _set_rpr(s, font=SERIF, size=9, color=INK)
    _set_ppr(s, align="left", before=0, after=3, line=9 * 1.42, left=16, hanging=16)
    s = _style(styles, "TableText", "Table Text")
    _set_rpr(s, font=SANS, size=8, color=INK)
    _set_ppr(s, align="left", before=0, after=0, line=8 * 1.42)

    for s in styles.findall(_q("style")):                 # replace pandoc's table style
        if s.get(_q("styleId")) == "Table":
            styles.remove(s)
    styles.append(etree.fromstring(TABLE_STYLE))
    tree.write(str(st_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    # A4 page, margins from style.css (@page 18 mm 16 mm), no header or footer
    doc_path = work / "word" / "document.xml"
    d = doc_path.read_text()
    sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1020" w:right="907" w:bottom="1020" w:left="907" w:header="567" w:footer="567" w:gutter="0"/>'
            '</w:sectPr>')
    d = re.sub(r"<w:sectPr>.*?</w:sectPr>", sect, d, flags=re.S)
    doc_path.write_text(d)

    # no automatic hyphenation: Chrome does not hyphenate the PDF either (the HTML has no lang attribute)

    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    return path


# ------------------------------------------------------------------ after pandoc

def _cell_texts(row_xml):
    cells = re.findall(r"<w:tc>(.*?)</w:tc>", row_xml, flags=re.S)
    return ["".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", c, flags=re.S)) for c in cells]


def _auto_widths(tbl):
    """Approximate the browser's auto layout at width 100%: columns proportional to their widest content."""
    rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    ncol = len(re.findall(r"<w:gridCol ", tbl))
    widest = [4.0] * ncol
    for r_i, row in enumerate(rows):
        for j, text in enumerate(_cell_texts(row)[:ncol]):
            w = len(text) * (1.08 if r_i == 0 else 1.0) + 2      # bold header is a little wider; + padding
            widest[j] = max(widest[j], w)
    total = sum(widest)
    widths = [round(TEXT_WIDTH_TW * w / total) for w in widest]
    widths[-1] += TEXT_WIDTH_TW - sum(widths)
    return widths


def _fix_table(tbl):
    if '<w:tblW w:type="auto" w:w="0" />' in tbl:            # fallback: emulate auto layout at 100 %
        widths = _auto_widths(tbl)
        tbl = tbl.replace('<w:tblW w:type="auto" w:w="0" />',
                          '<w:tblW w:type="pct" w:w="5000" /><w:tblLayout w:type="fixed" />')
    else:                                                      # pandoc's widths are relative to a narrower page
        cols = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)" />', tbl)]
        widths = [round(TEXT_WIDTH_TW * c / sum(cols)) for c in cols]
    widths[-1] += TEXT_WIDTH_TW - sum(widths)                  # exactly the full text width
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>",
                 "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}" />' for w in widths) + "</w:tblGrid>",
                 tbl, flags=re.S)
    # every cell also states its width (Word uses tcW when the grid and the cells disagree)
    def cell_widths(row):
        k = iter(widths)
        return re.sub(r"<w:tc><w:tcPr />", lambda m: f'<w:tc><w:tcPr><w:tcW w:w="{next(k)}" w:type="dxa" /></w:tcPr>', row.group(0))
    tbl = re.sub(r"<w:tr>.*?</w:tr>", cell_widths, tbl, flags=re.S)
    tbl = tbl.replace('<w:pStyle w:val="Compact" />', '<w:pStyle w:val="TableText" />')
    tbl = tbl.replace('w:type="dxa" /></w:tcPr>', 'w:type="dxa" /><w:vAlign w:val="center" /></w:tcPr>')
    tbl = tbl.replace("<w:tcPr />", '<w:tcPr><w:vAlign w:val="center" /></w:tcPr>')
    # a row never breaks across pages (style.css: tr { break-inside: avoid })
    tbl = tbl.replace("<w:tr><w:trPr>", "<w:tr><w:trPr><w:cantSplit />")
    tbl = re.sub(r"<w:tr>(?!<w:trPr>)", "<w:tr><w:trPr><w:cantSplit /></w:trPr>", tbl)
    # header cells are centred, as <th> is in the browser
    first_row = re.search(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    if first_row:
        hdr = re.sub(r'(<w:pStyle w:val="TableText" />)(</w:pPr>)', r'\1<w:jc w:val="center" />\2', first_row.group(0))
        tbl = tbl[:first_row.start()] + hdr + tbl[first_row.end():]
    return tbl


# gap (twips) between a table and the next block: CSS max(4 pt, that block's top margin)
AFTER_TABLE = {"BodyText": 80, "FirstParagraph": 80, "CaptionedFigure": 160, "TableCaption": 160,
               "Heading2": 320, "Heading3": 220}


def _space_after_lists(d):
    """The last item of a list gets the list's 6 pt bottom margin (CSS ul margin-bottom)."""
    paras = list(re.finditer(r"<w:p>.*?</w:p>", d, flags=re.S))
    out, last = [], 0
    for i, m in enumerate(paras):
        p = m.group(0)
        nxt = paras[i + 1].group(0) if i + 1 < len(paras) else ""
        if "<w:numPr>" in p and "<w:numPr>" not in nxt:
            p = re.sub(r'(<w:pStyle w:val="[^"]+" />(?:<w:numPr>.*?</w:numPr>)?)', r'\1<w:spacing w:after="120" />', p, count=1, flags=re.S)
        out.append(d[last:m.start()] + p)
        last = m.end()
    return "".join(out) + d[last:]


def postprocess(docx_path):
    docx_path = Path(docx_path)
    tmp = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx_path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                d = data.decode("utf-8")
                d = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: _fix_table(m.group(0)), d, flags=re.S)
                # CSS puts 4 pt below a table (collapsing with the next block's own top margin);
                # Word needs that gap on the paragraph that follows
                d = re.sub(r'(</w:tbl>(?:\s*<w:bookmark(?:Start|End)[^>]*/>)*\s*<w:p><w:pPr><w:pStyle w:val="([^"]+)" />)',
                           lambda m: m.group(1) + f'<w:spacing w:before="{AFTER_TABLE.get(m.group(2), 80)}" />', d)
                d = _space_after_lists(d)
                data = d.encode("utf-8")
            elif item.filename == "word/numbering.xml":
                n = data.decode("utf-8")                    # list text at 17 pt, marker at 4 pt, as in the PDF
                n = n.replace('<w:ind w:left="720" w:hanging="360" />', '<w:ind w:left="340" w:hanging="260" />')
                data = n.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx_path)
