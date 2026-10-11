"""Build the report.

    python report/build.py              # new draft: submission/drafts/group_2_report_<date>_<time>.docx (+ contribution)
    python report/build.py --final      # final PDFs: submission/group_2_report.pdf, group_2_contribution.pdf

Drafts are never overwritten: every run adds a new, time-stamped .docx.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def include_tables(md):
    """Replace {{table:name}} with report/tables/name.md, then size every table's columns by content."""
    md = re.sub(r"\{\{table:(\w+)\}\}", lambda m: (HERE / "tables" / f"{m.group(1)}.md").read_text(), md)
    return size_tables(md)


SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")
BR = re.compile(r"<br\s*/?>")         # line break inside a table cell (metric headers: name, then unit and arrow)
TEXT_PT, PAD_PT = 504.6, 8.0     # 178-mm text width; 4 pt cell padding on each side (style.css, docx_style.py)
_FONTS = {}


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(text):
    text = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"[\\{}_^]", "", m.group(1)), text)   # $s_T$ -> sT
    return re.sub(r"[*`]", "", text)


ARROWS = re.compile("[←→↑↓]")     # not in Helvetica Neue; the fallback font draws them about 1 em wide


def _text_pt(text, bold=False):
    """Width of text in the 8-pt Helvetica Neue table font, plus 6% for differences between renderers."""
    if bold not in _FONTS:
        try:
            from PIL import ImageFont
            _FONTS[bold] = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 80, index=1 if bold else 0)
        except Exception:                                  # no font or no Pillow: average glyph width
            _FONTS[bold] = None
    f = _FONTS[bold]
    n_arrows = len(ARROWS.findall(text))
    text = ARROWS.sub("", text)
    return 1.06 * (f.getlength(text) / 10 if f else 4.4 * len(text) * (1.05 if bold else 1)) + 8.0 * n_arrows


def _column_widths(rows):
    """Column widths in pt for a full-width table, close to a browser's auto layout: no column is narrower
    than its longest word, and a table that fits on one line per cell is simply stretched. Otherwise short
    columns (labels, numbers) keep their full width, and the long text columns share the rest in proportion
    to how much text they hold."""
    n = len(rows[0])
    lo, hi = [PAD_PT + 12] * n, [PAD_PT + 12] * n
    for r_i, row in enumerate(rows):
        bold = r_i == 0                                    # header row
        for j, cell in enumerate(row[:n]):
            lines = [_plain(t) for t in BR.split(cell)]
            words = [w for t in lines for w in t.split()] or [""]
            lo[j] = max(lo[j], max(_text_pt(w, bold) for w in words) + PAD_PT)
            hi[j] = max(hi[j], max(_text_pt(t, bold) for t in lines) + PAD_PT)
    if sum(hi) <= TEXT_PT:
        return [h * TEXT_PT / sum(hi) for h in hi]
    short = [j for j in range(n) if hi[j] <= 0.2 * TEXT_PT]
    long_ = [j for j in range(n) if j not in short]
    rest = TEXT_PT - sum(hi[j] for j in short)
    if not long_ or rest < sum(lo[j] for j in long_):
        long_, rest = list(range(n)), TEXT_PT
    w = list(hi)
    L, H = sum(lo[j] for j in long_), sum(hi[j] for j in long_)
    for j in long_:
        w[j] = lo[j] * rest / L if L >= rest else lo[j] + (hi[j] - lo[j]) * (rest - L) / (H - L)
    return w


def size_tables(md):
    """Rewrite the dash line of every pipe table so pandoc gives content-heavy columns more width.
    (pandoc turns dash counts into relative column widths; see --columns in to_pdf/to_docx.)"""
    lines = md.split("\n")
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and SEP.match(lines[i + 1]):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            rows = [_cells(lines[i])] + [_cells(l) for l in lines[i + 2:j]]
            widths = _column_widths(rows)
            new = []
            for k, (cell, w) in enumerate(zip(_cells(lines[i + 1]), widths)):
                d = "-" * max(3, round(200 * w / TEXT_PT))
                body = [_plain(r[k]) for r in rows[1:] if k < len(r)]
                if body and sum(map(len, body)) / len(body) > 20:     # running text reads better left-aligned
                    new.append(d)
                else:
                    new.append((":" if cell.startswith(":") else "") + d + (":" if cell.endswith(":") else ""))
            lines[i + 1] = "|" + "|".join(new) + "|"
            i = j
        else:
            i += 1
    return "\n".join(lines)


def to_pdf(md_path, pdf_path):
    md = include_tables(md_path.read_text())
    tmp_md = HERE / f".{md_path.stem}.build.md"
    tmp_md.write_text(md)
    html = HERE / f".{md_path.stem}.html"
    subprocess.run(["pandoc", str(tmp_md), "-o", str(html), "--standalone", "--embed-resources", "--columns=20",
                    "--css", str(HERE / "style.css"), "--resource-path", str(HERE),
                    "--metadata", f"title={md_path.stem}", "--mathml"], check=True, cwd=HERE)
    chrome = CHROME if Path(CHROME).exists() else shutil.which("chromium") or shutil.which("google-chrome")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_md.unlink()
    print("wrote", pdf_path)


def to_docx(md_path, docx_path):
    """Word version with the PDF's layout: HTML blocks become pandoc custom styles, which
    report/reference.docx defines to match style.css; docx_style.postprocess does the rest."""
    import docx_style
    md = include_tables(md_path.read_text())
    md = re.sub(r'<figure><img src="([^"]+)"(?: style="width:\s*(\d+)%")?><figcaption>(.*?)</figcaption></figure>',
                lambda m: f"![{m.group(3)}]({m.group(1)}){{width={m.group(2) or 100}%}}", md, flags=re.S)
    md = re.sub(r'<p class="tcap">(.*?)</p>', r'::: {custom-style="Table Caption"}\n\1\n:::', md, flags=re.S)
    md = re.sub(r'<div class="authors">\s*(.*?)\s*</div>', r'::: {custom-style="Authors"}\n\1\n:::', md, flags=re.S)
    md = re.sub(r'<div class="abstract">\s*(.*?)\s*</div>', r'::: {custom-style="Abstract"}\n\1\n:::', md, flags=re.S)
    md = re.sub(r'<div class="refs">\s*(## References)\s*(.*?)</div>', r'\1\n\n::: {custom-style="Bibliography"}\n\2\n:::', md, flags=re.S)
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = BR.sub("`<w:r><w:br/></w:r>`{=openxml}", md)          # raw HTML is dropped in docx; use a Word line break
    ref = docx_style.make_reference(HERE / "reference.docx")
    tmp_md = HERE / f".{md_path.stem}.docx.md"
    tmp_md.write_text(md)
    subprocess.run(["pandoc", str(tmp_md), "-o", str(Path(docx_path).resolve()), "--resource-path", str(HERE),
                    "--reference-doc", str(ref), "--columns=20"], check=True, cwd=HERE)
    tmp_md.unlink()
    docx_style.postprocess(docx_path)
    print("wrote", docx_path)


def main():
    from datetime import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="2")
    ap.add_argument("--final", action="store_true", help="write the final PDFs instead of a draft .docx")
    a = ap.parse_args()
    out = HERE.parent / "submission"
    out.mkdir(exist_ok=True)
    if a.final:
        to_pdf(HERE / "report.md", out / f"group_{a.group}_report.pdf")
        to_pdf(HERE / "contribution.md", out / f"group_{a.group}_contribution.pdf")
        return
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    drafts = out / "drafts"
    drafts.mkdir(exist_ok=True)
    to_docx(HERE / "report.md", drafts / f"group_{a.group}_report_{stamp}.docx")
    to_docx(HERE / "contribution.md", drafts / f"group_{a.group}_contribution_{stamp}.docx")


if __name__ == "__main__":
    sys.exit(main())
