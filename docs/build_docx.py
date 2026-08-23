#!/usr/bin/env python3
"""
Assemble the MSc dissertation into a single .docx.

Arial 12, double-spaced, justified body. Automatic table of contents, list of
figures and list of tables (Word field codes - press Ctrl+A then F9 in Word to
populate them). Roman page numbers in the front matter, arabic from Chapter 1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Mm, RGBColor

DOCS = Path(sys.argv[1])
FIGDIR = Path(sys.argv[2])
OUT = Path(sys.argv[3])

BODY_FONT = "Arial"

# --- figure placement -------------------------------------------------------
# Literal numbering rather than SEQ fields: the numbers are referenced by name
# throughout the prose, so they must not shift. Page numbers in the List of
# Figures still come from a TOC field over the caption style.
FIGURES = {
    "6.1 Test plan": [
        ("fig_5_01_class_balance.png", "Class balance in both datasets"),
        ("fig_5_02_samples_dataset_a.png", "Sample images from Dataset A (StyleGAN fakes, FFHQ reals)"),
        ("fig_5_03_samples_dataset_b.png", "Sample images from Dataset B (OpenForensics)"),
        ("fig_5_04_domain_gap.png", "Low-level statistical differences between the two datasets"),
        ("fig_5_05_class_separability.png", "Class separability of simple image statistics, Dataset A"),
        ("fig_5_06_colour_channels.png", "Colour channel distributions by class"),
    ],
    "6.2 In-domain results": [
        ("fig_6_01_baseline_roc.png", "ROC curves for both classical baselines, in-domain"),
        ("fig_6_cm_logistic_regression_A.png", "Confusion matrix, logistic regression, in-domain"),
        ("fig_6_cm_hog_lbp_svm_A.png", "Confusion matrix, HOG + LBP with linear SVC, in-domain"),
        ("fig_6_scores_logistic_regression.png", "Score distribution by class, logistic regression"),
        ("fig_6_scores_hog_lbp_svm.png", "Score distribution by class, HOG + LBP"),
        ("fig_6_cnn_curves_mobilenet_v2.png", "Training and validation curves, MobileNetV2"),
        ("fig_6_cnn_curves_efficientnet_b0.png", "Training and validation curves, EfficientNet-B0"),
        ("fig_6_cnn_cm_mobilenet_v2.png", "Confusion matrix, MobileNetV2, in-domain"),
        ("fig_6_cnn_cm_efficientnet_b0.png", "Confusion matrix, EfficientNet-B0, in-domain"),
    ],
    "6.3 Cross-dataset results": [
        ("fig_6_cm_logistic_regression_B.png", "Confusion matrix, logistic regression, cross-dataset"),
        ("fig_6_cm_hog_lbp_svm_B.png", "Confusion matrix, HOG + LBP, cross-dataset"),
        ("fig_6_cnn_scores_mobilenet_v2.png", "Score distributions in-domain against cross-dataset, MobileNetV2"),
        ("fig_6_cnn_scores_efficientnet_b0.png", "Score distributions in-domain against cross-dataset, EfficientNet-B0"),
        ("fig_6_curves_mobilenet_v2.png", "ROC and precision-recall curves, MobileNetV2"),
        ("fig_6_curves_efficientnet_b0.png", "ROC and precision-recall curves, EfficientNet-B0"),
    ],
    "6.5 Robustness results": [
        ("fig_5_07_jpeg_preview.png", "Visual effect of JPEG compression at the tested quality levels"),
        ("fig_5_08_noise_preview.png", "Visual effect of Gaussian noise at the tested magnitudes"),
        ("fig_7_01_robustness_mobilenet_v2.png", "Accuracy against perturbation magnitude, all families"),
        ("fig_7_02_worst_case_mobilenet_v2.png", "Worst-case accuracy by perturbation family"),
    ],
}
_n = 0
for _sec in FIGURES:
    lst = []
    for fn, cap in FIGURES[_sec]:
        _n += 1
        lst.append((fn, f"Figure 6.{_n}: {cap}"))
    FIGURES[_sec] = lst
TOTAL_FIGURES = _n

FILES = [
    ("ch1_introduction.md", "Chapter 1: Introduction"),
    ("ch2_literature_review.md", "Chapter 2: Literature Review and Fact Finding"),
    ("ch3_methodology.md", "Chapter 3: Requirements and Methodology"),
    ("ch4_design.md", "Chapter 4: Design"),
    ("ch5_implementation.md", "Chapter 5: Implementation"),
    ("ch6_results.md", "Chapter 6: Testing, Results and Evaluation"),
    ("ch7_critical_evaluation.md", "Chapter 7: Critical Evaluation"),
    ("ch8_conclusion.md", "Chapter 8: Conclusion and Further Work"),
]


# --- low-level helpers ------------------------------------------------------
def field(paragraph, instr: str, placeholder: str = "Right-click and choose Update Field"):
    """Insert a Word field code (TOC, PAGE, ...)."""
    r = paragraph.add_run()
    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin")
    r._r.append(fc)
    r2 = paragraph.add_run()
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    r2._r.append(it)
    r3 = paragraph.add_run()
    sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
    r3._r.append(sep)
    r4 = paragraph.add_run(placeholder)
    r5 = paragraph.add_run()
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    r5._r.append(end)


def page_numbering(section, fmt: str, start: int | None = None):
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    el = OxmlElement("w:pgNumType")
    el.set(qn("w:fmt"), fmt)
    if start is not None:
        el.set(qn("w:start"), str(start))
    sectPr.append(el)


def footer_page_number(section, link_previous=False):
    section.footer.is_linked_to_previous = link_previous
    if link_previous:
        return
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    field(p, "PAGE", "1")
    for run in p.runs:
        run.font.name = BODY_FONT
        run.font.size = Pt(11)


# --- inline markdown --------------------------------------------------------
TOKEN = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*(?!\*).+?(?<!\*)\*(?!\*)|`[^`]+`)")


def add_inline(paragraph, text: str, size=12, base_italic=False):
    text = text.replace("—", "—")
    for part in TOKEN.split(text):
        if not part:
            continue
        bold = italic = mono = False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, italic = part[1:-1], True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            part, mono = part[1:-1], True
        run = paragraph.add_run(part)
        run.bold = bold
        run.italic = italic or base_italic
        run.font.name = "Consolas" if mono else BODY_FONT
        run.font.size = Pt(size - 1 if mono else size)
    if not paragraph.runs:
        paragraph.add_run("")


def body_para(doc, text, style=None, size=12, justify=True, space_after=10):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(space_after)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_inline(p, text, size=size)
    return p


def caption(doc, text, style_name):
    p = doc.add_paragraph(style=style_name)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(14)
    add_inline(p, text, size=10)
    return p


def add_table(doc, rows, tbl_caption=None):
    if tbl_caption:
        caption(doc, tbl_caption, "TableCaption")
    ncols = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j in range(ncols):
            cell = cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            txt = row[j] if j < len(row) else ""
            add_inline(p, txt, size=10)
            if i == 0:
                for run in p.runs:
                    run.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return t


def bullet(doc, text, numbered=False):
    p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(4)
    add_inline(p, text, size=12)
    return p


def code_block(doc, lines):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.left_indent = Inches(0.3)
    pf.space_after = Pt(12)
    for i, ln in enumerate(lines):
        run = p.add_run(ln)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        if i < len(lines) - 1:
            run.add_break()
    return p


def heading(doc, text, level):
    h = doc.add_heading(level=level)
    h.paragraph_format.space_before = Pt(16 if level <= 2 else 12)
    h.paragraph_format.space_after = Pt(8)
    h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = h.add_run(text)
    run.font.name = BODY_FONT
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = True
    run.font.size = Pt({1: 17, 2: 14, 3: 12.5}.get(level, 12))
    return h


# --- markdown block parser --------------------------------------------------
SKIP_LINE = re.compile(r"^\s*(\*Draft word count|\*\*Figures to insert:|<div|</div>)")


def render_markdown(doc, md: str, figures_by_section=None, drop_h1=True):
    md = re.sub(r"^>.*$", "", md, flags=re.M)          # draft note blocks
    lines = md.split("\n")
    i = 0
    current_section = None
    pending_table_caption = None

    def flush_figures():
        nonlocal current_section
        if figures_by_section and current_section in figures_by_section:
            for fn, cap in figures_by_section[current_section]:
                path = FIGDIR / fn
                if not path.exists():
                    print(f"  MISSING FIGURE: {fn}")
                    continue
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.add_run().add_picture(str(path), width=Inches(5.6))
                caption(doc, cap, "FigureCaption")
            figures_by_section.pop(current_section, None)

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line or line == "---" or SKIP_LINE.match(line):
            i += 1
            continue

        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            code_block(doc, block)
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 and drop_h1:
                i += 1
                continue
            flush_figures()
            heading(doc, text, min(level, 3))
            if level == 2:
                current_section = text
            i += 1
            continue

        # a bold standalone line immediately before a table is its caption
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                add_table(doc, rows, pending_table_caption)
            pending_table_caption = None
            continue

        m = re.match(r"^[-*]\s+(.*)$", line)
        if m:
            bullet(doc, m.group(1))
            i += 1
            continue

        m = re.match(r"^\d+\.\s+(.*)$", line)
        if m:
            text = m.group(1)
            i += 1
            while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                text += " " + lines[i].strip()
                i += 1
            bullet(doc, text, numbered=True)
            continue

        # paragraph: gather until blank
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*(\||#{1,4}\s|[-*]\s|\d+\.\s|```|---$)", lines[i]):
            if SKIP_LINE.match(lines[i].strip()):
                break
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf)

        nxt = lines[i].strip() if i < len(lines) else ""
        if re.fullmatch(r"\*\*Table [^*]+\*\*", text) and nxt.startswith("|"):
            pending_table_caption = text.strip("*")
            continue
        body_para(doc, text)

    flush_figures()


# --- document ---------------------------------------------------------------
def main():
    doc = Document()

    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    st.paragraph_format.space_after = Pt(10)

    for name, base in (("FigureCaption", "Caption"), ("TableCaption", "Caption")):
        s = doc.styles.add_style(name, 1)
        s.base_style = doc.styles[base]
        s.font.name = BODY_FONT
        s.font.size = Pt(10)
        s.font.italic = True
        s.font.color.rgb = RGBColor(0, 0, 0)

    # A4, not python-docx's US Letter default: this is a UK submission.
    def page_setup(s):
        s.page_width = Mm(210)
        s.page_height = Mm(297)
        s.top_margin = s.bottom_margin = Inches(1)
        s.left_margin = Inches(1.25)   # extra for binding
        s.right_margin = Inches(1)

    sec = doc.sections[0]
    page_setup(sec)

    # ---------- title page (no page number) ----------
    for _ in range(4):
        doc.add_paragraph()
    for text, size, bold in [
        ("AI-Based Deepfake Detection Using Machine Learning", 20, True),
        ("with Robustness Evaluation", 20, True),
        ("", 12, False),
        ("", 12, False),
        ("Muhammad Obaid Ur Rehman", 14, False),
        ("STU1215887", 12, False),
        ("", 12, False),
        ("MSc Advanced Computing Project (COM7014)", 13, False),
        ("MSc Cyber Security", 13, False),
        ("Berlin School of Computing, Arden University", 13, False),
        ("", 12, False),
        ("Supervisor: Dr. Muhammad Saqib", 12, False),
        ("Second marker: Mr. Sharmarke Gabayre", 12, False),
        ("", 12, False),
        ("Submitted October 2026", 12, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(text)
        r.bold = bold
        r.font.name = BODY_FONT
        r.font.size = Pt(size)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p.add_run("Submitted in partial fulfilment of the requirements for the "
                  "degree of MSc Computing (Advanced Computing)")
    r.italic = True
    r.font.name = BODY_FONT
    r.font.size = Pt(11)

    sec.different_first_page_header_footer = False
    page_numbering(sec, "lowerRoman", 1)
    footer_page_number(sec)

    fm = DOCS.joinpath("ch0_frontmatter.md").read_text(encoding="utf-8")

    def slice_between(text, start, end=None):
        a = text.index(start)
        b = text.index(end) if end else len(text)
        return text[a:b]

    # ---------- abstract ----------
    doc.add_page_break()
    heading(doc, "Abstract", 1)
    abstract = slice_between(fm, "## Abstract", "## Acknowledgements")
    abstract = re.sub(r"^##.*$", "", abstract, flags=re.M)
    abstract = re.sub(r"^\*Target.*$", "", abstract, flags=re.M)
    render_markdown(doc, abstract)

    # ---------- acknowledgements ----------
    doc.add_page_break()
    heading(doc, "Acknowledgements", 1)
    ack = slice_between(fm, "## Acknowledgements", "## Abbreviations")
    ack = re.sub(r"^##.*$", "", ack, flags=re.M)
    render_markdown(doc, ack)

    # ---------- contents ----------
    doc.add_page_break()
    heading(doc, "Table of Contents", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    field(p, r'TOC \o "1-3" \h \z \u')

    doc.add_page_break()
    heading(doc, "List of Figures", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    field(p, r'TOC \h \z \t "FigureCaption,1"')

    doc.add_page_break()
    heading(doc, "List of Tables", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    field(p, r'TOC \h \z \t "TableCaption,1"')

    # ---------- abbreviations ----------
    doc.add_page_break()
    heading(doc, "Abbreviations", 1)
    abbr = slice_between(fm, "## Abbreviations")
    abbr = re.sub(r"^##.*$", "", abbr, flags=re.M)
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in abbr.split("\n")
            if ln.strip().startswith("|") and not re.fullmatch(
                r"\|[\s:|-]+\|", ln.strip())]
    rows = [r for r in rows if any(r)]
    add_table(doc, [["Abbreviation", "Expansion"]] + rows)

    # ---------- body: new section, arabic from 1 ----------
    body = doc.add_section(WD_SECTION.NEW_PAGE)
    page_setup(body)
    page_numbering(body, "decimal", 1)
    footer_page_number(body)

    for idx, (fname, title) in enumerate(FILES):
        if idx:
            doc.add_page_break()
        heading(doc, title, 1)
        md = DOCS.joinpath(fname).read_text(encoding="utf-8")
        figs = FIGURES if fname == "ch6_results.md" else None
        render_markdown(doc, md, figures_by_section=figs)

    # ---------- references ----------
    doc.add_page_break()
    heading(doc, "References", 1)
    refs = DOCS.joinpath("references.md").read_text(encoding="utf-8")
    refs = re.sub(r"^#\s+References\s*$", "", refs, flags=re.M)
    for block in [b.strip() for b in refs.split("\n\n") if b.strip()]:
        text = " ".join(block.split())
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        pf.space_after = Pt(10)
        pf.left_indent = Inches(0.5)
        pf.first_line_indent = Inches(-0.5)   # hanging indent
        add_inline(p, text, size=12)

    # ---------- appendices ----------
    doc.add_page_break()
    apx = DOCS.joinpath("appendices.md").read_text(encoding="utf-8")
    apx = re.sub(r"^#\s+Appendices\s*$", "", apx, flags=re.M)
    apx = apx.replace("\n# Appendix", "\n<<PB>>\n# Appendix")
    for n, chunk in enumerate(apx.split("<<PB>>")):
        if not chunk.strip():
            continue
        if n:
            doc.add_page_break()
        render_markdown(doc, chunk, drop_h1=False)

    doc.save(OUT)
    print(f"saved {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print(f"figures placed: {TOTAL_FIGURES}")


if __name__ == "__main__":
    main()
