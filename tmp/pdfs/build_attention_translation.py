from pathlib import Path
from re import sub
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"E:\_WORKING")
SOURCE = ROOT / "tmp" / "pdfs" / "attention_translation.md"
FIGURE_PAGES = ROOT / "tmp" / "pdfs" / "attention_figures"
TMP = ROOT / "tmp" / "pdfs" / "attention_translation"
OUTPUT = ROOT / "output" / "pdf" / "Attention_Is_All_You_Need_中文精译.pdf"
TMP.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(
    TTFont("MicrosoftYaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0)
)
pdfmetrics.registerFont(
    TTFont("MicrosoftYaHei-Bold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0)
)
pdfmetrics.registerFontFamily(
    "MicrosoftYaHei",
    normal="MicrosoftYaHei",
    bold="MicrosoftYaHei-Bold",
    italic="MicrosoftYaHei",
    boldItalic="MicrosoftYaHei-Bold",
)

base = getSampleStyleSheet()
styles = {
    "body": ParagraphStyle(
        "CNBody",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=9.4,
        leading=15.6,
        alignment=TA_JUSTIFY,
        firstLineIndent=18.8,
        spaceAfter=5,
        wordWrap="CJK",
        textColor=colors.HexColor("#20242A"),
    ),
    "body0": ParagraphStyle(
        "CNBodyNoIndent",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=9.4,
        leading=15.6,
        alignment=TA_JUSTIFY,
        firstLineIndent=0,
        spaceAfter=5,
        wordWrap="CJK",
        textColor=colors.HexColor("#20242A"),
    ),
    "title": ParagraphStyle(
        "CNTitle",
        parent=base["Title"],
        fontName="MicrosoftYaHei-Bold",
        fontSize=24,
        leading=33,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=9,
        wordWrap="CJK",
    ),
    "subtitle": ParagraphStyle(
        "CNSubtitle",
        parent=base["Normal"],
        fontName="MicrosoftYaHei",
        fontSize=11.5,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4A5D6B"),
        spaceAfter=7,
        wordWrap="CJK",
    ),
    "h2": ParagraphStyle(
        "CNSection",
        parent=base["Heading1"],
        fontName="MicrosoftYaHei-Bold",
        fontSize=15.5,
        leading=23,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=13,
        spaceAfter=7,
        keepWithNext=True,
        wordWrap="CJK",
    ),
    "h3": ParagraphStyle(
        "CNSubsection",
        parent=base["Heading2"],
        fontName="MicrosoftYaHei-Bold",
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#1B5E6D"),
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
        wordWrap="CJK",
    ),
    "h4": ParagraphStyle(
        "CNSubsubsection",
        parent=base["Heading3"],
        fontName="MicrosoftYaHei-Bold",
        fontSize=10.4,
        leading=16,
        textColor=colors.HexColor("#276C77"),
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
        wordWrap="CJK",
    ),
    "note": ParagraphStyle(
        "CNNote",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=8.2,
        leading=13,
        leftIndent=5 * mm,
        rightIndent=5 * mm,
        firstLineIndent=0,
        spaceAfter=6,
        wordWrap="CJK",
        textColor=colors.HexColor("#52616B"),
        backColor=colors.HexColor("#EEF5F7"),
        borderColor=colors.HexColor("#B7CDD5"),
        borderWidth=0.5,
        borderPadding=6,
    ),
    "bullet": ParagraphStyle(
        "CNBullet",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=9.4,
        leading=15.6,
        leftIndent=8 * mm,
        firstLineIndent=-4 * mm,
        spaceAfter=4,
        wordWrap="CJK",
        textColor=colors.HexColor("#20242A"),
    ),
    "equation": ParagraphStyle(
        "CNEquation",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=10.4,
        leading=18,
        alignment=TA_CENTER,
        firstLineIndent=0,
        spaceBefore=4,
        spaceAfter=7,
        textColor=colors.HexColor("#102A43"),
    ),
    "caption": ParagraphStyle(
        "CNCaption",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=8.3,
        leading=13,
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceBefore=3,
        spaceAfter=8,
        wordWrap="CJK",
        textColor=colors.HexColor("#344955"),
    ),
    "reference": ParagraphStyle(
        "Reference",
        parent=base["BodyText"],
        fontName="MicrosoftYaHei",
        fontSize=7.3,
        leading=10.4,
        leftIndent=5 * mm,
        firstLineIndent=-5 * mm,
        spaceAfter=3,
        wordWrap="CJK",
        textColor=colors.HexColor("#252525"),
    ),
}


def inline_markup(text):
    safe = escape(text)
    safe = sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = safe.replace("  ", "&nbsp;&nbsp;")
    return safe


def para(text, style="body"):
    return Paragraph(inline_markup(text), styles[style])


def crop(source_name, box, target_name):
    source = PILImage.open(FIGURE_PAGES / source_name).convert("RGB")
    target = TMP / target_name
    source.crop(box).save(target, quality=92, optimize=True)
    return target


figures = {
    "fig1": (crop("page-03.png", (300, 90, 1740, 1320), "figure_1.jpg"), 126 * mm, 108 * mm),
    "fig2": (crop("page-04.png", (260, 80, 1780, 760), "figure_2.jpg"), 142 * mm, 63 * mm),
    "fig3": (crop("page-13.png", (240, 210, 1800, 1670), "figure_3.jpg"), 150 * mm, 140 * mm),
    "fig4": (crop("page-14.png", (120, 200, 1930, 2220), "figure_4.jpg"), 151 * mm, 168 * mm),
    "fig5": (crop("page-15.png", (160, 220, 1920, 2230), "figure_5.jpg"), 148 * mm, 169 * mm),
}


def table_from_lines(lines):
    rows = []
    for line in lines:
        cells = [item.strip() for item in line.strip().strip("|").split("|")]
        if all(set(item) <= {"-", ":"} for item in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    col_count = max(len(row) for row in rows)
    rows = [row + [""] * (col_count - len(row)) for row in rows]
    usable = 166 * mm
    if col_count == 3:
        widths = [0.47 * usable, 0.32 * usable, 0.21 * usable]
    elif col_count == 4:
        widths = [0.28 * usable, 0.25 * usable, 0.22 * usable, 0.25 * usable]
    elif col_count == 5:
        widths = [0.36 * usable, 0.14 * usable, 0.14 * usable, 0.18 * usable, 0.18 * usable]
    elif col_count == 6:
        widths = [0.08 * usable, 0.18 * usable, 0.40 * usable, 0.12 * usable, 0.12 * usable, 0.10 * usable]
    else:
        widths = [usable / col_count] * col_count
    font_size = 5.5 if len(rows) > 16 else 6.6
    cell_style = ParagraphStyle(
        "TableCell",
        fontName="MicrosoftYaHei",
        fontSize=font_size,
        leading=font_size + 3,
        alignment=TA_CENTER,
        wordWrap="CJK",
        textColor=colors.HexColor("#20242A"),
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="MicrosoftYaHei-Bold",
        textColor=colors.HexColor("#17324D"),
    )
    wrapped = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else cell_style
        wrapped.append([Paragraph(inline_markup(item), style) for item in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF0")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9AAAB4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for idx in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#F6F9FA")))
    table.setStyle(TableStyle(commands))
    return [table, Spacer(1, 3 * mm)]


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#C7D6DD"))
    canvas.setLineWidth(0.45)
    canvas.line(22 * mm, height - 16 * mm, width - 22 * mm, height - 16 * mm)
    canvas.setFillColor(colors.HexColor("#60727D"))
    canvas.setFont("MicrosoftYaHei", 7.4)
    if doc.page > 1:
        canvas.drawString(22 * mm, height - 12.5 * mm, "Attention Is All You Need · 中文精译")
    canvas.drawCentredString(width / 2, 10.5 * mm, str(doc.page))
    canvas.restoreState()


lines = SOURCE.read_text(encoding="utf-8").splitlines()
story = []
paragraph_buffer = []
in_references = False
index = 0


def flush_paragraph():
    if paragraph_buffer:
        text = " ".join(item.strip() for item in paragraph_buffer).strip()
        if text:
            style = "reference" if in_references and text.startswith("[") else "body"
            if text.startswith("图 ") or text.startswith("表 "):
                style = "caption"
            story.append(para(text, style))
        paragraph_buffer.clear()


while index < len(lines):
    line = lines[index].rstrip()
    stripped = line.strip()
    if not stripped:
        flush_paragraph()
        index += 1
        continue
    if stripped == "<!-- PAGEBREAK -->":
        flush_paragraph()
        story.append(PageBreak())
        index += 1
        continue
    if stripped.startswith("|"):
        flush_paragraph()
        table_lines = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            table_lines.append(lines[index].strip())
            index += 1
        story.extend(table_from_lines(table_lines))
        continue
    if stripped.startswith("![") and "](" in stripped:
        flush_paragraph()
        key = stripped.split("](", 1)[1].split(")", 1)[0]
        path, width, height = figures[key]
        story.append(Image(str(path), width=width, height=height, hAlign="CENTER"))
        index += 1
        continue
    if stripped.startswith("# "):
        flush_paragraph()
        story.append(Spacer(1, 23 * mm))
        story.append(para(stripped[2:], "title"))
        index += 1
        continue
    if stripped.startswith("## "):
        flush_paragraph()
        heading = stripped[3:]
        in_references = heading.startswith("参考文献")
        story.append(para(heading, "h2"))
        index += 1
        continue
    if stripped.startswith("### "):
        flush_paragraph()
        story.append(para(stripped[4:], "h3"))
        index += 1
        continue
    if stripped.startswith("#### "):
        flush_paragraph()
        story.append(para(stripped[5:], "h4"))
        index += 1
        continue
    if stripped.startswith("> "):
        flush_paragraph()
        story.append(para(stripped[2:], "note"))
        index += 1
        continue
    if stripped.startswith("- "):
        flush_paragraph()
        story.append(para("• " + stripped[2:], "bullet"))
        index += 1
        continue
    if stripped.startswith("$$"):
        flush_paragraph()
        story.append(para(stripped[2:].strip(), "equation"))
        index += 1
        continue
    if stripped.startswith("CENTER:"):
        flush_paragraph()
        centered = ParagraphStyle(
            "CenteredDynamic",
            parent=styles["body0"],
            alignment=TA_CENTER,
            fontSize=9.5,
            leading=15,
        )
        story.append(Paragraph(inline_markup(stripped[7:].strip()), centered))
        index += 1
        continue
    paragraph_buffer.append(stripped)
    index += 1

flush_paragraph()

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=22 * mm,
    leftMargin=22 * mm,
    topMargin=22 * mm,
    bottomMargin=18 * mm,
    title="Attention Is All You Need - 中文精译",
    author="Ashish Vaswani et al.；中文精译",
    subject="Transformer 论文中文精译，参考文献保留英文原文",
)
doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
print(OUTPUT)
