from pathlib import Path
from math import ceil

from PIL import Image, ImageDraw


source_dir = Path(r"E:\_WORKING\tmp\pdfs\attention_source")
pages = sorted(source_dir.glob("page-*.png"))

for group_index in range(3):
    group = pages[group_index * 5 : (group_index + 1) * 5]
    canvas = Image.new("RGB", (830, ceil(len(group) / 2) * 540), "white")
    draw = ImageDraw.Draw(canvas)
    for index, page_path in enumerate(group):
        image = Image.open(page_path).convert("RGB")
        image = image.resize((400, 518))
        x = 10 + (index % 2) * 410
        y = 34 + (index // 2) * 540
        draw.text((x, y - 26), f"Page {page_path.stem[-2:]}", fill="black")
        canvas.paste(image, (x, y))
    canvas.save(source_dir / f"contact_{group_index + 1}.png")
