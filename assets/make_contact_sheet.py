#!/usr/bin/env python3
"""Build a docs contact sheet from rendered animation frames (2x5 grid)."""
import sys
from PIL import Image, ImageDraw, ImageFont

frames = [(0, "t=0.0 远景"), (50, "t=1.7 接近"), (84, "t=2.8 进入"), (105, "t=3.5 盒中(浅)"),
          (123, "t=4.1 快掉头"), (132, "t=4.4 锁定金属球"), (156, "t=5.2 下探"),
          (177, "t=5.9 掠过玻璃球"), (198, "t=6.6 出盒回望"), (240, "t=8.0 复位原视角")]
src, out = sys.argv[1], sys.argv[2]
tw, th = 426, 240
sheet = Image.new("RGB", (tw * 5, (th + 26) * 2), (18, 18, 18))
try:
    font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 15)
except Exception:
    font = ImageFont.load_default()
for i, (f, label) in enumerate(frames):
    im = Image.open(f"{src}/frame_{f:04d}.png").convert("RGB").resize((tw, th), Image.LANCZOS)
    x, y = (i % 5) * tw, (i // 5) * (th + 26)
    sheet.paste(im, (x, y + 26))
    d = ImageDraw.Draw(sheet)
    d.text((x + 6, y + 5), label, fill=(220, 220, 220), font=font)
sheet.save(out, optimize=True)
print(f"wrote {out} ({sheet.size[0]}x{sheet.size[1]})")
