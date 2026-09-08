"""Regenerate the course's schematic cover images. Optional dependency: Pillow."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "theme" / "covers"
OUT.mkdir(exist_ok=True)


def font(size):
    for path in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def label(draw, xy, text, color, size=18):
    draw.text(xy, text, fill=color, font=font(size))


def attention():
    im = Image.new("RGB", (1000, 480), "#edf3ef")
    d = ImageDraw.Draw(im)
    label(d, (46, 38), "01 / THE TRANSFORMER", "#477367", 17)
    label(d, (46, 120), "Q", "#12664f", 54)
    label(d, (46, 191), "K", "#12664f", 54)
    label(d, (46, 262), "V", "#12664f", 54)
    for y in (155, 226, 297):
        d.line((110, y, 218, y), fill="#a4b9ad", width=2)
        for n in range(5):
            d.rounded_rectangle((135+n*13, y-15, 143+n*13, y-3), radius=2, fill="#9cc5b1")
    words = ["the", "model", "learns", "what", "to", "attend", "to"]
    for r in range(7):
        for c in range(7):
            x, y = 300+c*40, 100+r*40
            v = max(0, .82 - abs(c-r)*.18) if c <= r else .025
            color = tuple(round(a+(b-a)*v) for a,b in zip((221,233,225),(14,111,81)))
            d.rounded_rectangle((x,y,x+33,y+33),radius=3,fill=color)
        label(d,(600,103+r*40),words[r],"#416051",18)
    d.line((244, 100, 244, 379),fill="#a4b9ad",width=2)
    d.line((242, 100, 260, 100),fill="#a4b9ad",width=2)
    d.line((242, 379, 260, 379),fill="#a4b9ad",width=2)
    label(d,(46,421),"ATTENTION IS A LEARNED CONNECTION.","#477367",15)
    label(d,(807,40),"QK / sqrt(d)","#477367",15)
    for i, c in enumerate(("#c2d9cb","#8ebca3","#438a67","#116d50")):
        d.rounded_rectangle((825+i*29,408,848+i*29,431),radius=3,fill=c)
    im.save(OUT/"transformers.webp",quality=94)


def inference():
    im=Image.new("RGB",(1000,480),"#eef0f8")
    d=ImageDraw.Draw(im)
    label(d,(46,38),"02 / INFERENCE ENGINEERING","#616b8c",17)
    label(d,(46,104),"REQUESTS","#616b8c",15)
    colors=["#6b77af","#c18471","#699690","#9c83a4"]
    for r,c in enumerate(colors):
        y=158+r*57
        label(d,(46,y+8),f"R{r+1:02}","#616b8c",18)
        offset=r*47
        d.rounded_rectangle((138+offset,y,315+offset,y+35),radius=4,fill=c)
        for j in range(6-r):
            x=326+offset+j*46
            d.rounded_rectangle((x,y,x+34,y+35),radius=4,fill=c)
    d.line((770,100,770,368),fill="#c5cada",width=2)
    label(d,(807,104),"KV CACHE","#616b8c",15)
    for r in range(5):
        for c in range(3):
            d.rounded_rectangle((807+c*39,149+r*39,837+c*39,178+r*39),radius=3,fill=colors[(r+c)%4] if r<4 else "#d9dcea")
    label(d,(46,421),"PREFILL. DECODE. REPEAT.","#616b8c",15)
    d.line((360,429,715,429),fill="#a6aec7",width=2)
    d.polygon(((714,424),(724,429),(714,434)),fill="#a6aec7")
    label(d,(805,421),"TIME ->","#616b8c",15)
    im.save(OUT/"inference.webp",quality=94)


if __name__ == "__main__":
    attention()
    inference()
