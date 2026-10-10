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


def pytorch():
    im=Image.new("RGB",(1000,480),"#f6eeeb")
    d=ImageDraw.Draw(im)
    ink,rule="#8a4a3b","#d6b8ae"
    ramp=lambda v: tuple(round(a+(b-a)*v) for a,b in zip((238,214,205),(170,66,38)))
    label(d,(46,38),"04 / PYTORCH","#8a4a3b",17)
    label(d,(807,40),"loss.backward()",ink,15)
    label(d,(46,100),"STORAGE",ink,15)
    for i in range(12):
        d.rounded_rectangle((46+i*40,130,80+i*40,164),radius=3,fill=ramp(.1+i*.08))
    label(d,(46,196),"VIEW STRIDE (4, 1)",ink,15)
    label(d,(276,196),"T STRIDE (1, 4)",ink,15)
    for r in range(4):
        for c in range(4):
            if r<3:
                d.rounded_rectangle((46+c*40,228+r*40,80+c*40,262+r*40),radius=3,fill=ramp(.1+(r*4+c)*.08))
            if c<3:
                d.rounded_rectangle((276+c*40,228+r*40,310+c*40,262+r*40),radius=3,fill=ramp(.1+(c*4+r)*.08))
    d.line((560,100,560,388),fill=rule,width=2)
    nodes={"x":(600,132),"W":(600,232),"mm":(708,182),"relu":(816,182),"loss":(816,282)}
    for a,b in (("x","mm"),("W","mm"),("mm","relu")):
        (x1,y1),(x2,y2)=nodes[a],nodes[b]
        d.line((x1+70,y1+17,x2,y2+17),fill=rule,width=2)
    d.line((851,216,851,282),fill=rule,width=2)
    for k,(x,y) in nodes.items():
        leaf=k in ("x","W")
        d.rounded_rectangle((x,y,x+70,y+34),radius=4,fill="#e9d3cb" if leaf else ramp(.75),outline=None)
        label(d,(x+10,y+8),k,ink if leaf else "#fbf3f0",15)
    d.line((600,352,800,352),fill="#b98a7c",width=2)
    d.polygon(((600,347),(590,352),(600,357)),fill="#b98a7c")
    label(d,(612,362),"GRAD_FN CHAIN",ink,13)
    label(d,(46,421),"FROM STRIDES TO A TRAINED GPT.",ink,15)
    for i,c in enumerate((ramp(.15),ramp(.4),ramp(.65),ramp(.9))):
        d.rounded_rectangle((825+i*29,408,848+i*29,431),radius=3,fill=c)
    im.save(OUT/"pytorch.webp",quality=94)


if __name__ == "__main__":
    attention()
    inference()
    pytorch()
