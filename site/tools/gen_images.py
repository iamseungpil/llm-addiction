"""Generate the pixel-art sprites of the project page with gpt-image-2.

The sprites are decoration only. Every number on the page is drawn by the
page's own code from ``assets/data.js`` and ``assets/games.js``; no number is
baked into an image.

    source ~/.config/secrets/tokens.env
    python3 site/tools/gen_images.py                 # generate + process every sprite
    python3 site/tools/gen_images.py robot slot      # only these
    python3 site/tools/gen_images.py --process-only  # re-run the pixel pass on cached raws
    python3 site/tools/gen_images.py --og            # rebuild assets/img/og.jpg from the sprites

Raw 1024px PNGs are cached outside the site (``$SPRITE_RAW_DIR``, default
``$TMPDIR/llm-addiction-sprites``) so they are never deployed. The pixel pass
crops each raw to its content, shrinks it to a small native size with an
area filter, snaps it to a limited palette and a hard alpha edge, and writes
``assets/img/<name>.png``. The page shows them at 3-5x with
``image-rendering: pixelated``.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
OUT = SITE / "assets" / "img"
RAW = Path(os.environ.get("SPRITE_RAW_DIR", Path(tempfile.gettempdir()) / "llm-addiction-sprites"))
MODEL = "gpt-image-2"
NAVY = (11, 20, 38)

STYLE = (
    "16-bit pixel art, limited palette: deep navy #0b1426, neon gold #f3c14b, "
    "casino red #e0453a, teal #3fb6a8, off-white; crisp pixels, no anti-aliasing, "
    "no text, no letters, no numbers, no logos. A single isolated game sprite, "
    "centered, with generous empty margin, on a fully transparent background, "
    "no floor, no shadow, no scenery."
)

# name: (subject, native height in px of the processed sprite, palette size)
SPRITES = {
    "robot": (
        "A small friendly retro robot mascot standing, facing the viewer. Boxy "
        "off-white head with a navy face screen and two big square glowing teal "
        "eyes, a short antenna with a gold bulb, compact off-white body with a "
        "small red chest light, short arms, stubby legs. Cute, iconic, readable "
        "at small size.",
        96, 20,
    ),
    "slot": (
        "A classic three-reel casino slot machine seen straight from the front, "
        "perfectly symmetric. Gold cabinet with red trim and a row of round "
        "marquee light bulbs on top, a lever with a red ball on the right side, "
        "a coin tray at the bottom. The three reel windows are side by side in "
        "the middle and are EMPTY: three plain flat off-white rectangles with no "
        "symbols in them.",
        104, 24,
    ),
    "chip": (
        "One casino poker chip seen from a low three-quarter angle, red with "
        "off-white edge stripes and a gold ring. Just the chip.",
        24, 12,
    ),
    "coins": (
        "One short stack of about five gold coins seen from the side, flat "
        "tops, like one step of a staircase. Just the stack.",
        28, 12,
    ),
    "flag": (
        "A small red pennant flag on a thin gold pole, waving slightly. Just the "
        "flag and pole.",
        40, 10,
    ),
    "dial": (
        "A round brass control knob seen from the front, with one short "
        "off-white pointer line from the centre to the top edge, a teal ring "
        "around it. Just the knob.",
        40, 12,
    ),
}


def generate(name: str) -> Path:
    subject = SPRITES[name][0]
    payload = {
        "model": MODEL,
        "prompt": f"{subject}\n\nStyle: {STYLE}",
        "size": "1024x1024",
        "n": 1,
        "background": "transparent",
        "output_format": "png",
    }

    def call(body: dict) -> dict:
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)["data"][0]

    try:
        data = call(payload)
    except urllib.error.HTTPError as err:
        # Fall back to a flat navy backdrop that the pixel pass keys out.
        print(f"{name}: transparent request refused ({err.code}); retrying on navy")
        payload.pop("background")
        payload["prompt"] = payload["prompt"].replace(
            "on a fully transparent background", "on a flat solid deep navy #0b1426 background"
        )
        data = call(payload)
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.png"
    path.write_bytes(base64.b64decode(data["b64_json"]))
    print(f"raw  {path} ({path.stat().st_size // 1024} KB)")
    return path


def pixelate(name: str) -> None:
    from PIL import Image

    _, height, colors = SPRITES[name]
    img = Image.open(RAW / f"{name}.png").convert("RGBA")
    px = img.load()
    w, h = img.size
    # Key out a navy backdrop if the model did not return transparency.
    if px[2, 2][3] == 255:
        for y in range(h):
            for x in range(w):
                r, g, b, _a = px[x, y]
                if abs(r - NAVY[0]) + abs(g - NAVY[1]) + abs(b - NAVY[2]) < 40:
                    px[x, y] = (0, 0, 0, 0)
    alpha = img.getchannel("A").point(lambda a: 255 if a > 96 else 0)
    img.putalpha(alpha)
    img = img.crop(alpha.getbbox())
    width = max(1, round(img.width * height / img.height))
    small = img.resize((width, height), Image.Resampling.BOX)
    a = small.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    rgb = small.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = rgb.convert("RGBA")
    out.putalpha(a)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{name}.png"
    out.save(dest, optimize=True)
    print(f"sprite {dest} {out.size[0]}x{out.size[1]} ({dest.stat().st_size} B)")


def blink_frame() -> None:
    """robot-blink.png: the processed robot with its eyes shut.

    Paints the two eye boxes with the face-screen colour and draws a one-row
    teal lid line, so the two frames stay pixel-aligned for the CSS blink.
    The boxes were read off the processed 56x96 robot; re-check them if the
    robot sprite is regenerated.
    """
    from PIL import Image

    img = Image.open(OUT / "robot.png").convert("RGBA")
    px = img.load()
    face = px[27, 38]
    lid = (63, 182, 168, 255)
    for x0, x1 in ((17, 24), (31, 38)):
        for y in range(33, 42):
            for x in range(x0, x1 + 1):
                px[x, y] = face
        for x in range(x0, x1 + 1):
            px[x, 39] = lid
    img.save(OUT / "robot-blink.png", optimize=True)
    print(f"sprite {OUT / 'robot-blink.png'}")


def build_og() -> None:
    """1200x630 share card: sprites at 4x on navy, title set with ImageMagick."""
    from PIL import Image

    card = Image.new("RGB", (1200, 630), NAVY)
    for name, scale, pos in (("slot", 4, (790, 150)), ("robot", 4, (600, 182))):
        sp = Image.open(OUT / f"{name}.png").convert("RGBA")
        sp = sp.resize((sp.width * scale, sp.height * scale), Image.Resampling.NEAREST)
        card.paste(sp, pos, sp)
    tmp = RAW / "og_base.png"
    RAW.mkdir(parents=True, exist_ok=True)
    card.save(tmp)
    font = "/System/Library/Fonts/Menlo.ttc"
    subprocess.run(
        [
            "magick", str(tmp),
            "-font", font, "-fill", "#f3c14b", "-pointsize", "58",
            "-annotate", "+64+170", "Can an AI get\nhooked on\ngambling?",
            "-fill", "#e8edf6", "-pointsize", "26",
            "-annotate", "+64+470", "Can Large Language Models\nDevelop Gambling Addiction?",
            "-fill", "#3fb6a8", "-pointsize", "24",
            "-annotate", "+64+575", "NeurIPS 2026",
            "-quality", "86", str(OUT / "og.jpg"),
        ],
        check=True,
    )
    print(f"og   {OUT / 'og.jpg'} ({(OUT / 'og.jpg').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--og" in args:
        build_og()
        sys.exit(0)
    process_only = "--process-only" in args
    names = [a for a in args if not a.startswith("--")] or list(SPRITES)
    for key in names:
        if not process_only:
            generate(key)
        pixelate(key)
        if key == "robot":
            blink_frame()
