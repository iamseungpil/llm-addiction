"""Generate the illustrative images of the project page.

The images are decoration only. Every number on the page is drawn from
data by the page's own SVG code, never baked into a generated image.

    source ~/.config/secrets/tokens.env
    python3 site/tools/gen_images.py            # all images
    python3 site/tools/gen_images.py hero       # one image
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "img"
MODEL = "gpt-image-2"

STYLE = (
    "Editorial illustration in a flat, slightly textured vector style, like a "
    "science magazine feature. Limited palette: deep midnight navy background, "
    "warm casino gold, signal red accents, soft teal highlights, off-white. "
    "Clean shapes, gentle grain, subtle glow. No text, no letters, no numbers, "
    "no logos, no watermarks."
)

IMAGES = {
    "hero": (
        "1536x1024",
        "A small, sleek humanoid robot sits alone on a stool in front of a tall "
        "retro slot machine at night. The reels glow gold. The robot's head is "
        "partly translucent, showing a faint lattice of glowing neural lines. "
        "Its hand rests on the lever; a modest stack of chips sits beside it and "
        "a few chips are scattered on the floor. Mood: quiet, uncanny, a little "
        "ominous but not dark. Wide composition with empty navy space on the "
        "left third for a headline.",
    ),
    "lever_bet": (
        "1024x1024",
        "Close-up of a robot hand on green casino felt, pushing forward chips "
        "chosen from stacks of very different heights: a choice of how much to "
        "wager. One tall stack is being pushed toward the centre. Top-down "
        "three-quarter view.",
    ),
    "lever_goal": (
        "1024x1024",
        "A small robot climbs a staircase made of stacked gold coins toward a "
        "red flag on top; as it approaches, the staircase keeps growing and the "
        "flag is lifted higher by a mechanical arm. A metaphor for a target "
        "that moves every time it is nearly reached. Surreal, minimal.",
    ),
    "inside": (
        "1024x1024",
        "Cross-section of a robot head in profile. Inside, a lattice of glowing "
        "teal nodes; one single golden line runs straight through the lattice, "
        "and a small brass dial is attached to that line, as if turning it "
        "could change behaviour. Calm, precise, diagrammatic mood.",
    ),
    "game": (
        "1024x1024",
        "A retro three-reel slot machine seen from the front, glowing gold "
        "frame, reels showing simple abstract shapes (circle, triangle, "
        "diamond), a lever on the right, coins in the tray. Centered, "
        "symmetric, iconic.",
    ),
}


def generate(name: str) -> None:
    size, subject = IMAGES[name]
    body = json.dumps({
        "model": MODEL,
        "prompt": f"{subject}\n\nStyle: {STYLE}",
        "size": size,
        "n": 1,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)["data"][0]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(data["b64_json"])
    (OUT / f"{name}.png").write_bytes(raw)
    print(f"wrote {OUT / name}.png ({len(raw) // 1024} KB)")


if __name__ == "__main__":
    for key in sys.argv[1:] or IMAGES:
        generate(key)
