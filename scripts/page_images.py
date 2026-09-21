"""Draw the page's icons and link-preview image into deploy/cloudflare/public/.

    uv run --with pillow python scripts/page_images.py

Black on white in Monaco, the page's fallback font on macOS (Lucida Console ships with
Windows only). The icon is the page's favicon.svg: three narrowing bars, a sieve.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PUBLIC = Path(__file__).resolve().parents[1] / "deploy" / "cloudflare" / "public"
FONT = "/System/Library/Fonts/Monaco.ttf"
# favicon.svg's bars on its 32-unit grid: (x, y, width, height).
BARS = [(4, 5, 24, 5), (8, 13, 16, 5), (13, 21, 6, 6)]


def icon(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    unit = size / 32
    for x, y, w, h in BARS:
        box = [
            round(x * unit),
            round(y * unit),
            round((x + w) * unit) - 1,
            round((y + h) * unit) - 1,
        ]
        draw.rectangle(box, fill="black")
    return image


def preview() -> Image.Image:
    """1200x630, the size Open Graph and X large cards expect."""
    image = Image.new("RGB", (1200, 630), "white")
    draw = ImageDraw.Draw(image)
    font = lambda size: ImageFont.truetype(FONT, size)
    image.paste(icon(96), (80, 72))
    draw.text((200, 82), "sifty", font=font(64), fill="black")
    draw.text((80, 236), "Text in. Label out.", font=font(56), fill="black")
    draw.text((80, 322), "Free text classification API. No key.", font=font(30), fill="black")
    draw.line([(80, 424), (1120, 424)], fill="black", width=2)
    draw.text(
        (80, 452), "$ curl sifty.dev/spam,not+spam/Win+a+free+iPhone", font=font(26), fill="black"
    )
    draw.text((80, 500), "spam", font=font(26), fill="black")
    return image


def main() -> None:
    preview().save(PUBLIC / "og.png", optimize=True)
    icon(180).save(PUBLIC / "apple-touch-icon.png", optimize=True)
    icon(192).save(PUBLIC / "icon-192.png", optimize=True)
    icon(512).save(PUBLIC / "icon-512.png", optimize=True)
    icon(48).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])


if __name__ == "__main__":
    main()
