"""Generate the blog's typographic sharing cards (requires Pillow).

Run from any directory. Cards are designed at 1200 x 630 pixels; they contain
editorial titles, not charts or claimed experimental results.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "static/img/social"


def font(size, serif=False):
    candidates = (
        ["/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"] if serif else
        ["/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install Georgia/Arial or DejaVu fonts before rendering cards.")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cards = {
        "probability-engine": ("MUSIC · MEANING · AI", ["Music, meaning,", "and reliable AI."], "Research and engineering by Carlos Daniel Jiménez"),
        "memory-is-not-context": ("AGENT MEMORY · EVALUATION", ["Memory is not", "context."], "Follow the evidence. Account for the whole trajectory."),
        "album-memory": ("MUSIC · AGENT MEMORY", ["What should an", "agent remember?"], "An experiment across four album narratives."),
        "prompts-release-artifacts": ("AI ENGINEERING · EVALUATION", ["Prompts are", "release artifacts."], "From an observed failure to a controlled change."),
        "aquamosh": ("MUSIC · MULTILINGUAL RETRIEVAL", ["When lyrics", "change language."], "An exploratory embedding study of Aquamosh."),
    }
    for name, (label, lines, subtitle) in cards.items():
        image = Image.new("RGB", (1200, 630), "#f6f4ee")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 12, 630), fill="#2465a4")
        draw.text((66, 47), "THE PROBABILITY ENGINE", fill="#242424", font=font(24))
        draw.line((66, 99, 1134, 99), fill="#cdd0cc", width=2)
        draw.text((66, 145), label, fill="#2465a4", font=font(22))
        for i, line in enumerate(lines):
            assert draw.textbbox((0, 0), line, font=font(72, True))[2] < 1050
            draw.text((62, 207 + i * 91), line, fill="#202d37", font=font(72, True))
        draw.text((66, 440), subtitle, fill="#525c64", font=font(25))
        draw.line((66, 528, 1134, 528), fill="#cdd0cc", width=2)
        draw.text((66, 559), "carlosdanieljimenez.com", fill="#2465a4", font=font(22))
        draw.text((891, 559), "Essays & experiments", fill="#525c64", font=font(20))
        image.save(DEST / f"{name}.png", optimize=True)
    print(f"Generated {len(cards)} sharing cards in {DEST}")


if __name__ == "__main__":
    main()
