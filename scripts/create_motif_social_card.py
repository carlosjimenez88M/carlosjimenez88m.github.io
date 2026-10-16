"""Render the motif-return essay's sharing card using the site's typography."""
from pathlib import Path
from PIL import Image, ImageDraw
from create_social_cards import font

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'static/img/social/motif-returns.png'


def main():
    image = Image.new('RGB', (1200, 630), '#f6f4ee')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 12, 630), fill='#2465a4')
    draw.text((66, 43), 'THE PROBABILITY ENGINE', fill='#242424', font=font(24))
    draw.line((66, 96, 1134, 96), fill='#cdd0cc', width=2)
    draw.text((66, 131), 'MUSIC · EMBEDDINGS · RAG', fill='#2465a4', font=font(22))
    title = ['When the motif returns,', 'does its meaning', 'return too?']
    for i, line in enumerate(title):
        face = font(66, True)
        assert draw.textbbox((0, 0), line, font=face)[2] < 1068
        draw.text((62, 178 + i * 79), line, fill='#202d37', font=face)
    subtitle = 'The Beatles · Metallica · Maluma featuring Farina'
    assert draw.textbbox((0, 0), subtitle, font=font(25))[2] < 1068
    draw.text((66, 454), subtitle, fill='#525c64', font=font(25))
    draw.line((66, 528, 1134, 528), fill='#cdd0cc', width=2)
    draw.text((66, 559), 'carlosdanieljimenez.com', fill='#2465a4', font=font(22))
    draw.text((891, 559), 'Essays & experiments', fill='#525c64', font=font(20))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    image.save(DEST, optimize=True)
    print(DEST)


if __name__ == '__main__':
    main()
