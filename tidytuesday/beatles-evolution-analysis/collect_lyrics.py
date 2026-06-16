"""
Beatles lyrical-evolution study — data collection.

Fetches the lyrics of four Beatles albums from the Genius API song-by-song
(canonical UK tracklists) and persists raw + processed corpora.

Album-level search on Genius is unreliable for the Beatles (it merges
compilations, singles and foreign-language versions — Sgt. Pepper's came back
with 216 "tracks"). Song-by-song retrieval against fixed tracklists is fully
reproducible and gives clean per-album control.

Albums (chronological — the spine of the evolution arc):
    Rubber Soul                                  (Dec 1965)
    Revolver                                     (Aug 1966)
    Sgt. Pepper's Lonely Hearts Club Band        (Jun 1967)
    Abbey Road                                   (Sep 1969)
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from lyricsgenius import Genius

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
load_dotenv(REPO / ".env")

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

ARTIST = "The Beatles"

ALBUMS = [
    {
        "title": "Rubber Soul", "year": 1965, "order": 1,
        "tracks": [
            "Drive My Car", "Norwegian Wood (This Bird Has Flown)", "You Won't See Me",
            "Nowhere Man", "Think for Yourself", "The Word", "Michelle",
            "What Goes On", "Girl", "I'm Looking Through You", "In My Life",
            "Wait", "If I Needed Someone", "Run for Your Life",
        ],
    },
    {
        "title": "Revolver", "year": 1966, "order": 2,
        "tracks": [
            "Taxman", "Eleanor Rigby", "I'm Only Sleeping", "Love You To",
            "Here, There and Everywhere", "Yellow Submarine", "She Said She Said",
            "Good Day Sunshine", "And Your Bird Can Sing", "For No One",
            "Doctor Robert", "I Want to Tell You", "Got to Get You into My Life",
            "Tomorrow Never Knows",
        ],
    },
    {
        "title": "Sgt. Pepper's Lonely Hearts Club Band", "year": 1967, "order": 3,
        "tracks": [
            "Sgt. Pepper's Lonely Hearts Club Band", "With a Little Help from My Friends",
            "Lucy in the Sky with Diamonds", "Getting Better", "Fixing a Hole",
            "She's Leaving Home", "Being for the Benefit of Mr. Kite!",
            "Within You Without You", "When I'm Sixty-Four", "Lovely Rita",
            "Good Morning Good Morning",
            "Sgt. Pepper's Lonely Hearts Club Band (Reprise)", "A Day in the Life",
        ],
    },
    {
        "title": "Abbey Road", "year": 1969, "order": 4,
        "tracks": [
            "Come Together", "Something", "Maxwell's Silver Hammer", "Oh! Darling",
            "Octopus's Garden", "I Want You (She's So Heavy)", "Here Comes the Sun",
            "Because", "You Never Give Me Your Money", "Sun King", "Mean Mr. Mustard",
            "Polythene Pam", "She Came In Through the Bathroom Window",
            "Golden Slumbers", "Carry That Weight", "The End", "Her Majesty",
        ],
    },
]

SECTION_RE = re.compile(r"^\[.*?\]$")
NOISE_RE = re.compile(r"(\d+\s*Contributors|Translations|You might also like|Embed$)", re.I)


def clean_lyrics(raw: str) -> list[str]:
    """Return a list of cleaned, non-empty lyric lines."""
    if not raw:
        return []
    raw = re.sub(r"^.*?Lyrics", "", raw, count=1, flags=re.S)  # drop "<Song> Lyrics" header
    lines: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or SECTION_RE.match(ln):
            continue
        if NOISE_RE.search(ln):
            ln = NOISE_RE.sub("", ln).strip()
        ln = re.sub(r"\d*Embed$", "", ln).strip()
        if ln:
            lines.append(ln)
    return lines


def main() -> None:
    token = os.getenv("GENIUS_API_TOKEN")
    assert token, "GENIUS_API_TOKEN missing"
    genius = Genius(
        token, timeout=20, retries=3,
        remove_section_headers=True, skip_non_songs=True, verbose=False,
    )

    corpus_lines: list[dict] = []
    corpus_songs: list[dict] = []

    for alb in ALBUMS:
        print(f"\n=== {alb['title']} ({alb['year']}) — {len(alb['tracks'])} tracks ===")
        for tnum, track_title in enumerate(alb["tracks"], start=1):
            song = genius.search_song(track_title, ARTIST)
            if song is None:
                print(f"   !! {tnum:02d}. {track_title}: NOT FOUND")
                continue
            lines = clean_lyrics(song.lyrics)
            if not lines:
                print(f"   -- {tnum:02d}. {track_title}: NO LYRICS")
                continue
            print(f"   .. {tnum:02d}. {track_title}: {len(lines)} lines")
            corpus_songs.append({
                "album": alb["title"], "album_order": alb["order"], "year": alb["year"],
                "track_num": tnum, "title": track_title,
                "n_lines": len(lines),
                "n_words": sum(len(l.split()) for l in lines),
                "full_text": "\n".join(lines),
            })
            for i, ln in enumerate(lines):
                corpus_lines.append({
                    "album": alb["title"], "album_order": alb["order"], "year": alb["year"],
                    "track_num": tnum, "title": track_title,
                    "line_num": i, "line_text": ln, "word_count": len(ln.split()),
                })
            time.sleep(0.5)

    df_lines = pd.DataFrame(corpus_lines)
    df_songs = pd.DataFrame(corpus_songs)
    for df, name in [(df_lines, "corpus_lines"), (df_songs, "corpus_songs")]:
        df.to_parquet(PROC / f"{name}.parquet", index=False)
        df.to_csv(PROC / f"{name}.csv", index=False)

    print("\n=== SUMMARY ===")
    summary = (df_songs.groupby(["album_order", "album", "year"])
               .agg(songs=("title", "nunique"), lines=("n_lines", "sum"),
                    words=("n_words", "sum"))
               .reset_index())
    print(summary.to_string(index=False))
    print(f"\nTotal songs: {df_songs.title.nunique()}  |  total lines: {len(df_lines)}")


if __name__ == "__main__":
    main()
