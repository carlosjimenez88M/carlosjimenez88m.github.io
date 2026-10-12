"""Collect private lyric records and export metadata-only coverage/provenance.

Do not publish the corpus or raw provider responses. Network calls are explicit;
reruns reuse the private cache unless --refresh is passed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode, quote

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / ".research-cache/fear-lyrics"
PUBLIC = Path(__file__).resolve().parent
UA = "ProbabilityEngine-FearLyrics/1.0 (https://carlosdanieljimenez.com)"
BAD_VERSION = re.compile(r"\b(live|unplugged|acoustic|demo|remix|karaoke|instrumental|translation|cover|sped up|slowed)\b", re.I)
LRC_TIMESTAMP_PREFIX = re.compile(r"^(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\])+")


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def artist_matches(observed: str, row: dict) -> bool:
    # Aliases are explicit corpus metadata, not fuzzy search against any artist.
    allowed = [row["artist"], *row.get("artist_aliases", [])]
    return any(normalized(observed).replace(" ", "") == normalized(value).replace(" ", "") for value in allowed)


def clean_lyrics(value: str) -> str:
    # Keep paragraph boundaries, pronouns, negation, and section order.
    lines = []
    for line in value.replace("\r", "").split("\n"):
        line = line.strip()
        # Some LRCLIB plainLyrics records contain synchronized LRC timestamps.
        # Remove only strict numeric tags at line starts; preserve section
        # headings, spoken passages and bracketed words in the lyric body.
        line = LRC_TIMESTAMP_PREFIX.sub("", line).strip()
        if re.search(r"\d+\s+Contributors|You might also like|^Translations$", line, re.I):
            continue
        if "Lyrics" in line and len(line) < 180 and not lines:
            continue
        line = re.sub(r"\d*Embed$", "", line).strip()
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def song_id(row: dict) -> str:
    base = f"{row['year']}|{normalized(row['artist'])}|{normalized(row['title'])}"
    return f"s{row['year']}-{hashlib.sha256(base.encode()).hexdigest()[:12]}"


def lyric_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def lrclib(row: dict, session: requests.Session) -> dict:
    if row.get('lrclib_record_id'):
        url = f"https://lrclib.net/api/get/{int(row['lrclib_record_id'])}"
        response = session.get(url, timeout=25)
        if response.status_code != 200: return {'status': response.status_code, 'source_url': url}
        item = response.json()
        if normalized(item.get('trackName','')) != normalized(row['title']) or not artist_matches(item.get('artistName',''),row):
            return {'status':'identity_mismatch', 'source_url':url}
        return {'status':200 if item.get('plainLyrics') else 'empty', 'lyrics':clean_lyrics(item.get('plainLyrics') or ''),
                'source_url':url, 'observed_title':item['trackName'], 'observed_artist':item['artistName'],
                'observed_album':item.get('albumName'), 'duration':item.get('duration'),
                'version_review':'Explicit record selected after language/recording review; prior search record retained privately.'}
    params = {"track_name": row["title"], "artist_name": row["artist"]}
    url = "https://lrclib.net/api/search?" + urlencode(params)
    response = session.get(url, timeout=25)
    if response.status_code != 200:
        return {"status": response.status_code, "source_url": url}
    candidates = []
    for item in response.json():
        if normalized(item.get("trackName", "")) != normalized(row["title"]):
            continue
        if not artist_matches(item.get("artistName", ""), row):
            continue
        if BAD_VERSION.search(item.get("albumName", "")):
            continue
        if not item.get("plainLyrics"):
            continue
        album_match = bool(row.get("album") and normalized(item.get("albumName", "")) == normalized(row["album"]))
        candidates.append((album_match, len(item["plainLyrics"]), item))
    if not candidates:
        return {"status": "no_exact_match", "source_url": url, "candidate_count": len(response.json())}
    item = max(candidates, key=lambda x: (x[0], x[1]))[2]
    return {"status": 200, "lyrics": clean_lyrics(item["plainLyrics"]),
            "source_url": f"https://lrclib.net/api/get/{item['id']}",
            "observed_title": item["trackName"], "observed_artist": item["artistName"],
            "observed_album": item.get("albumName"), "duration": item.get("duration"),
            "version_review": "Exact title/artist; album edition requires review when not in manifest."}


def genius(row: dict, client) -> dict:
    if client is None:
        return {"status": "not_configured"}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = client.search_song(row["title"], row["artist"])
    if result is None:
        return {"status": "not_found"}
    if normalized(result.title) != normalized(row["title"]) or not artist_matches(result.artist, row):
        return {"status": "identity_mismatch", "source_url": result.url,
                "observed_title": result.title, "observed_artist": result.artist}
    return {"status": 200, "lyrics": clean_lyrics(result.lyrics), "source_url": result.url,
            "observed_title": result.title, "observed_artist": result.artist,
            "version_review": "Exact artist/title from search; release metadata reviewed separately."}


def ovh(row: dict, session: requests.Session) -> dict:
    url = "https://api.lyrics.ovh/v1/" + quote(row["artist"], safe="") + "/" + quote(row["title"], safe="")
    response = session.get(url, timeout=20)
    if response.status_code != 200:
        return {"status": response.status_code, "source_url": url}
    text = response.json().get("lyrics", "")
    return {"status": 200 if text else "empty", "lyrics": clean_lyrics(text), "source_url": url,
            "version_review": "Provider does not return separate artist/version validation metadata."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--providers", default="lrclib,genius")
    p.add_argument("--limit", type=int)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--retry-failed-providers", default="", help="Comma-separated provider names whose cached failures may be retried.")
    p.add_argument("--delay", type=float, default=0.5)
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    rawdir = PRIVATE / "raw"; rawdir.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if isinstance(manifest, dict): manifest = manifest.get("songs", manifest.get("candidates", []))
    if args.limit: manifest = manifest[:args.limit]
    selected = args.providers.split(",")
    retry = set(args.retry_failed_providers.split(","))
    session = requests.Session(); session.headers["User-Agent"] = UA
    client = None
    if "genius" in selected:
        from lyricsgenius import Genius
        values = dotenv_values(ROOT / ".env")
        key = values.get("GENIUS_API_KEY") or values.get("GENIUS_API_TOKEN") or values.get("GENIUS_ACCESS_TOKEN")
        if key: client = Genius(key, timeout=15, retries=0, sleep_time=0.5, remove_section_headers=False, user_agent=UA)
    records, coverage, comparisons, public_manifest = [], [], [], []
    for index, original in enumerate(manifest):
        row = dict(original)
        row["song_id"] = song_id(row)
        row["language"] = {"English":"en", "Spanish":"es", "inglés":"en", "español":"es"}.get(row["language"], row["language"].lower())
        row["decade"] = int(row["year"]) // 10 * 10
        row["selection_label"] = row.get("possible_label", row.get("selection_label", "ambiguous"))
        responses = {}
        for provider in selected:
            suffix = f"-{row['lrclib_record_id']}" if provider == 'lrclib' and row.get('lrclib_record_id') else ''
            dest = rawdir / f"{row['song_id']}-{provider}{suffix}.json"
            started = time.perf_counter()
            cached_data = json.loads(dest.read_text()) if dest.is_file() else None
            use_cache = cached_data is not None and not args.refresh and not (provider in retry and cached_data.get("status") != 200)
            if use_cache:
                data = cached_data; cached = True
            else:
                try:
                    if provider == "lrclib": data = lrclib(row, session)
                    elif provider == "genius": data = genius(row, client)
                    elif provider == "ovh": data = ovh(row, session)
                    else: raise ValueError("Unknown provider")
                except Exception as error:
                    data = {"status": "error", "error_type": type(error).__name__,
                            "http_status": getattr(getattr(error, "response", None), "status_code", None)}
                data["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                data["latency_s"] = time.perf_counter() - started
                dest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                cached = False
                time.sleep(args.delay)
            # Cached responses remain immutable provenance records. Apply the
            # current cleaning policy in memory to cached and fresh lyrics.
            if data.get("lyrics"):
                data = {**data, "lyrics": clean_lyrics(data["lyrics"])}
            responses[provider] = data
            text = data.get("lyrics", "")
            coverage.append({"song_id":row["song_id"],"title":row["title"],"artist":row["artist"],
                             "year":row["year"],"language":row["language"],"provider":provider,
                             "status":data["status"],"lyrics_characters":len(text),"lyrics_hash":lyric_hash(text) if text else None,
                             "source_url":data.get("source_url"),"retrieved_at":data.get("retrieved_at"),
                             "latency_s":data.get("latency_s"),"cache_reused":cached,"version_review":data.get("version_review")})
        good = [(provider,data) for provider,data in responses.items() if data.get("status")==200 and len(data.get("lyrics", ""))>80]
        if good:
            preferred = row.get("preferred_provider")
            preferred_good = [pair for pair in good if pair[0] == preferred]
            provider,data = preferred_good[0] if preferred_good else good[0]
            record = {**row,"lyrics":data["lyrics"],"provider":provider,"lyrics_source_url":data.get("source_url"),
                      "lyrics_hash":lyric_hash(data["lyrics"]),"retrieved_at":data["retrieved_at"],
                      "version_review":data.get("version_review"),
                      "provider_selection_rule": "verified_manifest_override" if preferred_good else "first_available_provider_order"}
            records.append(record)
        for i,(pa,a) in enumerate(good):
            for pb,b in good[i+1:]:
                aa,bb = normalized(a["lyrics"]),normalized(b["lyrics"])
                comparisons.append({"song_id":row["song_id"],"provider_a":pa,"provider_b":pb,
                                    "normalized_exact_match":aa==bb,"sequence_similarity":SequenceMatcher(None,aa,bb,autojunk=False).ratio(),
                                    "word_count_a":len(aa.split()),"word_count_b":len(bb.split()),
                                    "note":"Agreement is textual, not proof of independent sourcing or correctness."})
        public_manifest.append({k:v for k,v in row.items() if k!="lyrics"})
        print(json.dumps({"progress":index+1,"total":len(manifest),"song_id":row["song_id"],"available_providers":[x[0] for x in good]},ensure_ascii=False),flush=True)
    (PRIVATE/"corpus.jsonl").write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in records)+"\n")
    (PUBLIC/"corpus-manifest.json").write_text(json.dumps(public_manifest,ensure_ascii=False,indent=2))
    pd.DataFrame(coverage).to_csv(PUBLIC/"provider-coverage.csv",index=False)
    pd.DataFrame(comparisons).to_csv(PUBLIC/"provider-agreement.csv",index=False)
    print(json.dumps({"collected_songs":len(records),"candidates":len(manifest),"providers":selected}))


if __name__ == "__main__":
    main()
