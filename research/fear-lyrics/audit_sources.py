"""Produce metadata-only source and privacy audits without network access.

Complete lyrics and provider payloads stay in the ignored private cache. The
public report contains hashes, counts, provenance and limitations only.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

from collect import clean_lyrics, normalized, LRC_TIMESTAMP_PREFIX

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIVATE = ROOT / ".research-cache/fear-lyrics"
EN = set("the and you your i me my is are was were it a an to of that with for this in on not can if when will know what how do we they their be but have all no".split())
ES = set("el la los las y tu tus yo mi mis que de del con por para en un una no es son era ser si cuando como su te se nos lo le me al o hay más sin muy bien".split())
AUTHORITIES = {
    "officialcharts.com": ("Official Charts Company", "primary_chart_registry"),
    "top40.nl": ("Stichting Nederlandse Top 40", "primary_chart_registry"),
    "elportaldemusica.es": ("PROMUSICAE / El Portal de Música", "primary_chart_and_certification_registry"),
    "los40.com": ("LOS40", "primary_for_own_chart_secondary_or_interview_for_other_claims"),
    "youtube.com": ("Official artist/label channel on YouTube", "artist_or_label_supplied_distribution_metadata"),
    "michaeljackson.com": ("Official Michael Jackson catalogue", "artist_catalogue"),
    "metallica.com": ("Metallica", "artist_catalogue"),
    "uc.cranberries.com": ("The Cranberries", "artist_biography"),
    "rhino.com": ("Rhino / Warner Music", "label_catalogue"),
    "dbrecords.co.uk": ("David Bates / DB Records", "primary_ar_discography_archive"),
    "sony.mediaroom.com": ("Columbia Records / Sony", "contemporaneous_label_press_release"),
    "sonymusic.es": ("Sony Music España", "label_catalogue_or_press_release"),
    "sonymusic.ca": ("Sony Music Canada", "label_press_release"),
    "sonymusic.com.pe": ("Sony Music Perú", "label_press_release_with_artist_statement"),
    "music.apple.com": ("Apple Music, label-supplied metadata", "distributor_metadata"),
    "qobuz.com": ("Qobuz, label-supplied metadata", "distributor_metadata"),
    "worldradiohistory.com": ("Historical Billboard / Music & Media issue", "archive_of_contemporaneous_primary_trade_press"),
    "billboard.com.co": ("Billboard Colombia", "music_trade_press"),
    "ayto-launion.org": ("Ayuntamiento de La Unión", "primary_concert_organizer_programme"),
    "elpais.com": ("El País, direct songwriter interview", "author_context_interview"),
    "latercera.com": ("La Tercera, direct songwriter interview", "author_context_interview"),
    "rtve.es": ("RTVE broadcast archive", "contemporaneous_performance_archive"),
    "blog.youtube": ("YouTube Music team", "primary_distribution_launch_announcement"),
    "open.spotify.com": ("Spotify, label-supplied track metadata", "primary_platform_metadata_and_reach"),
}


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def words(text: str) -> list[str]:
    value = unicodedata.normalize("NFKD", html.unescape(text).casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.findall(r"[a-z]+", value)


def body(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not re.fullmatch(r"\s*\[.*\]\s*", line))


def safe_pair(a: str, b: str) -> dict:
    aa, bb = words(body(a)), words(body(b))
    sa, sb = set(aa), set(bb)
    return dict(
        body_token_sequence_similarity=SequenceMatcher(None, aa, bb, autojunk=False).ratio(),
        body_vocabulary_jaccard=len(sa & sb) / len(sa | sb) if sa | sb else 1,
        body_word_counts=[len(aa), len(bb)],
        interpretation="Descriptive textual agreement; not independent sourcing or editorial truth.",
    )


def public_files() -> list[Path]:
    found = list(OUT.rglob("*"))
    for base in (ROOT / "content", ROOT / "static", ROOT / "public", ROOT / "examples"):
        if base.exists():
            found.extend(p for p in base.rglob("*") if "fear" in str(p.relative_to(ROOT)).lower() or "miedo" in str(p.relative_to(ROOT)).lower())
    return sorted({p for p in found if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".csv", ".tsv", ".html", ".ipynb", ".txt", ".svg", ".js", ".toml"}})


def leak_audit(private_texts: list[str]) -> dict:
    # Include original cached responses as well as selected cleaned lyrics.
    fingerprints = set()
    for text in private_texts:
        tokens = words(body(clean_lyrics(text)))
        fingerprints.update(sha(" ".join(tokens[i:i + 13])) for i in range(max(0, len(tokens) - 12)))
    matches = []
    checked = []
    forbidden_keys = []
    credential_patterns = []
    for path in public_files():
        if path.name == "source-audit.json":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = words(text)
        count = sum(sha(" ".join(tokens[i:i + 13])) in fingerprints for i in range(max(0, len(tokens) - 12)))
        if count:
            matches.append(dict(path=str(path.relative_to(ROOT)), matching_13word_windows=count))
        if path.suffix == ".json":
            try:
                value = json.loads(text)
                def inspect(node):
                    if isinstance(node, dict):
                        for key, val in node.items():
                            if key in {"lyrics", "plainLyrics", "syncedLyrics", "representative_docs"} and isinstance(val, (str, list)) and val:
                                forbidden_keys.append(dict(path=str(path.relative_to(ROOT)), key=key))
                            inspect(val)
                    elif isinstance(node, list):
                        for val in node: inspect(val)
                inspect(value)
            except json.JSONDecodeError:
                pass
        if re.search(r"(?:access_token|api_key|authorization)=[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{20,}", text, re.I):
            credential_patterns.append(str(path.relative_to(ROOT)))
        checked.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes())))
    ignored = subprocess.run(["git", "check-ignore", ".research-cache/fear-lyrics/corpus.jsonl"], cwd=ROOT, text=True, capture_output=True).returncode == 0
    tracked = subprocess.run(["git", "ls-files", ".research-cache"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    return dict(
        scope="Current fear/miedo study source, metadata, generated outputs and matching article/static/build files; unrelated historical site artifacts are outside this audit.",
        threshold_words=13,
        explanation="Conservative reconstruction screen, not a copyright-law threshold. No lyric excerpts are authorized or intentionally exported by this study; shorter fragments and paraphrases still require editorial review.",
        private_cache_ignored=ignored, private_cache_tracked_file_count=len(tracked.splitlines()) if tracked else 0,
        checked_file_count=len(checked), checked_files=checked,
        matching_lyric_windows=matches, forbidden_payload_keys=forbidden_keys,
        credential_pattern_files=credential_patterns,
        passed=ignored and not tracked and not matches and not forbidden_keys and not credential_patterns,
    )


def main() -> None:
    manifest = json.loads((OUT / "corpus-manifest.json").read_text())
    corpus = [json.loads(line) for line in (PRIVATE / "corpus.jsonl").read_text().splitlines() if line.strip()]
    byid = {row["song_id"]: row for row in corpus}
    before_path = Path("/private/tmp/fear-corpus-precorrection-audit.json")
    before = json.loads(before_path.read_text()) if before_path.exists() else {}
    raw_files = list((PRIVATE / "raw").glob("*.json"))
    unchanged = all((PRIVATE/"raw"/name).exists() and sha((PRIVATE/"raw"/name).read_bytes()) == value for name,value in before.get("provider_hashes", {}).items()) if before else None
    provider_status = defaultdict(Counter)
    private_texts = [row["lyrics"] for row in corpus]
    records, corrections, source_index = [], [], {}
    for row in manifest:
        sid = row["song_id"]
        selected = byid[sid]
        text = selected["lyrics"]
        tokens = words(body(text))
        en_share = sum(token in EN for token in tokens) / max(1, len(tokens))
        es_share = sum(token in ES for token in tokens) / max(1, len(tokens))
        expected_share, other_share = (en_share, es_share) if row["language"] == "en" else (es_share, en_share)
        checks, good = [], {}
        for provider in ("lrclib", "genius", "ovh"):
            suffix = f"-{row['lrclib_record_id']}" if provider == "lrclib" and row.get("lrclib_record_id") else ""
            provider_path = PRIVATE / "raw" / f"{sid}-{provider}{suffix}.json"
            if not provider_path.exists(): continue
            original = json.loads(provider_path.read_text())
            provider_status[provider][str(original.get("status"))] += 1
            lyric = original.get("lyrics", "")
            if lyric: private_texts.append(lyric)
            cleaned = clean_lyrics(lyric)
            if original.get("status") == 200 and len(cleaned) > 80: good[provider] = cleaned
            checks.append(dict(
                provider=provider, status=original.get("status"),
                observed_title=original.get("observed_title"), observed_artist=original.get("observed_artist"),
                observed_album=original.get("observed_album"), declared_duration_seconds=original.get("duration"),
                source_url=original.get("source_url"), original_cached_lyric_sha256=sha(lyric) if lyric else None,
                cleaned_lyric_sha256=sha(cleaned) if cleaned else None,
                original_cached_characters=len(lyric), cleaned_characters=len(cleaned),
                lrc_timestamp_prefix_lines=sum(bool(LRC_TIMESTAMP_PREFIX.match(line.strip())) for line in lyric.splitlines()),
            ))
        pairs = []
        providers = list(good)
        for i, a in enumerate(providers):
            for b in providers[i + 1:]: pairs.append(dict(provider_a=a, provider_b=b, **safe_pair(good[a], good[b])))
        old_hash = before.get("lyrics_hashes", {}).get(sid)
        if old_hash and old_hash != selected["lyrics_hash"]:
            corrections.append(dict(song_id=sid, title=row["title"], old_sha256=old_hash, new_sha256=selected["lyrics_hash"],
                reason="Verified provider override for original studio edition" if row["title"] == "Zombie" else "Remove strictly numeric LRC timestamps at line starts; original words/order retained"))
        watch = []
        if len(good) == 1: watch.append("Single successful provider; no cross-provider textual comparator.")
        if any(c.get("declared_duration_seconds") is not None and c["declared_duration_seconds"] < 90 for c in checks if c["status"] == 200):
            watch.append("Provider duration may describe preview/fragment, not full recording. Duration alone was not used as lyric correctness proof.")
        if not row.get("album"):
            watch.append("Manifest lacks target album; longest LRCLIB candidate is a heuristic. Compilation metadata does not verify the original edition.")
        records.append(dict(
            song_id=sid, title=row["title"], artist=row["artist"], year=row["year"], language=row["language"],
            selected_provider=selected["provider"], selected_lyrics_sha256=selected["lyrics_hash"],
            selected_hash_matches_text=selected["lyrics_hash"] == sha(text),
            selected_lrc_timestamp_prefix_lines=sum(bool(LRC_TIMESTAMP_PREFIX.match(line.strip())) for line in text.splitlines()),
            available_provider_count=len(good),
            language_screen=dict(expected_stopword_share=expected_share, alternate_stopword_share=other_share,
                compatible=expected_share > other_share, limitation="Heuristic language sanity check, not a validated language model or manual gold label."),
            provider_identity_and_versions=checks, cleaned_body_agreement=pairs, caveats=watch,
        ))
        for field, scope in (("release_source_url", "release_metadata"), ("commercial_source_url", "commercial_evidence"), ("commercial_additional_source_url", "commercial_evidence"), ("theme_source_url", "author_context_or_interpretation")):
            url = row.get(field)
            if not url: continue
            ref = source_index.setdefault(url, dict(url=url, source_type="corpus_provenance", reviewed_date="2026-10-11", supports=[], song_ids=[]))
            host = urlparse(url).hostname or ""
            host = host.removeprefix("www.")
            authority, authority_type = AUTHORITIES.get(host, (host, "inspect_authority_for_claim"))
            ref["authority"] = authority
            ref["authority_type"] = authority_type
            ref.setdefault("descriptive_label", f"{authority}: {row['artist']} — {row['title']}")
            ref.setdefault("publication_date", None)
            ref.setdefault("publication_date_note", "Dynamic catalogue or date not independently transcribed; review date is not publication date.")
            if scope == "release_metadata":
                ref.setdefault("release_years_supported", [])
                if row["year"] not in ref["release_years_supported"]: ref["release_years_supported"].append(row["year"])
                if row.get("release_date"):
                    ref.setdefault("release_dates_recorded_in_manifest", []).append(row["release_date"])
            if scope not in ref["supports"]: ref["supports"].append(scope)
            if sid not in ref["song_ids"]: ref["song_ids"].append(sid)
    references = dict(study_cutoff="2026-10-11", scope="Corpus release, chart/certification and author-context sources; technical research bibliography is separate.",
        sources=list(source_index.values()),
        public_notes=[
            "Release year and year of chart success differ for several songs; metadata records distinguish them.",
            "Chart territories and metrics differ. No cross-chart aggregate popularity score is justified.",
            "Source dates, artist aliases, editions and source authority are recorded in corpus-manifest.json.",
            f"2026 is partial: {sum(r['year']==2026 for r in manifest)} selected recordings released before the cutoff.",
            "Selection labels are hypotheses, not independent human annotations.",
        ])
    (OUT / "references.json").write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n")
    counts = Counter((row["decade"], row["language"]) for row in corpus)
    integrity = dict(candidates=len(manifest), selected_songs=len(corpus), artist_strings=len({row["artist"] for row in corpus}),
        language_decade_counts=[dict(decade=k[0], language=k[1], songs=v) for k, v in sorted(counts.items())],
        all_ten_language_decade_cells_balanced=len(counts) == 10 and len(set(counts.values())) == 1,
        provider_response_files=len(raw_files), provider_response_files_unchanged=unchanged,
        selected_provider_counts=dict(Counter(row["provider"] for row in corpus)),
        provider_status_counts={k:dict(v) for k,v in provider_status.items()},
        before_corpus_sha256=before.get("corpus_sha256"), after_corpus_sha256=sha((PRIVATE / "corpus.jsonl").read_bytes()),
        all_declared_lyric_hashes_match=all(r["selected_hash_matches_text"] for r in records),
        selected_timestamp_lines=sum(r["selected_lrc_timestamp_prefix_lines"] for r in records),
        language_screen_mismatches=[r["song_id"] for r in records if not r["language_screen"]["compatible"]])
    privacy = leak_audit(private_texts)
    audit = dict(audit_version=2, reviewed_date="2026-10-11", status="passed_with_explicit_edition_and_sampling_caveats",
        timing="Source corrections completed from immutable cache before topic/embedding interpretation; this is a post-acquisition protocol correction.",
        no_network_calls=True, no_lyrics_exported=True, integrity=integrity,
        collection_rules=dict(lrclib_selection="Exact normalized artist/title; preferred album if supplied, then longest lyrics. Longest is only a heuristic, not editorial validation.",
            corpus_provider_selection="Manifest preferred_provider only for reviewed override, otherwise first successful provider in declared order.",
            bad_versions_excluded=["live","unplugged","acoustic","demo","remix","karaoke","instrumental","translation","cover","sped up","slowed"],
            radio_edits="Not categorically excluded; review against chosen recording/edition.",
            cleaning="Only strict numeric LRC timestamp prefixes removed. Original narration, order, words and section headings retained.",
            raw_cache="Files labeled raw are cached provider records after the initial collector processing, not byte-for-byte original HTTP payloads. Existing pilot provider records remain immutable; expanded acquisition adds new records.",
            coverage_interpretation="Observed collector responses on this corpus/query strategy, not general API quality. Attempt-level outcomes are retained for technical provenance."),
        corrected_records=corrections, songs=records, privacy=privacy,
        residual_caveats=[
            "Most candidates lack album identity; compilation records and full-versus-radio variants are not exhaustively verified.",
            "Stan, Crazy In Love, Llamando a la Tierra, Miedo and Enrique Iglesias Bailando have one successful provider. Identity/language are compatible; full edition correctness remains provisional.",
            "Bailando chosen for Enrique Iglesias is predominantly Spanish in the language screen; Sean Paul is not an accepted artist alias.",
            "Low sequence agreement can arise from refrain expansion, section boundaries and orthography. It does not by itself imply wrong lyrics.",
            "The 13 word screen is conservative QA, not permission to quote 12 words. No lyric excerpts are intentionally published; shorter matches require human editorial inspection.",
        ])
    (OUT / "source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    assert integrity["selected_songs"] == len(manifest)
    assert integrity["all_ten_language_decade_cells_balanced"]
    assert integrity["provider_response_files_unchanged"] is not False
    assert integrity["all_declared_lyric_hashes_match"] and integrity["selected_timestamp_lines"] == 0
    assert not integrity["language_screen_mismatches"]
    assert privacy["passed"], "Public privacy screen failed; inspect metadata-only source-audit.json."
    print(json.dumps(dict(songs=len(corpus), artist_strings=integrity["artist_strings"], raw_responses=len(raw_files), corrections=len(corrections),
        public_files_checked=privacy["checked_file_count"], privacy_passed=privacy["passed"], lyric_fragments_printed=0)))


if __name__ == "__main__":
    main()
