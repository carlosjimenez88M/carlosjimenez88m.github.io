#!/usr/bin/env python3
"""Reproducible, exploratory bilingual lyric analysis without publishing lyrics.

Input: a PRIVATE JSONL corpus. Output: IDs, aggregates, short topic vocabularies,
and explicitly synthetic diagnostic sentences. Full lyrics and fitted BERTopic
objects are never written to the public results directory. Embedding matrices
are cached privately. Nothing in this module fetches lyrics or calls paid APIs.

Example (downloads are disabled unless explicitly requested):
    python analyze.py --input .research-cache/fear-lyrics/corpus.jsonl \
        --output research/fear-lyrics/results --offline

Required for the full analysis: numpy, scipy, scikit-learn, sentence-transformers,
bertopic, umap-learn, hdbscan, pandas. --dry-run needs only the standard library.
The corpus is a purposive case series, not a representative chart sample. Song
selection labels are hypotheses, never human-validated emotion/topic gold labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
import unicodedata


DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]
# An explicit, fixed bilingual list, not an inferred stop list fitted by decade.
STOP_WORDS = sorted(set("""
a an and are as at be been being but by can could did do does doing for from had
has have he her here hers herself him himself his how i if in into is it its
itself just me might more most my myself no nor not of on once only or other our
ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves oh yeah na la
al algo algunas algunos ante antes aquel aquella aquellas aquellos aquí así
aun aunque bajo bien cada como con contra cual cuando de del desde donde dos
el ella ellas ello ellos en entre era erais eran eras eres es esa esas ese esos
esta estaba estaban estamos están estar estas este esto estos fue fueron ha
haber había han hasta hay la las le les lo los más me mi mis mucha muchas mucho
muchos muy nada ni nos nosotros nuestra nuestras nuestro nuestros o os otra
otras otro otros para pero poco por porque qué que quien quienes se sea ser si
sí sin sobre somos son su sus también te tener ti tiene todo todos tras tu tus
un una unas uno unos usted ustedes ya yo e y
""".split()))
NEGATION_WORDS = {"no", "not", "nor", "never", "neither", "without", "ni", "sin", "nunca", "jamás", "tampoco", "nada"}
STOP_WORDS = sorted(set(STOP_WORDS) - NEGATION_WORDS)

HEADER = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]{1,100})\]|"
    r"(?P<plain>(?:verse|chorus|pre[- ]?chorus|post[- ]?chorus|bridge|intro|outro|"
    r"refrain|hook|interlude|instrumental|estrofa|verso|estribillo|coro|puente|"
    r"introducción|salida)(?:\s+\d+)?))\s*:?\s*$", re.I)
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Original, bilingual probes: they contain no quotations from songs.
PROTOTYPES = {
    "fear": {
        "en": ["I feel afraid of what might happen next.",
               "Uncertainty makes me anxious and vulnerable.",
               "I dread losing someone and do not feel safe."],
        "es": ["Siento miedo de lo que podría pasar después.",
               "La incertidumbre me hace sentir ansiedad y vulnerabilidad.",
               "Temo perder a alguien y no me siento a salvo."],
    },
    "threat": {
        "en": ["Someone is pursuing me and threatens to hurt me.",
               "Danger approaches through violence and destruction.",
               "An outside force puts our safety at risk."],
        "es": ["Alguien me persigue y amenaza con hacerme daño.",
               "El peligro se acerca mediante violencia y destrucción.",
               "Una fuerza externa pone nuestra seguridad en riesgo."],
    },
    "avoidance": {
        "en": ["I hide and run away to avoid the danger.",
               "I remain silent and keep my distance from what scares me.",
               "I escape instead of facing what I fear."],
        "es": ["Me escondo y huyo para evitar el peligro.",
               "Guardo silencio y me alejo de lo que me asusta.",
               "Escapo en lugar de enfrentar lo que temo."],
    },
    "coping": {
        "en": ["I face the danger and find support to carry on.",
               "I act despite my fear and regain a sense of agency.",
               "Together we confront uncertainty and learn to cope."],
        "es": ["Enfrento el peligro y encuentro apoyo para seguir adelante.",
               "Actúo a pesar del miedo y recupero mi capacidad de actuar.",
               "Juntos afrontamos la incertidumbre y aprendemos a sobrellevarla."],
    },
}

PROBES = [
    dict(id="negation_en", category="negation", query="I am afraid of you.",
         related="I am not afraid of you.", contrast="I am preparing dinner.",
         interpretation="Semantic relatedness is not polarity recognition."),
    dict(id="negation_es", category="negation", query="Tengo miedo de ti.",
         related="No tengo miedo de ti.", contrast="Estoy preparando la cena.",
         interpretation="La similitud temática no distingue por sí sola la negación."),
    dict(id="role_en", category="role_reversal", query="The singer fears the stranger.",
         related="The stranger fears the singer.", contrast="The singer fears the storm.",
         interpretation="Related vocabulary can conceal a reversal of experiencer and object."),
    dict(id="role_es", category="role_reversal", query="La narradora teme al visitante.",
         related="El visitante teme a la narradora.", contrast="La narradora teme a la tormenta.",
         interpretation="No hay un umbral validado de fidelidad a los roles narrativos."),
    dict(id="translation_en_es", category="crosslingual", query="I fear being abandoned by everyone.",
         related="Temo que todos me abandonen.", contrast="Me alegra cocinar con mis amigos.",
         interpretation="A handcrafted translation should outrank an unrelated sentence."),
    dict(id="translation_es_en", category="crosslingual", query="Temo la violencia que se acerca.",
         related="I fear the violence that is approaching.", contrast="I enjoy listening to quiet music.",
         interpretation="This is a tiny positive control, not bilingual corpus validation."),
    dict(id="context_en", category="fear_context", query="I fear that my partner will leave me.",
         related="I dread being alone after a breakup.", contrast="I fear being attacked in the street.",
         interpretation="Both candidates concern fear; the preferred candidate preserves its context."),
    dict(id="context_es", category="fear_context", query="Temo que mi pareja me abandone.",
         related="Me asusta quedarme solo después de la ruptura.", contrast="Temo un ataque en la calle.",
         interpretation="La recuperación debe preservar el objeto del miedo, además del tema."),
    dict(id="repetition_en", category="repetition", query="I hide because I fear the approaching danger.",
         related=" ".join(["I hide because I fear the approaching danger."] * 8),
         contrast="I face the danger with support from my friends.",
         interpretation="Embedding similarity to repetition does not justify counting eight independent observations."),
    dict(id="repetition_es", category="repetition", query="Me escondo porque temo el peligro que se acerca.",
         related=" ".join(["Me escondo porque temo el peligro que se acerca."] * 8),
         contrast="Enfrento el peligro con el apoyo de mis amigos.",
         interpretation="La repetición se analiza como sensibilidad del peso, no como ocho canciones."),
    dict(id="paraphrase_en", category="paraphrase", query="I dread the possibility that my loved ones might leave.",
         related="The thought of losing everyone close to me fills me with fear.",
         contrast="The sky is dark, but I feel peaceful and safe.",
         interpretation="Positive control: a paraphrase should preserve fear of abandonment despite different words."),
    dict(id="paraphrase_es", category="paraphrase", query="Me angustia la posibilidad de que mis seres queridos se marchen.",
         related="Pensar que podría perder a quienes amo me llena de miedo.",
         contrast="El cielo está oscuro, pero me siento tranquilo y a salvo.",
         interpretation="Una paráfrasis positiva se compara con una escena oscura sin miedo explícito."),
    dict(id="polarity_context_en", category="polarity_context", query="A narrator reports currently feeling afraid of the approaching storm.",
         related="I am afraid of the storm that is coming.",
         contrast="I am no longer afraid of the storm that is coming.",
         interpretation="Query-conditioned polarity preference, not a requirement that negated statements have low cosine similarity."),
    dict(id="polarity_context_es", category="polarity_context", query="La voz narrativa declara que ahora siente miedo de la tormenta que se acerca.",
         related="Tengo miedo de la tormenta que se acerca.",
         contrast="Ya no tengo miedo de la tormenta que se acerca.",
         interpretation="Se evalúa el margen entre candidatos respecto a una consulta explícita de polaridad."),
    dict(id="motive_en", category="motive_context", query="A person hides to escape something they fear.",
         related="I hide behind the door because I am afraid of being found.",
         contrast="I hide behind the door because I want to surprise my friend at a party.",
         interpretation="The same physical action has different intentions; hiding alone is not evidence of fear."),
    dict(id="motive_es", category="motive_context", query="Una persona se esconde para escapar de algo que teme.",
         related="Me escondo detrás de la puerta porque temo que me encuentren.",
         contrast="Me escondo detrás de la puerta para sorprender a mi amiga en una fiesta.",
         interpretation="Una acción de esconderse puede expresar miedo o una intención lúdica."),
    dict(id="experiencer_en", category="experiencer", query="The narrator feels personally frightened while walking through a dark place.",
         related="I walk down the dark corridor and feel afraid.",
         contrast="I walk down the dark corridor and calmly observe someone else feeling afraid.",
         interpretation="Distinguish who experiences fear from the setting and observation of another person."),
    dict(id="experiencer_es", category="experiencer", query="La narradora siente miedo personalmente al caminar por un lugar oscuro.",
         related="Camino por el pasillo oscuro y siento miedo.",
         contrast="Camino por el pasillo oscuro con calma y observo a otra persona que siente miedo.",
         interpretation="La oscuridad del contexto y el miedo ajeno no equivalen al miedo de la narradora."),
    dict(id="order_en", category="narrative_order", query="The narrative begins with fear and ends with courage.",
         related="At first I feel afraid. Later I find courage.",
         contrast="At first I find courage. Later I feel afraid.",
         interpretation="The candidates share the same lexical multiset but reverse chronology; cosine is not a plot parser."),
    dict(id="long_ending_es", category="long_ending", query="El desenlace muestra que la narradora afronta el peligro con valentía.",
         related=" ".join(["Al principio siento miedo y me escondo ante el peligro."] * 120) +
                 " Al final enfrento el peligro con valentía y decido actuar.",
         contrast=" ".join(["Al principio siento miedo y me escondo ante el peligro."] * 120) +
                  " Al final sigo escondida, paralizada por el miedo, y no actúo.",
         interpretation="Controlled long inputs have an identical prefix and incompatible endings. Check truncation before interpreting a tie."),
]


@dataclass
class Song:
    song_id: str
    title: str
    artist: str
    year: int
    decade: int
    language: str
    selection_label: str
    lyrics: str
    provider: str
    source_url: str
    lyrics_hash: str
    release_date: str | None = None


@dataclass
class Section:
    document_id: str
    song_id: str
    order: int
    text: str
    label: str
    segmentation: str
    canonical_id: str
    is_repeat: bool
    words: int
    lines: int


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", model)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False prevents quietly publishing invalid numerical values.
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".analysis-", delete=False) as handle:
        handle.write(encoded + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=".analysis-", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def safe_failure_trace(exc: Exception) -> list[dict]:
    # Frame location is useful for repair, whereas exception messages/locals
    # can accidentally contain copyrighted input texts or provider credentials.
    return [dict(module=Path(frame.filename).name, function=frame.name, line=frame.lineno)
            for frame in traceback.extract_tb(exc.__traceback__)[-6:]]


def read_corpus(path: Path, args) -> tuple[list[Song], list[dict]]:
    songs, notices, seen = [], [], set()
    languages = {"en": "en", "eng": "en", "english": "en", "inglés": "en",
                 "es": "es", "spa": "es", "spanish": "es", "español": "es"}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                song_id = str(record["song_id"])
                year = int(record["year"])
                language = languages[str(record["language"]).lower()]
                lyrics = str(record.get("lyrics") or "")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid corpus record on line {line_no}: {type(exc).__name__}") from None
            if song_id in seen:
                raise ValueError(f"Duplicate song_id on corpus line {line_no}")
            seen.add(song_id)
            if not args.min_year <= year <= args.max_year:
                notices.append(dict(song_id=song_id, reason="outside_year_window"))
                continue
            release = record.get("release_date")
            if release:
                try:
                    parsed_release = date.fromisoformat(str(release))
                except ValueError:
                    raise ValueError(f"Invalid release_date on corpus line {line_no}") from None
                if parsed_release > date.fromisoformat(args.cutoff):
                    notices.append(dict(song_id=song_id, reason="released_after_cutoff"))
                    continue
            elif year == args.max_year:
                notices.append(dict(song_id=song_id, reason="release_date_unverified_in_partial_year"))
            if not lyrics.strip():
                notices.append(dict(song_id=song_id, reason="missing_lyrics"))
                continue
            actual_hash = sha(lyrics)
            declared_hash = record.get("lyrics_hash")
            if declared_hash and declared_hash != actual_hash:
                notices.append(dict(song_id=song_id, reason="declared_hash_differs_from_utf8_lyrics_hash"))
            supplied_decade = str(record.get("decade", ""))
            decade_match = re.search(r"\d{4}", supplied_decade)
            if decade_match and int(decade_match.group()) != year // 10 * 10:
                notices.append(dict(song_id=song_id, reason="decade_corrected_from_release_year"))
            songs.append(Song(song_id, str(record.get("title", "")), str(record.get("artist", "")),
                              year, year // 10 * 10, language,
                              str(record.get("selection_label", "ambiguous")), lyrics,
                              str(record.get("provider", "unknown")),
                              str(record.get("lyrics_source_url", "")), actual_hash,
                              str(release) if release else None))
    if not songs:
        raise ValueError("No eligible songs with lyrics remain in the corpus.")
    return songs, notices


def canonical_text(text: str) -> str:
    # Exact normalized stanza repeats only: near-repeats remain distinct.
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE))


def segment(song: Song, max_lines: int, min_words: int) -> list[Section]:
    paragraphs, buffer, label = [], [], "unlabeled"

    def flush() -> None:
        if buffer:
            paragraphs.append((list(buffer), label))
            buffer.clear()

    for raw in unicodedata.normalize("NFKC", song.lyrics).replace("\r", "").split("\n"):
        line = raw.strip()
        match = HEADER.match(line)
        if match:
            flush()
            label = match.group("bracket") or match.group("plain") or "unlabeled"
        elif not line:
            flush()
        else:
            buffer.append(line)
    flush()
    sections, first = [], {}
    for lines, raw_label in paragraphs:
        # A provider without blank stanza boundaries must not silently force a
        # whole song into one truncated embedding. These are labeled windows.
        step = max_lines if max_lines else len(lines)
        for offset in range(0, len(lines), step):
            chunk = lines[offset:offset + step]
            text = "\n".join(chunk)
            count = len(TOKEN.findall(text))
            if count < min_words:
                continue
            index = len(sections)
            document_id = f"{song.song_id}::s{index:03d}"
            signature = canonical_text(text)
            is_repeat = signature in first
            canonical_id = first.setdefault(signature, document_id)
            # Provider-supplied headings can themselves contain quoted lyrics:
            # keep them PRIVATE; public outputs use only generic segmentation.
            segmentation = "line_window" if len(lines) > step else "paragraph_or_header"
            sections.append(Section(document_id, song.song_id, index, text, raw_label,
                                    segmentation, canonical_id, is_repeat, count, len(chunk)))
    return sections


def public_section(section: Section) -> dict:
    return dict(document_id=section.document_id, song_id=section.song_id,
                order=section.order, canonical_id=section.canonical_id,
                exact_repeat=section.is_repeat, segmentation=section.segmentation,
                word_count=section.words, line_count=section.lines)


def probe_inputs() -> list[dict]:
    return [dict(input_id=f"{probe['id']}::{key}", text=probe[key], text_sha256=sha(probe[key]), role="passage")
            for probe in PROBES for key in ("query", "related", "contrast")]


def prototype_inputs() -> list[dict]:
    return [dict(input_id=f"{state}::{lang}::{i:02d}", text=text, text_sha256=sha(text), role="query")
            for state in PROTOTYPES for lang in ("en", "es")
            for i, text in enumerate(PROTOTYPES[state][lang])]


def export_embedding_input(path: Path, sections: list[Section], args) -> None:
    if not path.resolve().is_relative_to(args.private_cache.resolve()):
        raise ValueError("Embedding input exports contain lyrics and must be inside --private-cache.")
    documents = [dict(input_id=s.document_id, text=s.text, text_sha256=sha(s.text), role="passage")
                 for s in sections if not s.is_repeat]
    write_json(path, dict(privacy="PRIVATE: lyric blocks must never be copied to public results.",
                         documents=documents, probes=probe_inputs(), prototypes=prototype_inputs(),
                         provider_contract="NPZ arrays documents/probes/prototypes in these exact orders. "
                         "Sidecar JSON: model_id, document_ids, document_hashes, probe_hashes, prototype_hashes; "
                         "optional provider, revision, token_counts, max_seq_length, usage, prefixes."))


def corpus_summary(songs: list[Song], sections: list[Section], notices: list[dict], args) -> dict:
    grouped = defaultdict(list)
    for song in songs:
        grouped[(song.decade, song.language)].append(song)
    cells = [dict(decade=d, language=lang, songs=len(group),
                  artists=len({s.artist.casefold() for s in group}),
                  hypothesis_labels=dict(Counter(s.selection_label for s in group)))
             for (d, lang), group in sorted(grouped.items())]
    repeats = sum(s.is_repeat for s in sections)
    zero = sorted({s.song_id for s in songs} - {s.song_id for s in sections})
    return dict(
        songs_with_lyrics=len(songs), songs_with_eligible_blocks=len(songs) - len(zero),
        artists=len({s.artist.casefold() for s in songs}), blocks=len(sections),
        unique_blocks=len(sections) - repeats, exact_repeated_blocks=repeats,
        decade_language_cells=cells, notices=notices, songs_without_eligible_blocks=zero,
        providers=dict(Counter(s.provider for s in songs)),
        cutoff=args.cutoff, year_window=[args.min_year, args.max_year],
        partial_final_decade=args.max_year % 10 != 9,
        partial_final_year=(args.max_year == date.fromisoformat(args.cutoff).year and
                            date.fromisoformat(args.cutoff).strftime("%m-%d") != "12-31"),
        missing_language_cells=[dict(decade=d, language=lang)
                                for d in sorted({s.decade for s in songs})
                                for lang in ("en", "es") if (d, lang) not in grouped],
        corpus_identity=sha("\n".join(f"{s.song_id}\t{s.lyrics_hash}" for s in songs)),
        design="Purposive bilingual case series. Selection labels are unvalidated hypotheses.",
        limitation="Release years describe recording cohorts, not when songs were listened to or charted. "
                   "A missing release_date in the partial year is explicitly unverified.",
    )


def model_snapshot_receipt(model_id: str, cache: Path, revision: str | None = None) -> dict:
    """Identify cached public model files offline; hash weights, never secrets."""
    from huggingface_hub import try_to_load_from_cache
    config = try_to_load_from_cache(model_id, "config.json", cache_dir=str(cache),
                                    revision=revision or "main")
    if not isinstance(config, str):
        raise ValueError("Model config is not present in the private/local cache.")
    # A model can be fully usable while optional ONNX/evaluation files are
    # deliberately absent. Resolve one required file, not the whole repo tree.
    snapshot = Path(config).parent
    name = snapshot.name
    commit = name if re.fullmatch(r"[0-9a-f]{40}", name) else None
    weights = sorted(snapshot.glob("*.safetensors")) or sorted(snapshot.glob("pytorch_model*.bin"))
    files = []
    for path in weights:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files.append(dict(filename=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return dict(resolved_snapshot_commit=commit, cached_weight_files=files,
                requested_revision=revision, revision_pinned=bool(revision),
                receipt_method="Local Hugging Face snapshot identity and SHA256 of cached model weight files; no network.")


class Encoder:
    def __init__(self, model_id: str, args):
        from sentence_transformers import SentenceTransformer
        self.model_id, self.args = model_id, args
        self.revision = dict(v.split("=", 1) for v in args.model_revision).get(model_id)
        if args.offline or not args.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        kwargs = dict(device=args.device, cache_folder=str(args.model_cache),
                      local_files_only=args.offline or not args.allow_download)
        if self.revision:
            kwargs["revision"] = self.revision
        self.model = SentenceTransformer(model_id, **kwargs)
        self.commit = getattr(getattr(self.model[0], "auto_model", None), "config", None)
        self.commit = getattr(self.commit, "_commit_hash", None)
        self.receipt = model_snapshot_receipt(model_id, args.model_cache, self.revision)
        self.commit = self.commit or self.receipt["resolved_snapshot_commit"]
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)
        self.cache = args.private_cache / "embeddings" / slug(model_id)
        self.cache.mkdir(parents=True, exist_ok=True)

    def encode(self, texts: list[str], role: str = "passage"):
        import numpy as np
        prefixed = [f"{role}: {text}" for text in texts] if "multilingual-e5" in self.model_id else texts
        identity = sha(json.dumps(dict(model=self.model_id, revision=self.revision,
                                       resolved_commit=self.commit, role=role,
                                       max_seq_length=self.max_seq_length,
                                       texts=prefixed), ensure_ascii=False))
        cache = self.cache / f"{identity}.npy"
        self.last_cache_path = cache
        if cache.exists():
            matrix = np.load(cache, allow_pickle=False)
            if matrix.shape[0] == len(texts) and np.isfinite(matrix).all():
                return matrix
        matrix = self.model.encode(prefixed, batch_size=self.args.batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
        matrix = np.asarray(matrix, dtype=np.float32)
        with tempfile.NamedTemporaryFile("wb", dir=self.cache, prefix=".embedding-", delete=False) as handle:
            np.save(handle, matrix, allow_pickle=False)
            temporary = Path(handle.name)
        temporary.replace(cache)
        return matrix

    def token_diagnostics(self, sections: list[Section], kind: str = "documents") -> list[dict]:
        texts = [f"passage: {s.text}" if "multilingual-e5" in self.model_id else s.text for s in sections]
        tokens = self.model.tokenizer(texts, padding=False, truncation=False,
                                      add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [dict(document_id=s.document_id, tokenizer_tokens=len(ids),
                     model_max_seq_length=self.max_seq_length,
                     exceeds_max_seq_length=len(ids) > self.max_seq_length)
                for s, ids in zip(sections, tokens)]

    def manifest(self) -> dict:
        return dict(model_id=self.model_id, requested_revision=self.revision,
                    resolved_commit=self.commit, max_seq_length=self.max_seq_length,
                    dimensions=self.dimensions, weight_receipt=self.receipt,
                    normalization="L2, cosine similarity", device=self.args.device,
                    passage_prefix="passage: " if "multilingual-e5" in self.model_id else "",
                    prototype_prefix="query: " if "multilingual-e5" in self.model_id else "",
                    revision_pinned=bool(self.revision))


class FileEncoder:
    """Consume already-computed embeddings, without network or provider secrets.

    The sidecar hashes bind every vector to an exact input and prevent a silent
    swap between full and deduplicated corpus order. No pickle data is accepted.
    """
    def __init__(self, args, sections: list[Section]):
        import numpy as np
        path = args.embedding_file
        metadata_path = args.embedding_metadata or path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_ids = [s.document_id for s in sections if not s.is_repeat]
        expected_hashes = [sha(s.text) for s in sections if not s.is_repeat]
        checks = {
            "document_ids": expected_ids, "document_hashes": expected_hashes,
            "probe_hashes": [p["text_sha256"] for p in probe_inputs()],
            "prototype_hashes": [p["text_sha256"] for p in prototype_inputs()],
        }
        for key, expected in checks.items():
            if metadata.get(key) != expected:
                raise ValueError(f"External embedding metadata mismatch: {key}.")
        self.arrays, dimensions = {}, set()
        with np.load(path, allow_pickle=False) as bundle:
            for key, expected in (("documents", len(expected_ids)), ("probes", len(PROBES) * 3),
                                  ("prototypes", len(prototype_inputs()))):
                matrix = np.asarray(bundle[key], dtype=np.float32)
                if matrix.ndim != 2 or matrix.shape[0] != expected or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid external embedding matrix: {key}.")
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if np.any(norms < 1e-12):
                    raise ValueError(f"Zero external embedding vector: {key}.")
                self.arrays[key] = matrix / norms
                dimensions.add(matrix.shape[1])
        if len(dimensions) != 1:
            raise ValueError("External documents, probes, and prototypes must share dimensionality.")
        self.hashes = {"documents": expected_hashes,
                       "probes": checks["probe_hashes"], "prototypes": checks["prototype_hashes"]}
        self.model_id = args.model_label or str(metadata.get("model_id", "external_embeddings"))
        self.metadata = metadata
        self.max_seq_length = metadata.get("max_seq_length")

    def encode(self, texts: list[str], role: str = "passage"):
        hashes = [sha(text) for text in texts]
        for name in ("documents", "probes", "prototypes"):
            if hashes == self.hashes[name]:
                expected_role = "query" if name == "prototypes" else "passage"
                if role != expected_role:
                    raise ValueError("External embedding role mismatch.")
                return self.arrays[name]
        raise ValueError("External embedding inputs differ from the verified bundle.")

    def token_diagnostics(self, sections: list[Section], kind: str = "documents") -> list[dict]:
        counts = self.metadata.get("token_counts", {}).get(kind, [])
        bounds = self.metadata.get("input_token_upper_bounds", {}).get(kind, [])
        has_counts = len(counts) == len(sections) and all(v is not None for v in counts)
        has_bounds = len(bounds) == len(sections)
        return [dict(document_id=s.document_id,
                     tokenizer_tokens=counts[i] if has_counts else None,
                     conservative_token_upper_bound=bounds[i] if has_bounds else None,
                     model_max_seq_length=self.max_seq_length,
                     exceeds_max_seq_length=bool(counts[i] > self.max_seq_length)
                     if self.max_seq_length is not None and has_counts else
                     False if self.max_seq_length is not None and has_bounds and bounds[i] <= self.max_seq_length else None,
                     tokenization_status="provider_reported" if has_counts else "conservative_byte_bound" if has_bounds else "unknown")
                for i, s in enumerate(sections)]

    def manifest(self) -> dict:
        # Whitelist metadata: do not copy arbitrary provider payloads or secrets.
        allowed = {key: self.metadata.get(key) for key in ("provider", "model_id", "revision", "prefixes", "usage")}
        return dict(model_id=self.model_id, input_type="verified_external_embedding_bundle",
                    normalization="L2, cosine similarity", dimensions=self.arrays["documents"].shape[1],
                    max_seq_length=self.max_seq_length, provider_metadata=allowed,
                    revision_pinned=bool(self.metadata.get("revision")))


def fit_topics(sections: list[Section], embeddings, seed: int, args):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    count = len(sections)
    if count < max(6, args.min_cluster_size):
        raise ValueError("too_few_unique_blocks_for_declared_cluster_size")
    reducer = UMAP(n_neighbors=min(args.n_neighbors, count - 1),
                   n_components=min(5, count - 2), min_dist=0.0, metric="cosine",
                   random_state=seed, n_jobs=1, low_memory=True)
    clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size,
                        min_samples=min(args.min_samples, count - 1),
                        metric="euclidean", cluster_selection_method="eom",
                        prediction_data=True, core_dist_n_jobs=1)
    vectorizer = CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2),
                                 token_pattern=r"(?u)\b[^\W\d_]{2,}(?:['’][^\W\d_]+)*\b", min_df=1)
    topic_model = BERTopic(language="multilingual", embedding_model=None, umap_model=reducer,
                           hdbscan_model=clusterer, vectorizer_model=vectorizer,
                           ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                           top_n_words=args.top_words, calculate_probabilities=False,
                           low_memory=True, verbose=False)
    labels, _ = topic_model.fit_transform([s.text for s in sections], embeddings)
    parameters = dict(seed=seed, documents=count, umap_neighbors=min(args.n_neighbors, count - 1),
                      umap_components=min(5, count - 2), umap_min_dist=0.0,
                      umap_metric="cosine", hdbscan_metric="euclidean_in_umap_space",
                      min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                      cluster_selection_method="eom", ctfidf_reduce_frequent_words=True,
                      lexical_preprocessing_language="multilingual", lexical_accents_preserved=True,
                      ngram_range=[1, 2], topic_reduction="none", fit_weight="one_per_unique_song_block")
    return topic_model, [int(t) for t in labels], parameters


def agreement(left: list[int], right: list[int]) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    left, right = np.asarray(left), np.asarray(right)
    mask = (left != -1) & (right != -1)

    def aligned_fraction(a, b):
        if len(a) == 0:
            return None
        # Noise has a substantive meaning; it must not be renamed to a topic to
        # inflate agreement. Only nonnoise cluster IDs may be permuted.
        ua, ub = np.unique(a[a != -1]), np.unique(b[b != -1])
        matches = int(np.sum((a == -1) & (b == -1)))
        if len(ua) and len(ub):
            table = np.asarray([[np.sum((a == x) & (b == y)) for y in ub] for x in ua])
            rows, columns = linear_sum_assignment(-table)
            matches += int(table[rows, columns].sum())
        return matches / len(a)

    return dict(documents=len(left), ari_including_noise=float(adjusted_rand_score(left, right)),
                label_aligned_fraction_including_noise=aligned_fraction(left, right),
                both_nonnoise_documents=int(mask.sum()),
                both_nonnoise_fraction=float(mask.mean()),
                ari_on_both_nonnoise=float(adjusted_rand_score(left[mask], right[mask])) if mask.sum() >= 2 else None,
                label_aligned_fraction_both_nonnoise=aligned_fraction(left[mask], right[mask]),
                left_noise_fraction=float((left == -1).mean()),
                right_noise_fraction=float((right == -1).mean()),
                left_nonnoise_clusters=len(set(left.tolist()) - {-1}),
                right_nonnoise_clusters=len(set(right.tolist()) - {-1}),
                limitation="High agreement on a single/noise cluster is not meaningful topic validation.")


def topic_vocabulary(topic_model, labels: list[int], sections: list[Section]) -> list[dict]:
    result = []
    for topic in sorted(set(labels)):
        words = topic_model.get_topic(topic) or []
        result.append(dict(topic_id=topic, unique_blocks=labels.count(topic),
                           songs=len({s.song_id for s, t in zip(sections, labels) if t == topic}),
                           is_noise=topic == -1,
                           terms=[dict(term=str(word), ctfidf_weight=float(weight))
                                  for word, weight in words if word]))
    return result


def topic_hierarchy(topic_model, sections: list[Section], embeddings, labels: list[int]) -> dict:
    import numpy as np
    topics = sorted(set(labels) - {-1})
    if len(topics) < 2:
        return dict(status="insufficient_nonnoise_topics", edges=[],
                    limitation="No subtopic hierarchy can be estimated from fewer than two nonnoise topics.")
    try:
        hierarchy = topic_model.hierarchical_topics([s.text for s in sections])
        rows = []
        for _, row in hierarchy.iterrows():
            rows.append(dict(parent_id=str(row["Parent_ID"]), parent_name=str(row["Parent_Name"]),
                             child_left_id=str(row["Child_Left_ID"]), child_right_id=str(row["Child_Right_ID"]),
                             topic_ids=[int(v) for v in row["Topics"]], distance=float(row["Distance"])))
        return dict(status="ok", method="BERTopic hierarchical_topics / c-TF-IDF agglomeration",
                    edges=rows, limitation="A similarity hierarchy, not discovered causes or validated taxonomic subtopics.")
    except Exception as exc:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist
        centroids = np.asarray([embeddings[np.asarray(labels) == topic].mean(axis=0) for topic in topics])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        distances = np.clip(pdist(centroids, metric="cosine"), 0, 2)
        tree = linkage(distances, method="average")
        members, rows = {i: [t] for i, t in enumerate(topics)}, []
        for i, (a, b, distance, size) in enumerate(tree):
            parent = len(topics) + i
            members[parent] = members[int(a)] + members[int(b)]
            rows.append(dict(parent_id=f"node_{parent}", child_left_id=f"node_{int(a)}",
                             child_right_id=f"node_{int(b)}", topic_ids=members[parent],
                             distance=float(distance), size=int(size)))
        return dict(status="fallback", failed_method_error_type=type(exc).__name__,
                    method="average linkage of cosine distances between original-space topic centroids",
                    leaves=[dict(node_id=f"node_{i}", topic_id=t) for i, t in enumerate(topics)], edges=rows,
                    limitation="Fallback similarity hierarchy; it does not validate literary subtopics.")


def exploratory_map(embeddings, sections: list[Section], labels: list[int], args) -> list[dict]:
    from sklearn.decomposition import PCA
    from umap import UMAP
    if len(sections) >= 6:
        xy = UMAP(n_neighbors=min(args.n_neighbors, len(sections) - 1), n_components=2,
                  min_dist=0.15, metric="cosine", random_state=args.seeds[0],
                  n_jobs=1, low_memory=True).fit_transform(embeddings)
        method = "UMAP_2D_separate_visualization_fit"
    else:
        xy = PCA(n_components=2).fit_transform(embeddings)
        method = "PCA_2D_fallback"
    return [dict(document_id=s.document_id, song_id=s.song_id, topic_id=t,
                 x=float(p[0]), y=float(p[1]), method=method,
                 interpretation="Exploratory layout; do not measure semantic distances in 2D.")
            for s, t, p in zip(sections, labels, xy)]


def song_topic_matrix(songs: list[Song], sections: list[Section], assignment: dict[str, int], unique: bool):
    import numpy as np
    topics = sorted(set(assignment.values()))
    lookup = {t: i for i, t in enumerate(topics)}
    counts = defaultdict(Counter)
    for section in sections:
        if unique and section.is_repeat:
            continue
        counts[section.song_id][assignment[section.canonical_id]] += 1
    eligible = [s for s in songs if counts[s.song_id]]
    matrix = np.zeros((len(eligible), len(topics)))
    for i, song in enumerate(eligible):
        denominator = sum(counts[song.song_id].values())
        for topic, count in counts[song.song_id].items():
            matrix[i, lookup[topic]] = count / denominator
    return eligible, topics, matrix


def stratified_mean(songs: list[Song], matrix):
    import numpy as np
    languages = sorted({s.language for s in songs})
    return np.mean([matrix[[i for i, s in enumerate(songs) if s.language == lang]].mean(axis=0)
                    for lang in languages], axis=0)


def bootstrap_composition(songs: list[Song], topics: list[int], matrix, args, variant: str):
    import numpy as np
    rng = np.random.default_rng(args.bootstrap_seed)
    rows, draws_by_decade = [], {}
    for decade in sorted({s.decade for s in songs}):
        indices = [i for i, s in enumerate(songs) if s.decade == decade]
        group = [songs[i] for i in indices]
        values = matrix[indices]
        languages = sorted({s.language for s in group})
        language_draws, artist_means = [], []
        for lang in languages:
            loc = [i for i, s in enumerate(group) if s.language == lang]
            langvalues = values[loc]
            sample = rng.integers(0, len(loc), size=(args.bootstrap_samples, len(loc)))
            language_draws.append(langvalues[sample].mean(axis=1))
            artist_values = defaultdict(list)
            for i in loc:
                artist_values[group[i].artist.casefold()].append(values[i])
            artist_means.append(np.mean([np.mean(v, axis=0) for v in artist_values.values()], axis=0))
            for j, topic in enumerate(topics):
                rows.append(dict(variant=variant, decade=decade, language=lang, topic_id=topic,
                                 song_normalized_share=float(langvalues[:, j].mean()),
                                 ci_low=float(np.quantile(language_draws[-1][:, j], .025)),
                                 ci_high=float(np.quantile(language_draws[-1][:, j], .975)),
                                 songs=len(loc), artists=len(artist_values),
                                 weighting="equal_song_within_language", languages_present=len(languages)))
        draws = np.mean(language_draws, axis=0)
        draws_by_decade[decade] = draws
        estimate = stratified_mean(group, values)
        artist_estimate = np.mean(artist_means, axis=0)
        for j, topic in enumerate(topics):
            rows.append(dict(variant=variant, decade=decade, language="balanced_available_languages", topic_id=topic,
                             song_normalized_share=float(estimate[j]),
                             ci_low=float(np.quantile(draws[:, j], .025)),
                             ci_high=float(np.quantile(draws[:, j], .975)),
                             artist_balanced_share=float(artist_estimate[j]), songs=len(group),
                             artists=len({s.artist.casefold() for s in group}),
                             weighting="equal_language_then_equal_song", languages_present=len(languages)))
    contrasts = []
    decades = sorted(draws_by_decade)
    pairs = list(zip(decades[:-1], decades[1:]))
    if len(decades) > 2:
        pairs.append((decades[0], decades[-1]))
    for first, last in pairs:
        # Report a contrast only when the same two languages exist in both cells.
        lf = {s.language for s in songs if s.decade == first}
        ll = {s.language for s in songs if s.decade == last}
        if lf != {"en", "es"} or ll != {"en", "es"}:
            continue
        difference = draws_by_decade[last] - draws_by_decade[first]
        first_values = matrix[[i for i, s in enumerate(songs) if s.decade == first]]
        last_values = matrix[[i for i, s in enumerate(songs) if s.decade == last]]
        point = stratified_mean([s for s in songs if s.decade == last], last_values) - stratified_mean(
            [s for s in songs if s.decade == first], first_values)
        for j, topic in enumerate(topics):
            contrasts.append(dict(variant=variant, first_decade=first, last_decade=last, topic_id=topic,
                                  share_difference=float(point[j]),
                                  ci_low=float(np.quantile(difference[:, j], .025)),
                                  ci_high=float(np.quantile(difference[:, j], .975)),
                                  inference="Conditional resampling of selected songs; not a population trend test."))
    return rows, contrasts


def artist_controls(songs: list[Song], topics: list[int], matrix):
    import numpy as np
    artist_indices = defaultdict(list)
    for i, song in enumerate(songs):
        artist_indices[song.artist.casefold()].append(i)
    concentrations, leaveout = [], []
    for j, topic in enumerate(topics):
        masses = {artist: float(matrix[indices, j].sum()) for artist, indices in artist_indices.items()}
        total = sum(masses.values())
        shares = {a: m / total for a, m in masses.items()} if total else {}
        largest = max(shares, key=shares.get) if shares else None
        concentrations.append(dict(topic_id=topic, artist_hhi=sum(p * p for p in shares.values()) if shares else None,
                                   largest_artist=largest, largest_artist_share=shares.get(largest),
                                   contributing_artists=sum(m > 0 for m in masses.values()),
                                   total_song_normalized_mass=total,
                                   limitation="Artist concentration within this selected corpus, not generalization."))
    for decade in sorted({s.decade for s in songs}):
        indices = [i for i, s in enumerate(songs) if s.decade == decade]
        group, values = [songs[i] for i in indices], matrix[indices]
        baseline = stratified_mean(group, values)
        present_languages = {s.language for s in group}
        for artist in sorted({s.artist.casefold() for s in group}):
            keep = [i for i, s in enumerate(group) if s.artist.casefold() != artist]
            remaining = [group[i] for i in keep]
            comparable = bool(keep) and {s.language for s in remaining} == present_languages
            changed = stratified_mean(remaining, values[keep]) if comparable else None
            for j, topic in enumerate(topics):
                leaveout.append(dict(decade=decade, omitted_artist=artist, topic_id=topic,
                                     original_share=float(baseline[j]),
                                     leave_artist_out_share=float(changed[j]) if comparable else None,
                                     difference=float(changed[j] - baseline[j]) if comparable else None,
                                     same_language_support=comparable,
                                     songs_remaining=len(remaining),
                                     model_refitted=False))
    return concentrations, leaveout


def semantic_probes(encoder: Encoder) -> dict:
    import numpy as np
    inputs = probe_inputs()
    vectors = encoder.encode([p["text"] for p in inputs], "passage")
    diagnostics = encoder.token_diagnostics([
        Section(p["input_id"], "synthetic", i, p["text"], "synthetic", "synthetic", p["input_id"],
                False, len(TOKEN.findall(p["text"])), 1) for i, p in enumerate(inputs)], kind="probes")
    rows = []
    for i, probe in enumerate(PROBES):
        a, b, c = vectors[i * 3:i * 3 + 3]
        # Explicit float64 sums avoid spurious floating-point flags emitted by
        # some Accelerate BLAS builds on macOS for otherwise finite L2 vectors.
        related = float(np.sum(a.astype(np.float64) * b.astype(np.float64)))
        contrast = float(np.sum(a.astype(np.float64) * c.astype(np.float64)))
        if not math.isfinite(related) or not math.isfinite(contrast):
            raise FloatingPointError("Nonfinite synthetic cosine.")
        row = dict(**probe, related_cosine=related, contrast_cosine=contrast,
                         related_minus_contrast=related - contrast,
                         preferred_context_ranks_first=related > contrast if probe["category"] in
                         {"crosslingual", "fear_context", "paraphrase", "polarity_context", "motive_context",
                          "experiencer", "narrative_order", "long_ending"} else None)
        for offset, key in enumerate(("query", "related", "contrast")):
            diagnostic = diagnostics[i * 3 + offset]
            row[f"{key}_tokenizer_tokens"] = diagnostic["tokenizer_tokens"]
            row[f"{key}_exceeds_max_seq_length"] = diagnostic["exceeds_max_seq_length"]
        rows.append(row)
    return dict(status="executed", construction="Original synthetic bilingual sentences; predeclared diagnostic contrasts.",
                probes=rows, polarity_accuracy="not_measured", role_accuracy="not_measured",
                limitation="Twenty hand-authored probes diagnose possible retrieval confounds. They are neither a benchmark "
                           "nor evidence that a model recognizes fear in real songs.")


def bh_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, bound = [1.0] * len(p_values), 1.0
    for rank in range(len(ordered), 0, -1):
        index = ordered[rank - 1]
        bound = min(bound, p_values[index] * len(ordered) / rank)
        adjusted[index] = bound
    return adjusted


def semantic_arcs(encoder: Encoder, songs: list[Song], sections: list[Section], all_vectors, args, unique: bool):
    import numpy as np
    states = list(PROTOTYPES)
    prototype_texts = [p["text"] for p in prototype_inputs()]
    pvec = encoder.encode(prototype_texts, "query")
    centroids = pvec.astype(np.float64).reshape(len(states), 6, -1).mean(axis=1)
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    similarities = np.einsum("ij,kj->ik", all_vectors.astype(np.float64), centroids, optimize=False)
    if not np.isfinite(similarities).all() or np.abs(similarities).max() > 1.000001:
        raise FloatingPointError("Nonfinite or unbounded prototype cosine.")
    variant = "unique_first_occurrence" if unique else "all_blocks_with_repeats"
    metadata = {s.song_id: s for s in songs}
    by_song = defaultdict(list)
    for i, section in enumerate(sections):
        if not unique or not section.is_repeat:
            by_song[section.song_id].append(i)
    section_rows, song_rows, path_rows, nulls, tested_songs, observed = [], [], [], [], [], []
    rng = np.random.default_rng(args.permutation_seed)
    for song_id, indices in by_song.items():
        song = metadata[song_id]
        values = similarities[indices]
        for local_order, (index, scores) in enumerate(zip(indices, values)):
            section = sections[index]
            section_rows.append(dict(variant=variant, document_id=section.document_id, song_id=song_id,
                                     analysis_order=local_order, **{f"{state}_cosine": float(scores[j])
                                                                  for j, state in enumerate(states)}))
        # A diagnostic semantic contrast, not emotion valence or narrative quality.
        contrast = values[:, states.index("coping")] - (
            values[:, states.index("fear")] + values[:, states.index("threat")]) / 2
        row = dict(variant=variant, song_id=song_id, decade=song.decade, language=song.language,
                   blocks=len(indices), tested=len(indices) >= args.min_arc_blocks,
                   contrast_definition="coping_cosine - (fear_cosine + threat_cosine) / 2")
        if row["tested"]:
            time_axis = np.linspace(0, 1, len(indices))
            centered = time_axis - time_axis.mean()
            denominator = float(np.einsum("i,i->", centered, centered, optimize=False))
            slope = float(np.einsum("i,i->", centered, contrast, optimize=False) / denominator)
            perm = np.asarray([rng.permutation(contrast) for _ in range(args.permutations)])
            null = np.einsum("ij,j->i", perm, centered, optimize=False) / denominator
            if not np.isfinite(null).all():
                raise FloatingPointError("Nonfinite order permutation statistic.")
            p = float((1 + np.sum(np.abs(null) >= abs(slope) - 1e-12)) / (args.permutations + 1))
            row.update(order_slope=slope, permutation_p_two_sided=p,
                       null_mean=float(null.mean()), null_sd=float(null.std()),
                       first_to_last_difference=float(contrast[-1] - contrast[0]))
            nulls.append(null)
            observed.append(slope)
            tested_songs.append(song)
            grid = np.linspace(0, 1, args.arc_grid_points)
            for point in grid:
                interpolated = [float(np.interp(point, time_axis, values[:, j])) for j in range(len(states))]
                path_rows.append(dict(variant=variant, song_id=song_id, decade=song.decade,
                                      language=song.language, normalized_position=float(point),
                                      **{f"{state}_cosine": interpolated[j] for j, state in enumerate(states)},
                                      semantic_contrast=float(np.interp(point, time_axis, contrast))))
        song_rows.append(row)
    tested_rows = [r for r in song_rows if r["tested"]]
    for row, corrected in zip(tested_rows, bh_adjust([r["permutation_p_two_sided"] for r in tested_rows])):
        row["permutation_q_bh_within_variant"] = corrected
    tests = []
    if tested_songs:
        matrix, slopes = np.asarray(nulls), np.asarray(observed)
        groups = [("all_selected_songs", list(range(len(tested_songs))))]
        groups += [(str(d), [i for i, s in enumerate(tested_songs) if s.decade == d])
                   for d in sorted({s.decade for s in tested_songs})]
        for group, indices in groups:
            languages = sorted({tested_songs[i].language for i in indices})
            language_indices = [[i for i in indices if tested_songs[i].language == lang] for lang in languages]
            obs = float(np.mean([slopes[ix].mean() for ix in language_indices]))
            null = np.mean([matrix[ix].mean(axis=0) for ix in language_indices], axis=0)
            tests.append(dict(variant=variant, group=group, songs=len(indices), languages_present=len(languages),
                              mean_order_slope=obs, shuffled_mean=float(null.mean()),
                              shuffled_ci_low=float(np.quantile(null, .025)),
                              shuffled_ci_high=float(np.quantile(null, .975)),
                              permutation_p_two_sided=float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) /
                                                             (args.permutations + 1)),
                              weighting="equal_language_then_equal_song",
                              inference="Tests section order against exchangeability conditional on the selected texts and prototypes."))
        for row, corrected in zip(tests, bh_adjust([r["permutation_p_two_sided"] for r in tests])):
            row["permutation_q_bh_within_variant"] = corrected
    return dict(section_rows=section_rows, song_rows=song_rows, paths=path_rows,
                order_tests=tests, prototypes=PROTOTYPES,
                limitation="These are prototype-similarity trajectories, not validated emotions or literary plot arcs. "
                           "Stanza/window order is inferred from provider formatting; interpolation is for visualization. "
                           "Shuffling does not control repetition, metaphor, narrator changes, or historical confounds.")


def aggregate_paths(path_rows: list[dict], args):
    import numpy as np
    rng = np.random.default_rng(args.bootstrap_seed)
    groups = defaultdict(list)
    for row in path_rows:
        groups[(row["variant"], row["decade"], row["normalized_position"])].append(row)
    out = []
    metrics = [f"{state}_cosine" for state in PROTOTYPES] + ["semantic_contrast"]
    for (variant, decade, position), rows in sorted(groups.items()):
        langs = sorted({r["language"] for r in rows})
        language_draws, means = [], []
        for language in langs:
            values = np.asarray([[r[m] for m in metrics] for r in rows if r["language"] == language])
            means.append(values.mean(axis=0))
            sampled = rng.integers(0, len(values), size=(args.bootstrap_samples, len(values)))
            language_draws.append(values[sampled].mean(axis=1))
        mean, draws = np.mean(means, axis=0), np.mean(language_draws, axis=0)
        for j, metric in enumerate(metrics):
            out.append(dict(variant=variant, decade=decade, normalized_position=position,
                            metric=metric, mean=float(mean[j]), ci_low=float(np.quantile(draws[:, j], .025)),
                            ci_high=float(np.quantile(draws[:, j], .975)),
                            songs=len(rows), languages_present=len(langs),
                            weighting="equal_language_then_equal_song",
                            caution="Pointwise descriptive intervals, not simultaneous confidence bands."))
    return out


def package_versions() -> dict:
    result = {}
    for package in ("numpy", "scipy", "scikit-learn", "pandas", "sentence-transformers", "torch",
                    "transformers", "bertopic", "umap-learn", "hdbscan"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def analyze_model(model_id: str, songs: list[Song], sections: list[Section], args):
    import numpy as np
    output = args.output / slug(model_id)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    encoder = FileEncoder(args, sections) if args.embedding_file else Encoder(model_id, args)
    unique = [s for s in sections if not s.is_repeat]
    embeddings = encoder.encode([s.text for s in unique])
    if not args.embedding_file:
        write_json(encoder.cache / "document-index.json", dict(
            model_id=model_id, array_path=str(encoder.last_cache_path.resolve()),
            document_ids=[s.document_id for s in unique], document_hashes=[sha(s.text) for s in unique],
            shape=list(embeddings.shape), normalization="L2",
            resolved_snapshot_commit=encoder.commit,
            privacy="PRIVATE: embedding matrix is derived from copyrighted lyric blocks."))
    diagnostics = encoder.token_diagnostics(unique)
    write_csv(output / "tokenization_diagnostics.csv", diagnostics)
    write_json(output / "semantic_probes.json", semantic_probes(encoder))
    runs, failures, primary = [], [], None
    for seed in args.seeds:
        try:
            model, labels, parameters = fit_topics(unique, embeddings, seed, args)
            runs.append((seed, labels))
            if primary is None:
                primary = (model, labels, parameters)
        except Exception as exc:
            # Do not log exception messages: provider/model errors can include text.
            failures.append(dict(seed=seed, status="failed", error_type=type(exc).__name__,
                                 trace=safe_failure_trace(exc),
                                 reason="Topic estimation failed; no replacement labels were imposed."))
    write_json(output / "topic_fit_status.json", dict(
        requested_seeds=args.seeds, successful_seeds=[r[0] for r in runs], failures=failures,
        status="ok" if primary is not None else "failed_no_topics_estimated"))
    returned_labels = None
    if primary is not None:
        topic_model, labels, parameters = primary
        returned_labels = labels
        assignment = {s.document_id: t for s, t in zip(unique, labels)}
        write_json(output / "topics.json", dict(parameters=parameters,
                                                topics=topic_vocabulary(topic_model, labels, unique),
                                                interpretation="c-TF-IDF descriptors require human naming and inspection. "
                                                               "A topic is not evidence of an emotion."))
        write_json(output / "topic_hierarchy.json", topic_hierarchy(topic_model, unique, embeddings, labels))
        write_csv(output / "section_assignments.csv", [dict(**public_section(s),
                                                           topic_id=assignment[s.canonical_id]) for s in sections])
        write_csv(output / "seed_assignments.csv", [dict(seed=seed, document_id=s.document_id, topic_id=t)
                                                    for seed, ts in runs for s, t in zip(unique, ts)])
        stability = [dict(left_seed=runs[a][0], right_seed=runs[b][0], **agreement(runs[a][1], runs[b][1]))
                     for a in range(len(runs)) for b in range(a + 1, len(runs))]
        write_csv(output / "seed_stability.csv", stability)
        write_csv(output / "exploratory_map.csv", exploratory_map(embeddings, unique, labels, args))
        composition, contrasts, matrices = [], [], {}
        for remove_repeats, variant in ((True, "unique_first_occurrence"), (False, "all_blocks_with_repeats")):
            eligible, topics, matrix = song_topic_matrix(songs, sections, assignment, remove_repeats)
            matrices[variant] = (eligible, topics, matrix)
            values, differences = bootstrap_composition(eligible, topics, matrix, args, variant)
            composition.extend(values)
            contrasts.extend(differences)
            write_csv(output / f"song_topic_shares_{variant}.csv", [dict(song_id=s.song_id, topic_id=t,
                       share=float(matrix[i, j]), decade=s.decade, language=s.language)
                       for i, s in enumerate(eligible) for j, t in enumerate(topics)])
        write_csv(output / "decade_composition.csv", composition)
        write_csv(output / "decade_contrasts.csv", contrasts)
        indexed = {(r["variant"], r["decade"], r["language"], r["topic_id"]): r for r in composition}
        unique_songs, unique_topics, unique_matrix = matrices["unique_first_occurrence"]
        repeated_songs, repeated_topics, repeated_matrix = matrices["all_blocks_with_repeats"]
        if [s.song_id for s in unique_songs] != [s.song_id for s in repeated_songs] or unique_topics != repeated_topics:
            raise ValueError("Matched repetition comparison lost its common song/topic denominator.")
        paired, _ = bootstrap_composition(unique_songs, unique_topics, repeated_matrix - unique_matrix,
                                           args, "paired_repeated_minus_unique")
        paired_index = {(r["decade"], r["language"], r["topic_id"]): r for r in paired}
        sensitivity = []
        for row in composition:
            if row["variant"] != "unique_first_occurrence":
                continue
            repeated = indexed[("all_blocks_with_repeats", row["decade"], row["language"], row["topic_id"])]
            interval = paired_index[(row["decade"], row["language"], row["topic_id"])]
            sensitivity.append(dict(decade=row["decade"], language=row["language"], topic_id=row["topic_id"],
                                    unique_share=row["song_normalized_share"],
                                    repeated_share=repeated["song_normalized_share"],
                                    repeated_minus_unique=repeated["song_normalized_share"] - row["song_normalized_share"],
                                    paired_ci_low=interval["ci_low"], paired_ci_high=interval["ci_high"],
                                    same_fitted_topics=True))
        write_csv(output / "repeated_refrain_sensitivity.csv", sensitivity)
        concentration, leaveout = artist_controls(*matrices["unique_first_occurrence"])
        write_csv(output / "artist_concentration.csv", concentration)
        write_csv(output / "leave_one_artist_out_prevalence.csv", leaveout)
    # Arcs remain analyzable when a density-based topic fit honestly finds no clusters.
    embedding_lookup = {s.document_id: embeddings[i] for i, s in enumerate(unique)}
    all_vectors = np.asarray([embedding_lookup[s.canonical_id] for s in sections])
    arc_paths, arc_sections, arc_songs, order_tests = [], [], [], []
    arc_limit = None
    for remove_repeats in (True, False):
        arcs = semantic_arcs(encoder, songs, sections, all_vectors, args, remove_repeats)
        arc_paths.extend(arcs["paths"])
        arc_sections.extend(arcs["section_rows"])
        arc_songs.extend(arcs["song_rows"])
        order_tests.extend(arcs["order_tests"])
        arc_limit = arcs["limitation"]
    write_json(output / "narrative_method.json", dict(prototypes=PROTOTYPES, limitation=arc_limit,
                                                      min_blocks_for_order_test=args.min_arc_blocks,
                                                      permutations=args.permutations,
                                                      permutation_seed=args.permutation_seed,
                                                      arithmetic="float64 einsum with explicit finiteness checks",
                                                      permutation_tie_tolerance=1e-12,
                                                      p_adjustment="Benjamini-Hochberg within each variant; exploratory families"))
    write_csv(output / "narrative_section_scores.csv", arc_sections)
    write_csv(output / "narrative_song_order_tests.csv", arc_songs)
    write_csv(output / "narrative_paths.csv", arc_paths)
    write_csv(output / "narrative_decade_paths.csv", aggregate_paths(arc_paths, args))
    write_csv(output / "narrative_order_tests.csv", order_tests)
    write_json(output / "model_manifest.json", dict(
        encoder=encoder.manifest(), runtime_seconds=round(time.monotonic() - start, 2),
        unique_documents=len(unique), truncated_documents=sum(r["exceeds_max_seq_length"] is True for r in diagnostics),
        tokenization_unknown_documents=sum(r["exceeds_max_seq_length"] is None for r in diagnostics),
        packages=package_versions(), privacy="Lyrics, provider headings, embeddings, and fitted models are private.",
        results_status="executed", topic_status=("estimated" if set(returned_labels or []) - {-1}
                                                  else "all_noise_no_nonnoise_topics" if returned_labels is not None else "failed")))
    return returned_labels


def self_test() -> None:
    import numpy as np
    song = Song("synthetic", "Synthetic test", "Test artist", 1983, 1980, "en", "ambiguous",
                "[Verse 1]\nI fear the danger approaching me.\nI hide away from that danger.\n\n"
                "[Chorus]\nTogether we can face the unknown.\nWe will find a way forward.\n\n"
                "[Chorus]\nTogether we can face the unknown.\nWe will find a way forward.",
                "synthetic", "", "")
    sections = segment(song, 8, 4)
    assert len(sections) == 3 and sum(s.is_repeat for s in sections) == 1
    assert sections[2].canonical_id == sections[1].document_id
    assert all("text" not in public_section(s) and "label" not in public_section(s) for s in sections)
    assignment = {sections[0].document_id: 0, sections[1].document_id: 1}
    _, _, unique = song_topic_matrix([song], sections, assignment, True)
    _, _, repeated = song_topic_matrix([song], sections, assignment, False)
    assert np.allclose(unique, [[.5, .5]])
    assert np.allclose(repeated, [[1 / 3, 2 / 3]])
    assert all(0 <= p <= 1 for p in bh_adjust([.01, .04, .8]))
    assert agreement([0, 0, 1, 1], [7, 7, 3, 3])["ari_including_noise"] == 1
    assert agreement([-1, -1], [0, 0])["label_aligned_fraction_including_noise"] == 0
    print("Self-test passed: section privacy, exact-repeat deduplication, song weighting, label-invariant agreement.")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path(".research-cache/fear-lyrics/corpus.jsonl"))
    p.add_argument("--output", type=Path, default=Path("research/fear-lyrics/results"))
    p.add_argument("--private-cache", type=Path, default=Path(".research-cache/fear-lyrics"))
    p.add_argument("--model-cache", type=Path, default=Path(".research-cache/models"))
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--model-revision", action="append", default=[], metavar="MODEL_ID=COMMIT")
    p.add_argument("--embedding-file", type=Path, help="Private NPZ: documents, probes, prototypes; no provider calls.")
    p.add_argument("--embedding-metadata", type=Path, help="JSON sidecar; defaults to embedding-file with .json suffix.")
    p.add_argument("--model-label", help="Label for an external embedding bundle; no effect on local model loading.")
    p.add_argument("--export-embedding-input", type=Path, help="PRIVATE JSON inputs for another embedding provider.")
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20261011)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--permutation-seed", type=int, default=20261012)
    p.add_argument("--min-cluster-size", type=int, default=8)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--top-words", type=int, default=10)
    p.add_argument("--max-block-lines", type=int, default=8, help="0 keeps provider paragraphs intact.")
    p.add_argument("--min-block-words", type=int, default=4)
    p.add_argument("--min-arc-blocks", type=int, default=3)
    p.add_argument("--arc-grid-points", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-year", type=int, default=1980)
    p.add_argument("--max-year", type=int, default=2026)
    p.add_argument("--cutoff", default="2026-10-11")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--allow-download", action="store_true", help="Opt into free public model downloads.")
    p.add_argument("--dry-run", action="store_true", help="Validate and inventory input without loading models.")
    p.add_argument("--self-test", action="store_true", help="Run synthetic correctness checks, without model downloads.")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.offline and args.allow_download:
        raise ValueError("--offline and --allow-download are mutually exclusive.")
    date.fromisoformat(args.cutoff)
    if args.min_year > args.max_year or args.max_year > date.fromisoformat(args.cutoff).year:
        raise ValueError("The release-year window must end no later than the cutoff year.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be present and unique.")
    if min(args.bootstrap_samples, args.permutations, args.min_arc_blocks, args.arc_grid_points,
           args.min_block_words, args.batch_size, args.min_cluster_size, args.min_samples,
           args.n_neighbors, args.top_words) < 1 or args.max_block_lines < 0:
        raise ValueError("Sample counts and positive parameters must be >= 1.")
    if args.n_neighbors < 2 or args.min_cluster_size < 2 or args.arc_grid_points < 2 or args.min_arc_blocks < 3:
        raise ValueError("Neighbors/cluster size/grid points >= 2 and arc blocks >= 3 are required.")
    for specification in args.model_revision:
        if "=" not in specification:
            raise ValueError("--model-revision requires MODEL_ID=COMMIT.")
    if args.output.resolve() == args.private_cache.resolve() or args.output.resolve() in args.private_cache.resolve().parents:
        raise ValueError("The private cache cannot be placed inside the public output directory.")
    songs, notices = read_corpus(args.input, args)
    sections = [section for song in songs for section in segment(song, args.max_block_lines, args.min_block_words)]
    if not sections:
        raise ValueError("No eligible stanza/window blocks were found.")
    summary = corpus_summary(songs, sections, notices, args)
    if args.export_embedding_input:
        export_embedding_input(args.export_embedding_input, sections, args)
    if args.embedding_file:
        external_metadata = json.loads((args.embedding_metadata or args.embedding_file.with_suffix(".json")).read_text(encoding="utf-8"))
        args.models = [args.model_label or str(external_metadata.get("model_id", "external_embeddings"))]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "corpus_diagnostics.json", summary)
    write_csv(args.output / "section_inventory.csv", [public_section(s) for s in sections])
    write_json(args.output / "analysis_protocol.json", dict(
        executed=not args.dry_run, script_sha256=sha(Path(__file__).read_text(encoding="utf-8")),
        study_cutoff=args.cutoff, sampling_design=summary["design"],
        models=args.models, seeds=args.seeds, bootstrap_samples=args.bootstrap_samples,
        semantic_probe_triplets=len(PROBES), semantic_probe_vectors=len(probe_inputs()),
        cosine_arithmetic="float64 explicit sums/einsum; finite/bounded checks",
        permutation_tie_tolerance=1e-12,
        descriptor_negation_words_preserved=sorted(NEGATION_WORDS),
        bootstrap_seed=args.bootstrap_seed, permutations=args.permutations,
        permutation_seed=args.permutation_seed, max_block_lines=args.max_block_lines,
        min_block_words=args.min_block_words, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, umap_n_neighbors=args.n_neighbors,
        min_arc_blocks=args.min_arc_blocks, arc_grid_points=args.arc_grid_points,
        hypothesis_labels="Never used for clustering, prototype construction, validation, or retrieval scoring.",
        topics="One global density-based fit per model/seed on unique within-song blocks; no forced fear taxonomy.",
        denominators="Every eligible song contributes total weight one, including HDBSCAN noise. "
                     "Primary decade summaries average languages equally, then songs equally within language.",
        repetition="Exact normalized block deduplication only; compare repeated and unique weights using the same fitted topics.",
        uncertainty="Percentile resampling of songs within decade/language cells; intervals condition on the chosen corpus. "
                    "Selection, artists, provider coverage, embedding choice, and topic fit uncertainty are not covered by these CIs.",
        artist_control="Artist-balanced prevalence and leave-one-artist-out prevalence sensitivity; topic models are not refitted.",
        stability="Adjusted Rand and best label alignment across seeds/models on the same unique document IDs. "
                  "Noise-inclusive and jointly nonnoise denominators are both reported.",
        visualization="2D UMAP is a separate exploratory fit. Original embedding cosine values drive probes and prototypes.",
        literary_arcs="Ordered prototype similarity is an exploratory operationalization, not a validated plot-arc estimator.",
        privacy="No full lyrics, fragments, provider headings, representative documents, embeddings, or fitted models are public.",
        synthetic_texts="All diagnostic probes and prototypes were authored for this analysis and are public.",
    ))
    print(f"Validated {summary['songs_with_lyrics']} songs; {summary['unique_blocks']} unique blocks; "
          f"{summary['exact_repeated_blocks']} exact repeat blocks. Labels remain hypotheses.")
    if args.dry_run:
        print("Dry run complete. Models, clustering, probes, and order tests were not executed.")
        return 0
    model_labels, failures, topic_failed_models = {}, [], []
    for model_id in args.models:
        print(f"Analyzing {model_id} with {len(args.seeds)} topic seeds; lyrics stay private.", flush=True)
        try:
            labels = analyze_model(model_id, songs, sections, args)
            if labels is not None:
                model_labels[model_id] = labels
            else:
                topic_failed_models.append(model_id)
        except Exception as exc:
            failures.append(dict(model_id=model_id, status="failed", error_type=type(exc).__name__,
                                 trace=safe_failure_trace(exc),
                                 reason="Analysis could not complete. No simulated or invented results were substituted."))
            write_json(args.output / slug(model_id) / "model_failure.json", failures[-1])
            print(f"Model {model_id} incomplete ({type(exc).__name__}); inspect environment/cache locally.", flush=True)
    comparisons = [dict(left_model=a, right_model=b, **agreement(model_labels[a], model_labels[b]))
                   for i, a in enumerate(model_labels) for b in list(model_labels)[i + 1:]]
    write_csv(args.output / "embedding_model_cluster_agreement.csv", comparisons)
    write_json(args.output / "run_status.json", dict(
        status="partial_or_failed" if failures else "partial_topic_fit" if topic_failed_models else "complete",
        models_with_topic_fit=list(model_labels),
        models_with_nonnoise_topics=[m for m, labels in model_labels.items() if set(labels) - {-1}],
        topic_failed_models=topic_failed_models,
        failed_models=failures, packages=package_versions(),
        warning="Executed output does not constitute human validation or a representative historical conclusion."))
    return 0 if not failures and not topic_failed_models else 2


if __name__ == "__main__":
    sys.exit(main())
