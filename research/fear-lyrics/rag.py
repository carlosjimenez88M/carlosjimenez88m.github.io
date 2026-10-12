#!/usr/bin/env python3
"""Offline retrieval benchmark and bounded, rule-controlled LangGraph workflow.

Private inputs contain lyrics. Public outputs contain original questions, IDs,
metrics, configuration, and evidence-package templates only. This experiment
does not evaluate a generative LLM, autonomous agent, or literary answer quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import TypedDict
import unicodedata

import numpy as np
from langgraph.graph import END, START, StateGraph

import analyze

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / ".research-cache/fear-lyrics"
PUBLIC = Path(__file__).resolve().parent / "results/rag"
STOP = set("a an the and or of to in on for with is are was were be by from that this which what how who when where find passage passages lyric lyrics song songs text texts does do did show shows describe describes en el la los las un una unos unas de del y o a al por para con es son era eran que este esta estos estas cual cuales como quien cuando donde encuentra pasaje pasajes letra letras cancion canciones describe describen muestra muestran".split())

# Generic bilingual search expansion. No artist/title, song selection label,
# topic assignment, evidence ID, or benchmark answer enters these rules.
GLOSSARY = {
    "surveillance": "vigilancia watching privacy observando privacidad",
    "privacy": "privacidad privacy watching observando vigilancia",
    "vigilancia": "surveillance watching privacy privacidad",
    "privacidad": "privacy watching surveillance vigilancia",
    "shower": "ducha shower wash washing hair",
    "ducha": "shower wash washing hair ducha",
    "sleep": "dormir sleep bed night noche cama",
    "dormir": "sleep bed night dormir noche cama",
    "monsters": "monstruos monsters beasts bed",
    "monstruos": "monsters beasts bed monstruos cama",
    "bombs": "bombas bombs guns tanks guerra",
    "bombas": "bombs guns tanks bombas guerra",
    "violence": "violencia violence fighting sangre",
    "violencia": "violence fighting blood violencia sangre",
    "children": "ninos child children hijos",
    "ninos": "children child baby ninos hijos",
    "mother": "madre mother mama",
    "madre": "mother mama madre",
    "money": "dinero money loans trabajo",
    "dinero": "money loans dinero trabajo",
    "childhood": "infancia young pretend infancia",
    "infancia": "childhood young pretend infancia",
    "fears": "miedos fears afraid scared",
    "temor": "fear afraid scared miedo",
    "fear": "miedo fear afraid scared",
    "miedo": "fear afraid scared miedo",
    "forget": "olvidar forget olvidar",
    "olvidar": "forget olvidar",
    "breathing": "respirar breathe respirar",
    "respirar": "breathing breathe respirar",
    "safety": "seguridad safe salvo safety",
    "seguridad": "safety safe salvo seguridad",
    "escape": "liberacion escape libre safe salvo",
    "liberacion": "escape free safe salvo liberacion",
    "power": "poder power control",
    "poder": "power control poder",
    "blame": "culpa fault blame",
    "culpa": "blame fault culpa",
    "prayer": "oracion pray prayers pido dios",
    "oracion": "prayer pray prayers pido dios",
    "blood": "sangre blood violence",
    "sangre": "blood violence sangre",
    "betrayal": "traicion betrayal cheated engañado",
    "traicion": "betrayal cheated engañado traicion",
}


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return [word for word in re.findall(r"[a-z0-9]+", normalized) if word not in STOP]


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class BM25:
    """Okapi BM25 with nonnegative Robertson IDF; no learned supervision."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.rows = [Counter(tokens(text)) for text in texts]
        self.lengths = np.asarray([sum(row.values()) for row in self.rows], dtype=float)
        self.mean_length = float(self.lengths.mean()) if len(self.rows) else 1.0
        frequencies = Counter(term for row in self.rows for term in row)
        n = len(self.rows)
        self.idf = {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
                    for term, count in frequencies.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.rows), dtype=float)
        # Query repetition should not boost a document in a rewrite loop.
        for term in set(tokens(query)):
            idf = self.idf.get(term, 0.0)
            if idf == 0:
                continue
            counts = np.asarray([row.get(term, 0) for row in self.rows], dtype=float)
            denominator = counts + self.k1 * (1 - self.b + self.b * self.lengths / self.mean_length)
            scores += idf * counts * (self.k1 + 1) / denominator
        return scores

    def lexical_support(self, query: str, row: int) -> tuple[int, float]:
        query_terms = set(tokens(query))
        # Ignore OOV query terms in coverage but retain them in the match count.
        total = sum(self.idf.get(term, 0.0) for term in query_terms)
        matched = query_terms.intersection(self.rows[row])
        coverage = sum(self.idf.get(term, 0.0) for term in matched) / total if total else 0.0
        return len(matched), float(coverage)


def questions() -> list[dict]:
    """Original analyst questions; support IDs were read before evaluation.

    Supports are a narrow diagnostic annotation, not exhaustive corpus qrels.
    Both language versions of each family share the same split and evidence.
    """
    specs = [
        ("privacy", "dev", "s1983-753cd2a7d41b", [2, 5, 8, 11, 12],
         "Which passages express a recurring feeling of surveillance and an inability to enjoy privacy?",
         "¿Qué pasajes expresan una sensación recurrente de vigilancia y la dificultad de disfrutar de privacidad?"),
        ("sleep_monsters", "dev", "s1991-d25354f1d8f0", [4],
         "Which passage mixes a bedtime prayer with monsters imagined beneath a bed or inside a closet?",
         "¿Qué pasaje mezcla una oración antes de dormir con monstruos imaginados bajo la cama o dentro del armario?"),
        ("prayer_community", "dev", "s2002-74fa275abd80", [1],
         "Which passage offers a prayer for future generations and for a community to stop spilling blood?",
         "¿Qué pasaje ofrece una oración por las generaciones futuras y por un pueblo que deje de derramar sangre?"),
        ("shower_intruder", "test", "s1983-753cd2a7d41b", [4],
         "Where does the speaker fear an unexpected intruder while washing their hair in the shower?",
         "¿Dónde teme el hablante encontrar a un intruso mientras se lava el cabello en la ducha?"),
        ("violence_history", "test", "s1994-56ff6ca6c8bb", [4, 5],
         "Which passages connect a grieving mother with recurring military violence and an explicit historical year?",
         "¿Qué pasajes conectan a una madre doliente con violencia militar recurrente y un año histórico explícito?"),
        ("mutual_fear", "test", "s2004-58ef65dc2ae6", [2, 4, 7],
         "Which passages combine fear of being feared by a loved person with fear of both another encounter and permanent separation?",
         "¿Qué pasajes combinan el temor de que la persona querida nos tema con miedo tanto a un reencuentro como a una separación definitiva?"),
        ("childhood_money", "test", "s2015-c11afde63d3a", [3, 4, 5],
         "Which passages contrast childhood games and dreams of space with adult demands to make money?",
         "¿Qué pasajes contrastan los juegos de infancia y los sueños del espacio con la exigencia adulta de ganar dinero?"),
        ("escape_safety", "test", "s2017-2c3252eb3c12", [3, 7, 8],
         "Which passages portray escape through an imagined doorway, followed by running, laughing, and reaching safety?",
         "¿Qué pasajes narran liberación mediante una puerta imaginada, seguida de correr, reír y ponerse a salvo?"),
        ("power_blame", "test", "s2021-27c16cbb8cff", [4],
         "Which passage connects abuse of power with assigned blame, confinement, and an excuse about the other person's age?",
         "¿Qué pasaje conecta el abuso de poder con la culpa impuesta, el encierro y una excusa sobre la edad de la otra persona?"),
        ("ability_reversal", "test", "s2020-6f072260cb90", [0, 3, 10],
         "Which passages contrast the speaker's inability to speak or breathe with a later assertion that these actions are possible?",
         "¿Qué pasajes contrastan la imposibilidad de hablar o respirar con una afirmación posterior de que esas acciones sí son posibles?"),
        ("lunar_bargain", "test", "s1986-e6cc85a542a7", [2, 9, 10],
         "Which passages connect a bargain to surrender a first child with a later accusation of betrayal and the child's abandonment?",
         "¿Qué pasajes conectan un pacto de entregar al primer hijo con una acusación posterior de traición y el abandono del niño?"),
    ]
    result = []
    for family, split, song_id, blocks, en, es in specs:
        ids = [f"{song_id}::s{block:03d}" for block in blocks]
        for language, query in (("en", en), ("es", es)):
            result.append(dict(query_id=f"{family}_{language}", family=family, split=split,
                               language=language, query=query, gold_document_ids=ids,
                               answerable=True, requires_all_evidence=family in {"ability_reversal", "lunar_bargain", "violence_history"}))
    negatives = [
        ("quantum_calculation", "dev", "Which passage calculates the wavelength of a quantum photon using an equation?",
         "¿Qué pasaje calcula mediante una ecuación la longitud de onda de un fotón cuántico?"),
        ("vaccine_batch", "test", "Which passage reports a numeric vaccine batch identifier and a controlled clinical trial?",
         "¿Qué pasaje informa de un identificador numérico de un lote de vacunas y un ensayo clínico controlado?"),
        ("moscow_metro", "test", "Which passage names a Moscow metro station and supplies its opening date?",
         "¿Qué pasaje nombra una estación del metro de Moscú e indica su fecha de inauguración?"),
        ("intruder_identity_missing", "test", "Which passage mentions fear while washing hair in the shower and gives the intruder's verified legal identity and arrest date?",
         "¿Qué pasaje menciona el miedo al lavarse el cabello en la ducha y proporciona la identidad legal verificada del intruso y la fecha de su detención?"),
        ("bedtime_clinical_missing", "test", "Which passage combines a bedtime prayer and monsters beneath the bed with a named clinical diagnosis and a prescribed medication dose?",
         "¿Qué pasaje combina una oración antes de dormir y monstruos bajo la cama con un diagnóstico clínico nombrado y una dosis prescrita de medicación?"),
    ]
    for family, split, en, es in negatives:
        for language, query in (("en", en), ("es", es)):
            result.append(dict(query_id=f"{family}_{language}", family=family, split=split,
                               language=language, query=query, gold_document_ids=[],
                               answerable=False, requires_all_evidence=False))
    return result


def ranking(scores: np.ndarray, limit: int) -> list[int]:
    # Stable tie breaking by the frozen document order, never by relevance gold.
    return list(np.argsort(-scores, kind="stable")[:limit].astype(int))


def rrf(left: list[int], right: list[int], limit: int, constant: int = 60) -> list[int]:
    scores = Counter()
    for order in (left, right):
        for position, row in enumerate(order, 1):
            scores[row] += 1 / (constant + position)
    return [row for row, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def metrics(order: list[str], gold: list[str], k: int) -> dict:
    if not gold:
        return {"recall": None, "mrr": None, "ndcg": None}
    selected, relevant = order[:k], set(gold)
    hits = [index for index, doc_id in enumerate(selected, 1) if doc_id in relevant]
    dcg = sum(1 / math.log2(position + 1) for position in hits)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return dict(recall=len(set(selected).intersection(relevant)) / len(relevant),
                mrr=1 / min(hits) if hits else 0.0, ndcg=dcg / ideal if ideal else 0.0)


def expand_query(query: str, attempt: int) -> str:
    expansions = list(dict.fromkeys(GLOSSARY[term] for term in tokens(query) if term in GLOSSARY))
    # Rewriting is explicit lexical bilingual expansion, not a language-model plan.
    if attempt == 1:
        return query + " " + " ".join(expansions)
    return " ".join(dict.fromkeys(tokens(query + " " + " ".join(expansions))))


class State(TypedDict, total=False):
    query: str
    language: str
    method: str
    agentic: bool
    rewrite_enabled: bool
    neighbors_enabled: bool
    narrow_search: bool
    search_query: str
    attempt: int
    ranked: list[int]
    all_ranked: list[int]
    sufficient: bool
    context_rows: list[int]
    context_words: int
    retrieved_slots: int
    search_calls: int
    trace: list[str]
    answer: str
    citations: list[str]
    abstained: bool
    citation_ids_valid: bool


class RemoteCachedEncoder:
    """Read-only hosted embeddings; this adapter never performs network calls.

    NPZ documents/queries share their order with sidecar document_hashes and
    query_hashes. Document IDs and hashes must match the audited input bundle.
    Missing query/rewrite vectors fail rather than silently invoke an API.
    """

    def __init__(self, model_id: str, bundle: dict, npz_path: Path, metadata_path: Path):
        self.model_id = model_id
        self.metadata = json.loads(metadata_path.read_text())
        docs = bundle["documents"]
        self.document_hashes = [d["text_sha256"] for d in docs]
        if self.metadata["document_ids"] != [d["input_id"] for d in docs] or self.metadata["document_hashes"] != self.document_hashes:
            raise ValueError("Remote corpus vectors do not match audited document identities/order.")
        actual_model = self.metadata.get("model_id", "").removeprefix("openai/")
        if actual_model != model_id.removeprefix("openai/"):
            raise ValueError("Remote embedding model identity mismatch.")
        with np.load(npz_path, allow_pickle=False) as arrays:
            self.documents = self.normalized(arrays["documents"])
            self.queries = self.normalized(arrays["queries"])
        if len(self.documents) != len(docs) or self.documents.shape[1] != self.queries.shape[1]:
            raise ValueError("Remote corpus/query embedding shapes mismatch.")
        query_hashes = self.metadata["query_hashes"]
        if len(query_hashes) != len(self.queries) or len(set(query_hashes)) != len(query_hashes):
            raise ValueError("Remote query hashes must be unique and align with query vectors.")
        self.passage_lookup = {h: i for i, h in enumerate(self.document_hashes)}
        self.query_lookup = {h: i for i, h in enumerate(query_hashes)}

    @staticmethod
    def normalized(array):
        vectors = np.asarray(array, dtype=np.float64)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Remote embeddings must be finite matrices.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Remote embeddings cannot contain zero vectors.")
        return vectors / norms

    def encode(self, texts, role="passage"):
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        if role == "passage" and hashes == self.document_hashes:
            return self.documents.copy()
        lookup, vectors = (self.query_lookup, self.queries) if role == "query" else (self.passage_lookup, self.documents)
        if any(h not in lookup for h in hashes):
            raise ValueError("Missing cached embedding: export and embed all bounded queries before offline RAG.")
        return vectors[[lookup[h] for h in hashes]]

    def manifest(self):
        return dict(model_id=self.model_id, provider="openai", input_type="verified_cached_remote_embeddings",
                    dimensions=self.documents.shape[1], documents=len(self.documents), queries=len(self.queries),
                    normalization="L2", api_calls_in_RAG=0, acquisition_usage=self.metadata.get("usage"),
                    revision=self.metadata.get("revision"),
                    limitation="Offline RAG uses already-acquired hosted vectors. API embedding acquisition costs/latencies are separate; hosted weight revision may be unpinned.")


def query_embedding_inputs(annotated=None):
    annotated = annotated or questions()
    texts = list(dict.fromkeys(text for q in annotated for text in (q["query"], expand_query(q["query"], 1), expand_query(q["query"], 2))))
    return [{"input_id": f"rq{i:04d}", "text": text,
             "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "role": "query"} for i, text in enumerate(texts)]


class Runner:
    def __init__(self, bundle: dict, encoder, args):
        self.documents = bundle["documents"]
        self.ids = [doc["input_id"] for doc in self.documents]
        self.texts = [doc["text"] for doc in self.documents]
        self.bm25 = BM25(self.texts)
        self.encoder = encoder
        self.args = args
        self.vectors = encoder.encode(self.texts, role="passage") if encoder else None
        if self.vectors is not None and not np.isfinite(self.vectors).all():
            raise ValueError("Nonfinite document embeddings are not valid retrieval inputs.")
        self.query_vectors = {}
        self.positions = {}
        for row, doc_id in enumerate(self.ids):
            song_id, block = doc_id.split("::")
            self.positions.setdefault(song_id, []).append((int(block[1:]), row))
        for rows in self.positions.values():
            rows.sort()
        self.graph = self.build_graph()

    def prepare_queries(self, annotated: list[dict]):
        if self.encoder is None:
            return
        # All possible bounded reformulations are derived from questions only.
        # Pre-encode for every arm, preventing an earlier arm warming the cache
        # from appearing slower than a later contextual arm in runtime timing.
        texts = list(dict.fromkeys(text for query in annotated
                                  for text in (query["query"], expand_query(query["query"], 1),
                                               expand_query(query["query"], 2))))
        vectors = self.encoder.encode(texts, role="query")
        self.query_vectors.update(zip(texts, vectors))

    def retrieve(self, query: str, method: str, limit: int | None = None) -> list[int]:
        limit = limit or self.args.candidates
        lexical = ranking(self.bm25.scores(query), limit)
        if method == "bm25":
            return lexical
        if self.encoder is None:
            raise ValueError("Dense and hybrid retrieval require a local or verified cached remote encoder.")
        if query not in self.query_vectors:
            self.query_vectors[query] = self.encoder.encode([query], role="query")[0]
        # Explicit float64 summation avoids spurious Accelerate BLAS warnings
        # observed on finite, unit-norm float32 arrays on this macOS runtime.
        scores = np.einsum("ij,j->i", self.vectors.astype(np.float64),
                           np.asarray(self.query_vectors[query], dtype=np.float64), optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite similarity scores; retrieval was not evaluated.")
        dense = ranking(scores, limit)
        return dense if method == "dense" else rrf(lexical, dense, limit)

    def context(self, ranked: list[int], neighbors: bool) -> tuple[list[int], int]:
        # Equal maximum whitespace-word budget for all arms. Whole blocks only;
        # oversize blocks are skipped consistently, never silently truncated.
        selected, used = [], 0
        for row in ranked:
            candidates = [row]
            if neighbors:
                song_id, block = self.ids[row].split("::")
                number = int(block[1:])
                candidates += [r for n, r in self.positions[song_id] if abs(n - number) == 1]
            for candidate in candidates:
                size = len(self.texts[candidate].split())
                if candidate in selected or used + size > self.args.context_words:
                    continue
                selected.append(candidate)
                used += size
                if len(selected) >= self.args.k:
                    break
            if len(selected) >= self.args.k:
                break
        return selected, used

    def build_graph(self):
        def route(state: State):
            return dict(search_query=state["query"], attempt=0, ranked=[], all_ranked=[],
                        trace=["route: frozen corpus, no web or title/artist metadata"],
                        retrieved_slots=0, search_calls=0)

        def retrieve(state: State):
            # The total budget counts candidate slots, including repeated IDs.
            available = self.args.total_retrieval_slots - state["retrieved_slots"]
            # One-pass baseline spends the full budget in one broad retrieval;
            # the bounded controller may distribute the same maximum over up to
            # three narrower searches. Actual spend is always reported.
            width = self.args.candidates if state["narrow_search"] else self.args.total_retrieval_slots
            ranked = self.retrieve(state["search_query"], state["method"], width)[:max(0, min(width, available))]
            union = list(dict.fromkeys([*ranked, *state["all_ranked"]]))
            return dict(ranked=ranked, all_ranked=union,
                        retrieved_slots=state["retrieved_slots"] + len(ranked),
                        search_calls=state["search_calls"] + 1,
                        trace=[*state["trace"], f"retrieve:{state['method']}:attempt={state['attempt']}:slots={len(ranked)}"])

        def assess(state: State):
            coverage = [self.bm25.lexical_support(state["search_query"], row) for row in state["ranked"][:self.args.k]]
            sufficient = any(matches >= 2 and score >= self.args.support_coverage for matches, score in coverage)
            return dict(sufficient=sufficient, trace=[*state["trace"], f"assess:heuristic_support={sufficient}"])

        def after_assess(state: State):
            if state["sufficient"]:
                return "context"
            if (state["rewrite_enabled"] and state["attempt"] < self.args.max_rewrites
                    and state["retrieved_slots"] < self.args.total_retrieval_slots):
                return "rewrite"
            return "abstain"

        def rewrite(state: State):
            attempt = state["attempt"] + 1
            return dict(attempt=attempt, search_query=expand_query(state["query"], attempt),
                        trace=[*state["trace"], f"rewrite:generic_bilingual_expansion:{attempt}"])

        def context(state: State):
            rows, words = self.context(state["all_ranked"], neighbors=state["neighbors_enabled"])
            return dict(context_rows=rows, context_words=words,
                        trace=[*state["trace"], f"context:words={words}:neighbor_expansion={state['neighbors_enabled']}"])

        def answer(state: State):
            ids = [self.ids[row] for row in state["context_rows"]]
            # No model, gold answers, lyrical phrases or inferred emotions are
            # consulted. This is a neutral evidence-package template only.
            text = (f"Se recuperaron {len(ids)} bloques candidatos para revisión. La recuperación no prueba la interpretación."
                    if state["language"] == "es" else
                    f"Retrieved {len(ids)} candidate blocks for review. Retrieval does not establish the interpretation.")
            return dict(answer=text, citations=ids, abstained=False,
                        trace=[*state["trace"], "answer:deterministic_evidence_package_only"])

        def abstain(state: State):
            text = ("No hay suficiente apoyo según el criterio heurístico; se requiere revisión."
                    if state["language"] == "es" else
                    "Insufficient support under the heuristic criterion; review is required.")
            return dict(answer=text, citations=[], abstained=True, context_rows=[], context_words=0,
                        trace=[*state["trace"], "abstain:heuristic_insufficiency"])

        def citation_validation(state: State):
            context_ids = {self.ids[row] for row in state["context_rows"]}
            valid = all(doc_id in context_ids for doc_id in state["citations"])
            return dict(citation_ids_valid=valid,
                        trace=[*state["trace"], "citation_validation:ID_resolution_only_not_entailment"])

        graph = StateGraph(State)
        for name, function in (("route", route), ("retrieve", retrieve), ("assess", assess),
                               ("rewrite", rewrite), ("context", context), ("answer", answer),
                               ("abstain", abstain), ("citation_validation", citation_validation)):
            graph.add_node(name, function)
        graph.add_edge(START, "route")
        graph.add_edge("route", "retrieve")
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges("assess", after_assess, {name: name for name in ("rewrite", "context", "abstain")})
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("context", "answer")
        graph.add_edge("answer", "citation_validation")
        graph.add_edge("abstain", "citation_validation")
        graph.add_edge("citation_validation", END)
        return graph.compile()

    def run(self, query: dict, method: str, agentic: bool, *, arm: str | None = None,
            rewrite_enabled: bool | None = None, neighbors_enabled: bool | None = None,
            narrow_search: bool | None = None) -> dict:
        started = time.perf_counter()
        state = self.graph.invoke(dict(query=query["query"], language=query["language"],
                                       method=method, agentic=agentic,
                                       rewrite_enabled=agentic if rewrite_enabled is None else rewrite_enabled,
                                       neighbors_enabled=agentic if neighbors_enabled is None else neighbors_enabled,
                                       narrow_search=agentic if narrow_search is None else narrow_search),
                                  {"recursion_limit": 40})
        ranked_ids = [self.ids[row] for row in state["all_ranked"]]
        context_ids = [self.ids[row] for row in state["context_rows"]]
        result = dict(query_id=query["query_id"], family=query["family"], split=query["split"],
                      language=query["language"], method=method,
                      arm=arm or ("bounded_contextual" if agentic else "one_pass"),
                      rewrite_enabled=state["rewrite_enabled"], neighbors_enabled=state["neighbors_enabled"],
                      narrow_search=state["narrow_search"],
                      retrieved_ids=ranked_ids[:self.args.k], context_ids=context_ids,
                      context_words=state["context_words"], maximum_context_words=self.args.context_words,
                      retrieval_calls=state["search_calls"], candidate_slots=state["retrieved_slots"],
                      rewrites=state["attempt"], abstained=state["abstained"],
                      citation_ids_valid=state["citation_ids_valid"],
                      latency_seconds=time.perf_counter() - started, trace=state["trace"],
                      answer=state["answer"], cost_usd=0.0, api_calls=0)
        if query["answerable"]:
            score = metrics(ranked_ids, query["gold_document_ids"], self.args.k)
            result.update({f"{name}_at_{self.args.k}": value for name, value in score.items()})
            gold = set(query["gold_document_ids"])
            result["context_support_recall"] = len(gold.intersection(context_ids)) / len(gold)
            result["all_annotated_support_in_context"] = gold.issubset(context_ids)
        else:
            result.update({f"{name}_at_{self.args.k}": None for name in ("recall", "mrr", "ndcg")})
            result["negative_safe_abstention"] = state["abstained"]
        return result


def summarize(results: list[dict], annotations: list[dict], k: int) -> list[dict]:
    answerable = {q["query_id"]: q["answerable"] for q in annotations}
    groups = {}
    for row in results:
        key = (row["method"], row["arm"], row["split"], row["language"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (method, arm, split, language), rows in sorted(groups.items()):
        positive = [r for r in rows if answerable[r["query_id"]]]
        negative = [r for r in rows if not answerable[r["query_id"]]]
        average = lambda key, subset: float(np.mean([r[key] for r in subset])) if subset else None
        summaries.append(dict(method=method, arm=arm, split=split, language=language,
                              positive_queries=len(positive), negative_queries=len(negative),
                              **{f"{metric}_at_{k}": average(f"{metric}_at_{k}", positive) for metric in ("recall", "mrr", "ndcg")},
                              context_support_recall=average("context_support_recall", positive),
                              all_support_context_rate=average("all_annotated_support_in_context", positive),
                              positive_abstention_rate=average("abstained", positive),
                              negative_safe_abstention_rate=average("negative_safe_abstention", negative),
                              average_calls=average("retrieval_calls", rows),
                              average_context_words=average("context_words", rows)))
    return summaries


def self_test():
    assert tokens("No tengo miedo") == ["no", "tengo", "miedo"]
    bm25 = BM25(["fear darkness", "fear darkness darkness", "privacy watching"])
    assert ranking(bm25.scores("privacy"), 1) == [2]
    assert np.allclose(bm25.scores("fear fear"), bm25.scores("fear"))
    assert metrics(["a", "z", "b"], ["a", "b"], 3)["recall"] == 1
    assert metrics(["z", "a"], ["a"], 2)["mrr"] == 0.5
    assert metrics(["a"], [], 2)["recall"] is None
    assert rrf([0, 1], [1, 0], 2) == [0, 1]
    families = {}
    for q in questions():
        families.setdefault(q["family"], set()).add(q["split"])
    assert all(len(splits) == 1 for splits in families.values())
    args = argparse.Namespace(candidates=2, k=2, context_words=8, total_retrieval_slots=6,
                              max_rewrites=2, support_coverage=0.20)
    bundle = {"documents": [dict(input_id=f"synthetic::s{i:03d}", text=text)
                              for i, text in enumerate(["privacy watching fear", "safe escape courage", "night monsters sleep"])]}
    runner = Runner(bundle, None, args)
    unknown = dict(query_id="unknown", family="unknown", split="test", language="en",
                   query="mitochondrial vaccine identifier", answerable=False, gold_document_ids=[])
    result = runner.run(unknown, "bm25", True)
    assert result["abstained"] and result["rewrites"] == 2
    assert result["candidate_slots"] <= 6 and result["retrieval_calls"] == 3
    known = dict(query_id="known", family="known", split="test", language="en",
                 query="privacy watching", answerable=True, gold_document_ids=["synthetic::s000"])
    result = runner.run(known, "bm25", True)
    assert not result["abstained"] and result["citation_ids_valid"]
    assert result["context_words"] <= 8
    assert len(result["context_ids"]) <= args.k
    # Altering evaluation gold cannot affect controller decision or output.
    changed = runner.run({**known, "gold_document_ids": ["synthetic::s002"]}, "bm25", True)
    assert result["answer"] == changed["answer"] and result["context_ids"] == changed["context_ids"]
    assert all(text not in json.dumps(result) for text in runner.texts)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for doc in bundle["documents"]:
            doc["text_sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
        query_text = "privacy watching"
        np.savez(root / "vectors.npz", documents=np.eye(3), queries=np.array([[1., 0., 0.]]))
        metadata = dict(model_id="text-embedding-3-small", document_ids=[d["input_id"] for d in bundle["documents"]],
                        document_hashes=[d["text_sha256"] for d in bundle["documents"]],
                        query_hashes=[hashlib.sha256(query_text.encode()).hexdigest()])
        (root / "metadata.json").write_text(json.dumps(metadata))
        remote = RemoteCachedEncoder("openai/text-embedding-3-small", bundle, root / "vectors.npz", root / "metadata.json")
        assert np.allclose(remote.encode([query_text], role="query"), [[1., 0., 0.]])
        assert np.allclose(remote.encode([d["text"] for d in bundle["documents"]]), np.eye(3))
        try:
            remote.encode(["uncached query"], role="query")
            raise AssertionError("Missing remote query did not fail closed")
        except ValueError:
            pass
        corrupted = {"documents": [{**d, "text_sha256": "invalid"} for d in bundle["documents"]]}
        try:
            RemoteCachedEncoder("openai/text-embedding-3-small", corrupted, root / "vectors.npz", root / "metadata.json")
            raise AssertionError("Remote corpus identity check failed")
        except ValueError:
            pass
    print("RAG self-test passed: BM25, metrics, family splits, budgets, retries, gold independence, privacy, cached-remote identity and missing-query guards.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PRIVATE / "embedding-input.json")
    parser.add_argument("--output", type=Path, default=PUBLIC)
    parser.add_argument("--private-cache", type=Path, default=PRIVATE)
    parser.add_argument("--model-cache", type=Path, default=PRIVATE / "model-cache")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--remote-npz", type=Path)
    parser.add_argument("--remote-metadata", type=Path)
    parser.add_argument("--export-queries", type=Path, help="Export original and both bounded rewrites for separate hosted embedding acquisition; no RAG execution.")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Posthoc fixed-width 2x2 rewrite/neighbor decomposition; saves a separate experiment.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--total-retrieval-slots", type=int, default=30)
    parser.add_argument("--max-rewrites", type=int, default=2)
    parser.add_argument("--context-words", type=int, default=320)
    parser.add_argument("--support-coverage", type=float, default=0.20)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.export_queries:
        if not args.export_queries.resolve().is_relative_to(args.private_cache.resolve()):
            raise ValueError("Query embedding export must remain in the private cache.")
        payload = {"queries": query_embedding_inputs(), "benchmark_identity": identity(questions()),
                   "construction": "Original 32 questions plus up to two generic bounded rewrites, exact-text deduplicated; no answer/gold IDs or lyric text."}
        analyze.write_json(args.export_queries, payload)
        print(json.dumps(dict(status="queries_exported", unique_queries=len(payload["queries"]), path=str(args.export_queries))), flush=True)
        return
    if bool(args.remote_npz) != bool(args.remote_metadata):
        raise ValueError("Both remote NPZ and metadata are required.")
    if args.model.startswith("openai/") and not args.remote_npz:
        raise ValueError("Hosted models require explicitly supplied offline caches; RAG does not call APIs.")
    if args.remote_npz and any(not p.resolve().is_relative_to(args.private_cache.resolve()) for p in (args.remote_npz, args.remote_metadata)):
        raise ValueError("Remote caches must remain inside private storage.")
    if not args.bundle.resolve().is_relative_to(args.private_cache.resolve()):
        raise ValueError("The input bundle containing lyrics must be inside the private cache.")
    if args.output.resolve().is_relative_to(args.private_cache.resolve()):
        raise ValueError("Public results must not be inside the private cache.")
    if args.ablation and args.output.resolve() in (PUBLIC.resolve(), PUBLIC.with_name("rag-e5").resolve()):
        raise ValueError("Ablations must use a separate --output to preserve frozen primary runs.")
    if args.max_rewrites < 0 or args.max_rewrites > 2:
        raise ValueError("Rewrites must be between zero and two.")
    if args.k < 1 or args.candidates < args.k or args.context_words < 1 or args.total_retrieval_slots < args.candidates:
        raise ValueError("Invalid retrieval or context budget.")
    bundle = json.loads(args.bundle.read_text())
    annotated = questions()
    available = {doc["input_id"] for doc in bundle["documents"]}
    for query in annotated:
        if not set(query["gold_document_ids"]).issubset(available):
            raise ValueError(f"Annotation references absent document IDs: {query['query_id']}")
    analyze.write_json(args.private_cache / "rag-annotations.json", annotated)
    public_questions = [{key: value for key, value in query.items() if key != "gold_document_ids"} for query in annotated]
    analyze.write_json(args.output / "questions.json", public_questions)
    analyze.write_json(args.output / "protocol.json", dict(
        status="Executed offline: retrieval and real LangGraph, rule controller, no generative LLM.",
        question_authorship="Original EN/ES assistant questions created by reading private lyric blocks; one provisional annotation process, no independent human adjudication.",
        annotation_scope="Selected support IDs are diagnostic qrels, not exhaustive annotations of all semantically relevant corpus passages.",
        split="Both language variants of each family share dev/test. Fixed configuration; no tuning on test. Artists/songs overlap between query families: this is not an unseen-song evaluation.",
        private_inputs_hash=identity([(d['input_id'], d['text_sha256']) for d in bundle['documents']]),
        benchmark_hash=identity(annotated), model=args.model if not args.bm25_only else None,
        experiment="posthoc_fixed_width_rewrite_neighbor_2x2" if args.ablation else "frozen_primary_compound_comparison",
        posthoc_decomposition=bool(args.ablation),
        ablation_scope="Same original questions, qrels, thresholds, models, per-search width 10, maximum 30 returned candidates, 320-word/5-block context. Toggle only rewrite and neighbors; actual search count can differ. This decomposes an observed result, not independent held-out confirmation." if args.ablation else None,
        index_text="Unique original blocks only; no titles, artists, years, topic labels, selection hypotheses, or gold answers in retrieval text.",
        bm25=dict(k1=1.5, b=0.75, idf="log(1 + (N-df+0.5)/(df+0.5))", preprocessing="Case/accent folding, bilingual function-word list; negation retained."),
        hybrid="Equal-weight reciprocal rank fusion; constant=60; no learned reranker.",
        controller="Rule heuristic: any top-k block overlaps >=2 content terms and covers >=support_coverage of query IDF mass. Not an entailment test. Generic bilingual search expansion, maximum two rewrites.",
        support_coverage=args.support_coverage, k=args.k, contextual_candidate_slots_per_search=args.candidates,
        one_pass_candidate_slots=args.candidates if args.ablation else args.total_retrieval_slots,
        maximum_total_candidate_slots=args.total_retrieval_slots, maximum_rewrites=args.max_rewrites,
        context_budget=dict(maximum_whitespace_words=args.context_words, whole_blocks=True, maximum_blocks=args.k,
                            note="All arms have identical maximum context budget; used budget can differ. Adjacent block numbers within the same song only; unique-block index may omit repeats."),
        generation="Deterministic evidence-package template. No literary answer quality, faithfulness, provider-generation comparison, or autonomous planning was tested.",
        citation_validation="Identifier resolution inside supplied context only; does not test claim entailment.",
        latency="Warm offline graph latency: question vectors for original and both permitted rewrites are pre-encoded identically before all arms. Index/query preparation times are recorded separately. Not provider/API latency.",
        negatives="Three easy out-of-domain families and two adversarial in-domain missing-detail families. The latter combine relevant lyrical motifs with unsupported identifying or clinical details; a lexical support rule may accept these. One analyst's absence annotation, not calibrated safety.",
        sources=["https://docs.langchain.com/oss/python/langgraph/overview", "https://docs.langchain.com/oss/python/langgraph/agentic-rag", "https://www.anthropic.com/engineering/contextual-retrieval", "https://aclanthology.org/2026.findings-acl.338/"],
        packages={name: importlib.metadata.version(name) for name in ("langgraph", "numpy", "sentence-transformers")},
        cost_usd=0.0, api_calls=0, privacy="No lyric text, quoted fragments, input headings, embeddings, or gold answers are exported."))
    encoder = None
    if not args.bm25_only:
        if args.remote_npz:
            encoder = RemoteCachedEncoder(args.model, bundle, args.remote_npz, args.remote_metadata)
        else:
            encoder_args = argparse.Namespace(model_revision=[], offline=True, allow_download=False, device="cpu",
                                              model_cache=args.model_cache, private_cache=args.private_cache, batch_size=32)
            encoder = analyze.Encoder(args.model, encoder_args)
    started = time.perf_counter()
    runner = Runner(bundle, encoder, args)
    preparation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    runner.prepare_queries(annotated)
    query_preparation_seconds = time.perf_counter() - started
    analyze.write_json(args.output / "timing.json", dict(index_preparation_seconds=preparation_seconds,
                       query_preparation_seconds=query_preparation_seconds,
                       run_timing="Warm graph calls with original and bounded rewrite query vectors precomputed."))
    methods = ["bm25"] if args.bm25_only else ["bm25", "dense", "hybrid"]
    results = []
    for method in methods:
        if args.ablation:
            for rewrite in (False, True):
                for neighbors in (False, True):
                    arm = f"rewrite_{int(rewrite)}_neighbors_{int(neighbors)}"
                    for query in annotated:
                        results.append(runner.run(query, method, rewrite or neighbors, arm=arm,
                                                  rewrite_enabled=rewrite, neighbors_enabled=neighbors,
                                                  narrow_search=True))
                    print(json.dumps(dict(completed_method=method, ablation_arm=arm, queries=len(annotated))), flush=True)
        else:
            for agentic in (False, True):
                for query in annotated:
                    results.append(runner.run(query, method, agentic))
                print(json.dumps(dict(completed_method=method, contextual=agentic, queries=len(annotated))), flush=True)
    analyze.write_json(args.output / "runs.json", results)
    summaries = summarize(results, annotated, args.k)
    analyze.write_csv(args.output / "summary.csv", summaries)
    analyze.write_json(args.output / "summary.json", summaries)
    if encoder:
        analyze.write_json(args.output / "encoder-manifest.json", encoder.manifest())
    print(json.dumps(dict(runs=len(results), positive_queries=sum(q['answerable'] for q in annotated),
                          negative_queries=sum(not q['answerable'] for q in annotated), output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
