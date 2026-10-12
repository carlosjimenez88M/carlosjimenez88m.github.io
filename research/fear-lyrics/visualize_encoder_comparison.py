"""Compare four executed encoders on the SAME frozen 40-song cohort, offline.

Song neighbours are ranked in the original full-dimensional embedding spaces,
not in PCA or UMAP. Only ranks, public song metadata and aggregate comparisons
are exported. No lyrics, vectors, pairwise cosine matrices or API calls.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fear-encoder-plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.stats import spearmanr

import projections

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIVATE = ROOT / ".research-cache" / "fear-lyrics" / "pilot-comparison"
DEST = HERE / "results" / "encoder-comparison-new"
FIG = HERE / "figures"
SITE = ROOT / "static" / "research" / "fear-lyrics"

MODELS = [
    ("sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2", "MiniLM", "sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"),
    ("intfloat--multilingual-e5-small", "E5 small", "intfloat--multilingual-e5-small"),
    ("openai--text-embedding-3-large", "OpenAI large", "../pilot-40/openai-large"),
    ("openai--text-embedding-3-small", "OpenAI small", "openai-small"),
]
COLORS = ["#4063a6", "#269c89", "#8855b5", "#e2823d"]
# Established, already anchored examples; not selected by neighbour results.
CASE_TITLES = ["Somebody's Watching Me", "Rayando el Sol", "Out of the Woods", "Aperture"]


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, caption):
    fig.text(.03, .015, caption, fontsize=9, color="#4b5563", ha="left", va="bottom", wrap=True)
    for suffix in ("svg", "png"):
        path = FIG / f"{name}.{suffix}"
        fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
        shutil.copyfile(path, SITE / path.name)
    plt.close(fig)


def load_verified(slug, short_name, bundle_name, manifest):
    directory = HERE / "results" / slug
    mm = projections.read_json(directory / "model_manifest.json")
    receipt = projections.read_json(directory / "projections.json")["receipt"]
    meta_path = PRIVATE / f"{bundle_name}.json"
    vector_path = PRIVATE / f"{bundle_name}.npz"
    metadata = projections.read_json(meta_path)
    if not projections.model_matches(metadata.get("model_id"), mm["encoder"]["model_id"]):
        raise ValueError("The cached model differs from its executed manifest")
    with np.load(vector_path, allow_pickle=False) as bundle:
        x = np.asarray(bundle["documents"], dtype=np.float64)
    ids, hashes = metadata["document_ids"], metadata["document_hashes"]
    assignments = projections.unique_assignments(directory / "section_assignments.csv")
    if len(ids) != 386 or set(ids) != set(assignments) or len(set(ids)) != len(ids):
        raise ValueError("The frozen, unique, 386-block cohort is required")
    if x.shape != (386, mm["encoder"]["dimensions"]) or len(hashes) != 386 or not np.isfinite(x).all():
        raise ValueError("Private vectors and manifest dimensions disagree")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A zero vector cannot define cosine neighbours")
    x = x / norms
    corpus_hash = projections.digest(list(zip(ids, hashes)))
    order_hash = projections.digest(ids)
    normalized_hash = hashlib.sha256(x.astype("<f8").tobytes(order="C")).hexdigest()
    for key, actual in [("corpus_sha256", corpus_hash), ("document_order_sha256", order_hash), ("embedding_sha256", normalized_hash)]:
        if receipt.get(key) != actual:
            raise ValueError(f"The existing PCA receipt and the new comparison differ: {key}")
    if receipt["embedding_file_sha256"] != projections.file_digest(vector_path):
        raise ValueError("The executed private embedding archive has changed")
    song_indices = defaultdict(list)
    for i, document_id in enumerate(ids):
        song_indices[assignments[document_id]["song_id"]].append(i)
    songs = sorted(song_indices)
    if len(songs) != 40 or any(song_id not in manifest for song_id in songs):
        raise ValueError("The comparison must resolve exactly 40 public songs")
    centroids = np.array([x[song_indices[song_id]].mean(axis=0) for song_id in songs])
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(centroid_norms < 1e-12):
        raise ValueError("A zero song centroid cannot be compared")
    centroids /= centroid_norms
    # Elementwise reductions avoid environment BLAS floating-status warnings.
    cosine = np.array([[np.sum(a * b) for b in centroids] for a in centroids])
    if not np.isfinite(cosine).all():
        raise ValueError("The cosine matrix contains a nonfinite value")
    neighbours = {}
    all_ranks = {}
    for i, song_id in enumerate(songs):
        order = sorted([j for j in range(len(songs)) if j != i], key=lambda j: (-cosine[i, j], songs[j]))
        neighbours[song_id] = [songs[j] for j in order]
        all_ranks[song_id] = {songs[j]: rank + 1 for rank, j in enumerate(order)}
    return dict(slug=slug, label=short_name, song_ids=songs, neighbours=neighbours, ranks=all_ranks,
                receipt=dict(model_id=mm["encoder"]["model_id"], dimensions=x.shape[1], input_songs=40,
                    unique_blocks=386, corpus_sha256=corpus_hash, document_order_sha256=order_hash,
                    embedding_file_sha256=projections.file_digest(vector_path),
                    normalized_document_vectors_sha256=normalized_hash,
                    source_model_manifest_sha256=projections.file_digest(directory / "model_manifest.json"),
                    source_projection_receipt_sha256=projections.file_digest(directory / "projections.json"),
                    input_metadata_sha256=projections.file_digest(meta_path),
                    block_counts_by_song={song_id: len(song_indices[song_id]) for song_id in songs}))


def figure_overview(models, comparisons, aggregates):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), gridspec_kw={"width_ratios": [1, 1.05]})
    matrix = np.eye(4)
    by_pair = {(r["model_a"], r["model_b"]): r for r in comparisons if r["k"] == 5 and r["candidate_policy"] == "all_39_other_songs"}
    for i, j in itertools.combinations(range(4), 2):
        matrix[i, j] = matrix[j, i] = by_pair[(models[i]["slug"], models[j]["slug"])]["mean_overlap_fraction"]
    axes[0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    labels = [m["label"] for m in models]
    axes[0].set_xticks(range(4), labels, rotation=25, ha="right")
    axes[0].set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{matrix[i, j]:.1%}" if i != j else "100%", ha="center", va="center", fontsize=15,
                         weight="bold", color="white" if matrix[i, j] > .55 else "#12304a")
    axes[0].set_title("¿Cuántos de los cinco vecinos coinciden?", fontsize=12, loc="left", pad=20)
    y = np.arange(4)
    values = [r["same_language_fraction_k5"] for r in aggregates]
    axes[1].barh(y, values, color=COLORS, height=.52)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.07)
    axes[1].set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axes[1].axvline(19 / 39, color="#6b7280", ls="--", lw=1.2)
    for i, value in enumerate(values):
        axes[1].text(value + .015, i, f"{value:.1%}", va="center", fontsize=13, weight="bold")
    axes[1].set_title("¿Cuánto organiza la cercanía el idioma?", fontsize=12, loc="left", pad=20)
    axes[1].set_xlabel("Vecinos con el mismo idioma que la canción consultada")
    axes[1].text(.02, -.28, "Línea: 19/39 disponibles con el mismo idioma.\nReferencia de muestreo uniforme; no es un criterio de calidad.",
                 transform=axes[1].transAxes, fontsize=9, color="#4b5563")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Las mismas canciones, cuatro formas de ordenar sus relaciones", x=.03, ha="left", fontsize=17, weight="bold")
    fig.subplots_adjust(top=.8, bottom=.26, wspace=.5)
    save(fig, "encoder-comparison-neighborhoods",
         "40 consultas × cinco vecinos. Centroides de bloques únicos; coseno en el espacio completo de cada modelo.\n"
         "El porcentaje de acuerdo compara identidades de vecinos, nunca magnitudes de coseno entre encoders. Cercanía no equivale a miedo compartido.")


def figure_sensitivity(models, comparisons):
    fig, ax = plt.subplots(figsize=(11.8, 6.1))
    for index, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        data = [r for r in comparisons if (r["model_a"], r["model_b"]) == (models[i]["slug"], models[j]["slug"]) and r["candidate_policy"] == "all_39_other_songs"]
        xs = [r["k"] for r in data]
        ys = [r["mean_overlap_fraction"] for r in data]
        ax.plot(xs, ys, marker="o", lw=2.1, label=f'{models[i]["label"]} ↔ {models[j]["label"]}')
    ax.plot([3, 5, 10], [3 / 39, 5 / 39, 10 / 39], color="#7b8490", ls="--", lw=1.3, label="Independencia uniforme: k/39")
    ax.set_xticks([3, 5, 10])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Número de vecinos consultados (k)")
    ax.set_ylabel("Vecinos compartidos / k, media de las 40 canciones")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    fig.suptitle("El acuerdo cambia al ampliar el vecindario", x=.03, ha="left", fontsize=17, weight="bold")
    fig.subplots_adjust(top=.87, bottom=.21, right=.66, left=.085)
    save(fig, "encoder-comparison-k-sensitivity",
         "Los 40 títulos y sus 386 bloques son idénticos en los cuatro encoders. La referencia uniforme es descriptiva.\n"
         "Los conjuntos dependen del corpus candidato: el acuerdo no mide exactitud, intensidad emocional ni evolución histórica.")


def figure_language_control(models, comparisons):
    fig, ax = plt.subplots(figsize=(12.7, 6.2))
    labels, overall, controlled = [], [], []
    for a, b in itertools.combinations(models, 2):
        data = [r for r in comparisons if r["k"] == 5 and (r["model_a"], r["model_b"]) == (a["slug"], b["slug"])]
        by_policy = {r["candidate_policy"]: r for r in data}
        labels.append(f'{a["label"]} ↔ {b["label"]}')
        overall.append(by_policy["all_39_other_songs"]["mean_overlap_fraction"])
        controlled.append(by_policy["same_language_19_other_songs"]["mean_overlap_fraction"])
    y = np.arange(len(labels))
    ax.barh(y - .17, overall, .29, color="#7891a8", label="39 alternativas: EN + ES")
    ax.barh(y + .17, controlled, .29, color="#285b75", label="19 alternativas: mismo idioma")
    for yy, a, b in zip(y, overall, controlled):
        ax.text(a + .014, yy - .17, f"{a:.1%}", fontsize=10, va="center", color="#43546a")
        ax.text(b + .014, yy + .17, f"{b:.1%}", fontsize=10, va="center", color="#163d50", weight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Cinco vecinos compartidos / cinco, media de 40 consultas")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), frameon=False, ncol=2)
    fig.suptitle("Comparar sentido exige controlar primero el idioma", x=.03, ha="left", fontsize=17, weight="bold")
    fig.subplots_adjust(top=.78, bottom=.22, left=.29, right=.94)
    save(fig, "encoder-comparison-language-control",
         "Las mismas 40 consultas, los mismos centroides. Se filtra y reordena el ranking, sin traducción ni nuevos embeddings.\n"
         "Al reducir alternativas, el acuerdo esperado por independencia uniforme también sube: 12,8 % → 26,3 %. Ninguna barra mide exactitud emocional.")


def figure_cases(models, cases, metadata):
    fig, axes = plt.subplots(len(cases), 1, figsize=(14, 13.6))
    fig.suptitle("Una misma canción: ¿qué otras canciones recupera cada encoder?", x=.025, ha="left", fontsize=18, weight="bold")
    for ax, case in zip(axes, cases):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.2)
        ax.axis("off")
        query = metadata[case]
        ax.text(0, 1.29, f'{query["title"]} · {query["artist"]} · {query["year"]}', fontsize=13.8, weight="bold", color="#142438")
        for i, model in enumerate(models):
            box = FancyBboxPatch((i + .015, .025), .935, 1.14, boxstyle="round,pad=0.012,rounding_size=0.035",
                                 facecolor=COLORS[i] + "10", edgecolor=COLORS[i], lw=1.15)
            ax.add_patch(box)
            ax.text(i + .075, 1.075, model["label"], fontsize=12, color=COLORS[i], weight="bold")
            y = .90
            for rank, neighbour in enumerate(model["neighbours"][case][:3], 1):
                song = metadata[neighbour]
                title_lines = textwrap.wrap(f'{rank}. {song["title"]}', width=24, subsequent_indent="   ")
                ax.text(i + .075, y, "\n".join(title_lines), fontsize=10.3, va="top", weight="bold", color="#1f2937")
                artist = textwrap.shorten(song["artist"], width=18, placeholder="…")
                label_y = y - (.105 * len(title_lines) if len(title_lines) > 1 else .085)
                ax.text(i + .115, label_y, f'{artist} · {song["year"]} · {song["language"].upper()}', fontsize=8.3,
                        va="top", color="#526071")
                y -= .30
    fig.subplots_adjust(top=.9, bottom=.075, hspace=.5)
    save(fig, "encoder-comparison-cases",
         "Los tres vecinos más cercanos se eligen entre las mismas otras 39 canciones. Consulta fija; ranking calculado en vectores completos.\n"
         "Ejemplos fijados antes de inspeccionar vecinos. Promediar bloques elimina su orden narrativo: estas relaciones requieren volver a leer los arcos.")


def fear_label_checks(models, metadata):
    """Descriptive checks against existing automatic labels, never a gold test.

    Rank the SAME 19 same-language alternatives first, take five, then retain
    only edges with resolved labels. The uniform reference conditions on the
    same query and annotated candidate universe, weighted by retained edges.
    """
    path = HERE / "pilot-comparison" / "results" / "arc-panel" / "fieldwise-posthoc" / "song_annotations.json"
    annotations = {r["song_id"]: r for r in projections.read_json(path)}
    songs = models[0]["song_ids"]
    if set(annotations) != set(songs):
        raise ValueError("The automatic panel labels must refer to exactly the matched 40 songs")
    positive = {sid for sid in songs if annotations[sid]["consensus"]["fear_relevance"] in {"explicit", "implied"}}
    known_fear = positive | {sid for sid in songs if annotations[sid]["consensus"]["fear_relevance"] == "absent"}
    topics = {sid: set(annotations[sid]["fear_conditioned_subtopics"]) - {"unresolved", "no_supported_fear_reference"}
              for sid in positive}
    known_topics = {sid for sid, labels in topics.items() if labels}
    specs = [
        ("fear_binary_agreement", known_fear, known_fear, lambda q, n: (q in positive) == (n in positive)),
        ("fear_neighbors_of_fear_queries", positive, known_fear, lambda q, n: n in positive),
        ("fear_conditioned_subtopic_intersection", known_topics, known_topics, lambda q, n: bool(topics[q] & topics[n])),
    ]
    per_query, summary = [], []
    for model in models:
        for metric, queries, eligible_candidates, match in specs:
            rows = []
            for query in sorted(queries):
                candidates = [sid for sid in songs if sid != query and metadata[sid]["language"] == metadata[query]["language"]]
                ranked = [sid for sid in model["neighbours"][query] if sid in candidates][:5]
                retained = [sid for sid in ranked if sid in eligible_candidates]
                baseline_pool = [sid for sid in candidates if sid in eligible_candidates]
                if len(candidates) != 19 or not baseline_pool:
                    raise ValueError("The same-language reference needs 19 alternatives and at least one resolved label")
                rate = sum(match(query, sid) for sid in baseline_pool) / len(baseline_pool)
                row = dict(model_slug=model["slug"], model=model["label"], metric=metric,
                    song_id=query, language=metadata[query]["language"], planned_neighbors=5,
                    same_language_candidates=19, baseline_annotated_candidates=len(baseline_pool),
                    baseline_matching_candidates=sum(match(query, sid) for sid in baseline_pool),
                    retained_labeled_neighbors=len(retained), omitted_neighbors=5 - len(retained),
                    observed_matching_neighbors=sum(match(query, sid) for sid in retained),
                    uniform_conditional_expected_matches=len(retained) * rate,
                    ranked_neighbor_ids="|".join(ranked), retained_neighbor_ids="|".join(retained))
                rows.append(row)
                per_query.append(row)
            denominator = sum(r["retained_labeled_neighbors"] for r in rows)
            observed = sum(r["observed_matching_neighbors"] for r in rows)
            expected = sum(r["uniform_conditional_expected_matches"] for r in rows)
            summary.append(dict(model_slug=model["slug"], model=model["label"], metric=metric,
                eligible_queries=len(queries), queries_with_retained_neighbors=sum(r["retained_labeled_neighbors"] > 0 for r in rows),
                planned_directed_edges=len(queries) * 5, retained_labeled_directed_edges=denominator,
                omitted_directed_edges=len(queries) * 5 - denominator, observed_matching_edges=observed,
                uniform_conditional_expected_matching_edges=expected,
                observed_fraction=observed / denominator if denominator else None,
                uniform_conditional_expected_fraction=expected / denominator if denominator else None,
                observed_minus_conditional_baseline=(observed - expected) / denominator if denominator else None))
    write_csv(DEST / "fear_neighbor_checks.csv", summary)
    write_csv(DEST / "per_query_fear_neighbor_checks.csv", per_query)
    receipt = dict(annotation_sha256=projections.file_digest(path), annotation_scope="Existing fieldwise posthoc automatic panel on the same 40 songs",
        fear_positive_queries=len(positive), fear_resolved_queries=len(known_fear), fear_conditioned_subtopic_resolved_queries=len(known_topics),
        excluded_unresolved_fear=len(songs) - len(known_fear),
        candidate_policy="Rank all 19 other same-language songs; retain first five; mask unresolved labels afterward, not before retrieval",
        uniform_reference="For each query: resolved matching candidates / all resolved candidates within its 19 same-language alternatives. Weight this probability by that query's retained labeled edges.",
        aggregation="Micro-average directed labeled edges, not an equal per-query mean; repeated destinations and shared query songs are dependent",
        primary_k=5, metrics=[s[0] for s in specs],
        binary_collapse="explicit or implied = fear reference; absent = no supported fear reference; unresolved omitted",
        subtopic_match="At least one intersecting fear-conditioned consensus subtopic; both endpoints require a supported fear reference and resolved subtopics",
        limitations=["Automatic correlated reader labels are not independent semantic truth or accuracy gold",
            "Definitions and missing-label masks change the estimand and denominators",
            "Frequent uncertainty_change can raise intersection even when narrative role differs",
            "This is descriptive agreement with this candidate cohort; no population inference or p-value",
            "The original full-dimensional centroid retrieval omits narrative order and speaker roles"],
        api_calls=0)
    write_json(DEST / "fear_label_check_method.json", receipt)
    return summary, receipt


def figure_fear_checks(models, checks):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True)
    specs = [("fear_binary_agreement", "¿Coincide la referencia al miedo?\n33 consultas con etiqueta resuelta"),
             ("fear_conditioned_subtopic_intersection", "¿Coincide algún subtema del miedo?\n22 consultas con miedo y subtema resueltos")]
    for ax, (metric, title) in zip(axes, specs):
        data = [next(r for r in checks if r["metric"] == metric and r["model_slug"] == model["slug"]) for model in models]
        y = np.arange(4)
        ax.barh(y, [r["observed_fraction"] for r in data], height=.45, color=COLORS)
        ax.scatter([r["uniform_conditional_expected_fraction"] for r in data], y,
                   s=100, marker="D", facecolors="white", edgecolors="#243c50", linewidths=1.6, zorder=4)
        for i, row in enumerate(data):
            ax.text(row["observed_fraction"] + .02, i,
                    f'{row["observed_matching_edges"]}/{row["retained_labeled_directed_edges"]}', fontsize=11, va="center", weight="bold")
        ax.set_yticks(y, [model["label"] for model in models])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Fracción entre vecinos con etiquetas utilizables")
        ax.set_title(title, fontsize=12, loc="left", pad=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cercanía y miedo: comprobar la relación después de controlar el idioma", x=.03, ha="left", fontsize=16, weight="bold")
    fig.subplots_adjust(top=.76, bottom=.23, wspace=.46)
    save(fig, "encoder-comparison-fear-checks",
         "Barras: acuerdos entre cinco vecinos del mismo idioma; texto: coincidencias / aristas con etiquetas utilizables.\n"
         "Rombo: referencia uniforme condicionada a candidatos etiquetados y al idioma. Consenso automático, no exactitud; subtemas frecuentes pueden dominar.")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    public_manifest = HERE / "corpus-manifest.json"
    manifest = {r["song_id"]: r for r in projections.read_json(public_manifest)}
    models = [load_verified(*spec, manifest) for spec in MODELS]
    common_hashes = {r["receipt"]["corpus_sha256"] for r in models}
    common_orders = {r["receipt"]["document_order_sha256"] for r in models}
    if len(common_hashes) != 1 or len(common_orders) != 1 or any(m["song_ids"] != models[0]["song_ids"] for m in models):
        raise ValueError("The four models must compare the same document hashes, order and songs")
    songs = models[0]["song_ids"]
    neighbors_rows, comparisons, details, aggregates = [], [], [], []
    for model in models:
        local = []
        for song_id in songs:
            query = manifest[song_id]
            for rank, neighbor_id in enumerate(model["neighbours"][song_id][:10], 1):
                neighbor = manifest[neighbor_id]
                row = dict(model_slug=model["slug"], model=model["label"], song_id=song_id,
                    title=query["title"], artist=query["artist"], year=query["year"], language=query["language"],
                    neighbor_rank=rank, neighbor_song_id=neighbor_id, neighbor_title=neighbor["title"],
                    neighbor_artist=neighbor["artist"], neighbor_year=neighbor["year"], neighbor_language=neighbor["language"],
                    same_language=query["language"] == neighbor["language"], absolute_year_gap=abs(query["year"] - neighbor["year"]))
                neighbors_rows.append(row)
                if rank <= 5:
                    local.append(row)
        aggregates.append(dict(model_slug=model["slug"], model=model["label"], songs=40, directed_k5_edges=200,
            same_language_fraction_k5=float(np.mean([r["same_language"] for r in local])),
            mean_absolute_year_gap_k5=float(np.mean([r["absolute_year_gap"] for r in local])),
            median_absolute_year_gap_k5=float(np.median([r["absolute_year_gap"] for r in local]))))
    for policy in ("all_39_other_songs", "same_language_19_other_songs"):
        ranked = {}
        for model in models:
            ranked[model["slug"]] = {song_id: [other for other in model["neighbours"][song_id]
                if policy == "all_39_other_songs" or manifest[other]["language"] == manifest[song_id]["language"]]
                for song_id in songs}
        for a, b in itertools.combinations(models, 2):
            for k in (3, 5, 10):
                rows = []
                for song_id in songs:
                    set_a = set(ranked[a["slug"]][song_id][:k])
                    set_b = set(ranked[b["slug"]][song_id][:k])
                    overlap = len(set_a & set_b)
                    row = dict(model_a=a["slug"], model_b=b["slug"], candidate_policy=policy,
                               k=k, song_id=song_id, shared_neighbors=overlap,
                               overlap_fraction=overlap / k, jaccard_fraction=overlap / len(set_a | set_b),
                               exact_neighbor_set=set_a == set_b)
                    details.append(row)
                    rows.append(row)
                rank_correlation = []
                for song_id in songs:
                    rank_a = {other: i + 1 for i, other in enumerate(ranked[a["slug"]][song_id])}
                    rank_b = {other: i + 1 for i, other in enumerate(ranked[b["slug"]][song_id])}
                    others = sorted(rank_a)
                    if set(rank_a) != set(rank_b) or len(others) != (39 if policy == "all_39_other_songs" else 19):
                        raise ValueError("The candidate policy does not preserve the matched comparison")
                    rank_correlation.append(float(spearmanr([rank_a[other] for other in others],
                                                           [rank_b[other] for other in others]).statistic))
                comparisons.append(dict(model_a=a["slug"], model_b=b["slug"], candidate_policy=policy,
                    k=k, songs=40, candidates_per_song=len(others),
                    mean_shared_neighbors=float(np.mean([r["shared_neighbors"] for r in rows])),
                    mean_overlap_fraction=float(np.mean([r["overlap_fraction"] for r in rows])),
                    median_overlap_fraction=float(np.median([r["overlap_fraction"] for r in rows])),
                    mean_jaccard_fraction=float(np.mean([r["jaccard_fraction"] for r in rows])),
                    exact_neighbor_sets=sum(r["exact_neighbor_set"] for r in rows),
                    mean_spearman_candidate_ranks=float(np.mean(rank_correlation)),
                    independent_uniform_expected_overlap_fraction=k / len(others)))
    title_lookup = {manifest[song_id]["title"].casefold().replace("’", "'"): song_id for song_id in songs}
    cases = [title_lookup[title.casefold()] for title in CASE_TITLES]
    write_csv(DEST / "song_neighbors.csv", neighbors_rows)
    write_csv(DEST / "model_neighbor_overlap.csv", comparisons)
    write_csv(DEST / "per_song_neighbor_overlap.csv", details)
    write_csv(DEST / "model_neighbor_diagnostics.csv", aggregates)
    fear_checks, fear_method = fear_label_checks(models, manifest)
    write_json(DEST / "application_cases.json", dict(selection_rule="Four previously anchored examples spanning observed 1983, 1990, 2014 and 2026; chosen before inspecting neighbours",
        cases=[dict(query={k: manifest[song_id][k] for k in ("song_id", "title", "artist", "year", "language")},
                    models=[dict(model_slug=m["slug"], model=m["label"],
                                 neighbors=[{k: manifest[n][k] for k in ("song_id", "title", "artist", "year", "language")}
                                            for n in m["neighbours"][song_id][:3]]) for m in models]) for song_id in cases]))
    figure_overview(models, comparisons, aggregates)
    figure_sensitivity(models, comparisons)
    figure_language_control(models, comparisons)
    figure_cases(models, cases, manifest)
    figure_fear_checks(models, fear_checks)
    receipt = dict(schema_version=1, status="executed", computed_at=datetime.now(timezone.utc).isoformat(),
        study_window="1980 through 2026-10-11; observed frozen cohort years 1982–2026",
        scope="Same matched 40 songs, 386 unique blocks, four encoders; not the nested 120-song/local-only phase",
        model_receipts=[m["receipt"] for m in models], input_songs=40, unique_documents=386,
        corpus_sha256=next(iter(common_hashes)), document_order_sha256=next(iter(common_orders)),
        public_manifest_sha256=projections.file_digest(public_manifest),
        method="L2-normalize original document embeddings; arithmetic mean unique blocks within each song; L2-normalize song centroid; rank cosine within each encoder; exclude self; song ID resolves ties",
        k_values=[3, 5, 10], primary_k=5,
        candidate_policies=["all_39_other_songs", "same_language_19_other_songs"],
        language_control="Re-rank exactly the same 19 same-language alternatives for each query song; cosines, vectors and song centroids are not recomputed",
        metric="|N_k(model A,song) intersect N_k(model B,song)| / k; equal mean across 40 query songs",
        no_cross_model_cosine_comparison=True, original_full_dimension_space=True, pca_or_umap_used_for_neighbors=False,
        topic_ids_are_not_matched_or_compared=True, inference="Descriptive corpus-conditional neighbourhood agreement; no population interval or accuracy claim",
        lost_information="Arithmetic song centroids ignore block order, negation scope and narrative roles; neighbours are candidate relations, not voted emotion or narrative labels",
        automatic_fear_check=fear_method,
        api_calls=0, newly_accounted_usd=0, private_exports="No lyrics, fragments, vectors, pairwise cosine matrices, keys or private filesystem paths",
        software={package: importlib.metadata.version(package) for package in ("numpy", "scipy", "matplotlib")},
        figure_sha256={path.name: projections.file_digest(path) for path in sorted(FIG.glob("encoder-comparison-*"))},
        output_sha256={path.name: projections.file_digest(path) for path in sorted(DEST.glob("*")) if path.name != "receipt.json"})
    write_json(DEST / "receipt.json", receipt)
    print(json.dumps(dict(status="executed", songs=40, unique_blocks=386, models=4, api_calls=0,
                         overlap_k5=[r for r in comparisons if r["k"] == 5], diagnostics=aggregates,
                         fear_checks=fear_checks,
                         figures=["encoder-comparison-neighborhoods", "encoder-comparison-k-sensitivity", "encoder-comparison-language-control", "encoder-comparison-cases", "encoder-comparison-fear-checks"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
