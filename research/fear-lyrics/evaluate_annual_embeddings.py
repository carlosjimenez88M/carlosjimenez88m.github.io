"""Evaluate existing full-space neighbours and BERTopic fits offline.

The new paired readings supply descriptive comparison labels, never a human
gold standard. Frozen 40-song/four-encoder and 120-song/two-local-encoder
cohorts remain separate. Private original vectors are verified and read only;
lyrics, vectors, cosine matrices and private paths are never exported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import shutil

import visualize_encoder_comparison as previous
import projections
import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / "results" / "annual-120" / "embeddings"
FIG = HERE / "figures"
SITE = ROOT / "static" / "research" / "fear-lyrics"
ANNOTATIONS = HERE / "results" / "annual-120" / "annotations.json"
VOTES = HERE / "results" / "annual-120" / "votes.json"
K = 5
TOPICS = ["surveillance_pursuit", "collective_violence", "loss_abandonment",
          "fragility_identity", "uncertainty_change", "relational_coercion", "supernatural"]
TOPIC_LABELS = ["Vigilancia y persecución", "Violencia colectiva", "Abandono y pérdida",
                "Fragilidad e identidad", "Incertidumbre y cambio", "Coerción relacional", "Sobrenatural"]
MATCHED = "matched40"
EXPANDED = "expanded120_local"


def write_json(path, payload):
    previous.write_json(path, payload)


def write_csv(name, rows):
    if not rows:
        raise ValueError(f"Refuse to write an empty analysis table: {name}")
    previous.write_csv(DEST / name, rows)


def load_expanded(slug, label, manifest):
    directory = HERE / "expanded" / "results" / slug
    mm = projections.read_json(directory / "model_manifest.json")
    source_receipt = projections.read_json(directory / "projections.json")["receipt"]
    x, metadata, vector_path, sidecar = projections.locate_embeddings(
        mm["encoder"]["model_id"], slug, ROOT / ".research-cache" / "fear-lyrics", {})
    ids = metadata["document_ids"]
    hashes = metadata["document_hashes"]
    assignments = projections.unique_assignments(directory / "section_assignments.csv")
    if (len(ids) != 1137 or len(set(ids)) != 1137 or set(ids) != set(assignments)
            or len(hashes) != 1137 or x.shape != (1137, mm["encoder"]["dimensions"])
            or not np.isfinite(x).all()):
        raise ValueError("The frozen, unique, 1137-block expanded cohort is required")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A zero original vector cannot define cosine neighbours")
    x = x / norms
    corpus_hash = projections.digest(list(zip(ids, hashes)))
    order_hash = projections.digest(ids)
    normalized_hash = hashlib.sha256(x.astype("<f8").tobytes(order="C")).hexdigest()
    checks = {"corpus_sha256": corpus_hash, "document_order_sha256": order_hash,
              "embedding_sha256": normalized_hash,
              "embedding_file_sha256": projections.file_digest(vector_path),
              "metadata_sha256": projections.file_digest(sidecar),
              "topic_assignments_sha256": projections.file_digest(directory / "section_assignments.csv"),
              "public_manifest_sha256": projections.file_digest(HERE / "corpus-manifest.json")}
    for key, actual in checks.items():
        if source_receipt.get(key) != actual:
            raise ValueError(f"Expanded original cache differs from its executed receipt: {key}")
    song_indices = defaultdict(list)
    for index, document_id in enumerate(ids):
        song_indices[assignments[document_id]["song_id"]].append(index)
    songs = sorted(song_indices)
    if len(songs) != 120 or set(songs) != set(manifest):
        raise ValueError("Expanded retrieval must resolve the same 120 public songs")
    centroids = np.array([x[song_indices[song]].mean(axis=0) for song in songs])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A zero song centroid cannot define neighbours")
    centroids /= norms
    cosine = np.array([[np.sum(a * b) for b in centroids] for a in centroids])
    if not np.isfinite(cosine).all():
        raise ValueError("Expanded cosine ranking contains nonfinite values")
    neighbours = {}
    for index, song in enumerate(songs):
        order = sorted((j for j in range(len(songs)) if j != index),
                       key=lambda j: (-cosine[index, j], songs[j]))
        neighbours[song] = [songs[j] for j in order]
    receipt = dict(model_id=mm["encoder"]["model_id"], dimensions=x.shape[1],
        input_songs=120, unique_blocks=1137, corpus_sha256=corpus_hash,
        document_order_sha256=order_hash, normalized_document_vectors_sha256=normalized_hash,
        embedding_file_sha256=checks["embedding_file_sha256"],
        input_metadata_sha256=checks["metadata_sha256"],
        source_model_manifest_sha256=projections.file_digest(directory / "model_manifest.json"),
        source_projection_receipt_sha256=projections.file_digest(directory / "projections.json"),
        source_topic_assignments_sha256=checks["topic_assignments_sha256"],
        block_counts_by_song={song: len(song_indices[song]) for song in songs})
    return dict(slug=slug, label=label, cohort=EXPANDED, song_ids=songs,
                neighbours=neighbours, assignments=assignments, receipt=receipt)


def annotation_sets(songs, annotations):
    # Use the boolean consensus field: explicit_or_implied is also positive.
    positive = {s for s in songs if annotations[s]["fear_reference_consensus"] is True}
    resolved = positive | {s for s in songs if annotations[s]["fear_reference_consensus"] is False}
    topics = {s: set(annotations[s]["consensus"]["subtopics"]) for s in positive}
    topic_resolved = {s for s, labels in topics.items() if labels}
    return positive, resolved, topics, topic_resolved


def same_language_candidates(model, query, metadata):
    return [s for s in model["song_ids"] if s != query
            and metadata[s]["language"] == metadata[query]["language"]]


def neighbour_checks(models, annotations):
    summaries, per_query, neighbour_rows, diagnostics = [], [], [], []
    query_sets = {}
    for model in models:
        songs = model["song_ids"]
        positive, resolved, topics, topic_resolved = annotation_sets(songs, annotations)
        specs = [
            ("fear_binary_agreement", resolved, resolved,
             lambda q, n: (q in positive) == (n in positive)),
            ("fear_neighbors_of_fear_queries", positive, resolved,
             lambda q, n: n in positive),
            ("fear_conditioned_motif_intersection", topic_resolved, topic_resolved,
             lambda q, n: bool(topics[q] & topics[n])),
        ]
        for metric, queries, eligible_candidates, matches in specs:
            previous_queries = query_sets.setdefault((model["cohort"], metric), sorted(queries))
            if previous_queries != sorted(queries):
                raise ValueError("Models within a cohort must use identical query IDs per metric")
            rows = []
            for query in sorted(queries):
                candidates = same_language_candidates(model, query, annotations)
                candidate_set = set(candidates)
                ranked = [s for s in model["neighbours"][query] if s in candidate_set][:K]
                retained = [s for s in ranked if s in eligible_candidates]
                baseline_pool = [s for s in candidates if s in eligible_candidates]
                if len(candidates) != (19 if model["cohort"] == MATCHED else 59) or len(ranked) != K:
                    raise ValueError("Balanced language candidate counts or top-k size disagree")
                # No synthetic baseline is supplied if a label stratum is empty.
                rate = (sum(matches(query, s) for s in baseline_pool) / len(baseline_pool)
                        if baseline_pool else None)
                if retained and rate is None:
                    raise ValueError("Retained edges cannot lack an eligible baseline universe")
                row = dict(cohort=model["cohort"], model_slug=model["slug"], model=model["label"],
                    metric=metric, song_id=query, language=annotations[query]["language"],
                    planned_neighbors=K, same_language_candidates=len(candidates),
                    baseline_annotated_candidates=len(baseline_pool),
                    baseline_matching_candidates=sum(matches(query, s) for s in baseline_pool),
                    retained_labeled_neighbors=len(retained), omitted_neighbors=K - len(retained),
                    observed_matching_neighbors=sum(matches(query, s) for s in retained),
                    uniform_conditional_expected_matches=len(retained) * rate if rate is not None else 0,
                    uniform_conditional_candidate_fraction=rate,
                    ranked_neighbor_ids="|".join(ranked), retained_neighbor_ids="|".join(retained))
                rows.append(row)
                per_query.append(row)
            for language in ("all_languages", "en", "es"):
                selected = [r for r in rows if language == "all_languages" or r["language"] == language]
                denominator = sum(r["retained_labeled_neighbors"] for r in selected)
                observed = sum(r["observed_matching_neighbors"] for r in selected)
                expected = sum(r["uniform_conditional_expected_matches"] for r in selected)
                summaries.append(dict(cohort=model["cohort"], model_slug=model["slug"],
                    model=model["label"], metric=metric, language=language,
                    cohort_songs=len(songs), eligible_queries=len(selected),
                    query_ids="|".join(r["song_id"] for r in selected),
                    queries_with_retained_neighbors=sum(r["retained_labeled_neighbors"] > 0 for r in selected),
                    planned_directed_edges=len(selected) * K, retained_labeled_directed_edges=denominator,
                    omitted_directed_edges=len(selected) * K - denominator,
                    observed_matching_edges=observed, uniform_conditional_expected_matching_edges=expected,
                    observed_fraction=observed / denominator if denominator else None,
                    uniform_conditional_expected_fraction=expected / denominator if denominator else None,
                    observed_minus_conditional_baseline=(observed - expected) / denominator if denominator else None))
        same_language_top5 = 0
        for query in songs:
            same_language_top5 += sum(annotations[n]["language"] == annotations[query]["language"]
                                     for n in model["neighbours"][query][:K])
            candidates = set(same_language_candidates(model, query, annotations))
            ranked = [s for s in model["neighbours"][query] if s in candidates][:K]
            for rank, neighbour in enumerate(ranked, 1):
                neighbour_rows.append(dict(cohort=model["cohort"], model_slug=model["slug"],
                    model=model["label"], song_id=query, language=annotations[query]["language"],
                    neighbor_rank=rank, neighbor_song_id=neighbour,
                    neighbor_language=annotations[neighbour]["language"],
                    same_language_candidates=len(candidates),
                    query_fear_consensus=annotations[query]["fear_reference_consensus"],
                    neighbor_fear_consensus=annotations[neighbour]["fear_reference_consensus"],
                    query_consensus_motifs="|".join(sorted(annotations[query]["consensus"]["subtopics"])),
                    neighbor_consensus_motifs="|".join(sorted(annotations[neighbour]["consensus"]["subtopics"]))))
        diagnostics.append(dict(cohort=model["cohort"], model_slug=model["slug"], model=model["label"],
            songs=len(songs), directed_k5_edges=len(songs) * K,
            unfiltered_same_language_edges=same_language_top5,
            unfiltered_same_language_fraction=same_language_top5 / (len(songs) * K),
            uniform_same_language_candidate_fraction=(len(songs) / 2 - 1) / (len(songs) - 1),
            fear_consensus_positive_songs=len(positive), fear_binary_resolved_songs=len(resolved),
            fear_unresolved_songs=len(songs) - len(resolved),
            fear_with_at_least_one_consensus_motif_songs=len(topic_resolved)))
    write_csv("neighbor_checks.csv", summaries)
    write_csv("per_query_neighbor_checks.csv", per_query)
    write_csv("same_language_song_neighbors_k5.csv", neighbour_rows)
    write_csv("language_diagnostics.csv", diagnostics)
    return summaries, diagnostics, query_sets


def neighbour_overlap(models, metadata):
    rows = []
    for a, b in itertools.combinations(models, 2):
        if a["cohort"] != b["cohort"]:
            continue
        if a["song_ids"] != b["song_ids"]:
            raise ValueError("Neighbour overlap requires an identical query/candidate cohort")
        for policy in ("all_other_songs", "same_language_other_songs"):
            values = []
            for query in a["song_ids"]:
                candidates = set(a["song_ids"]) - {query}
                if policy == "same_language_other_songs":
                    candidates = set(same_language_candidates(a, query, metadata))
                na = {s for s in [n for n in a["neighbours"][query] if n in candidates][:K]}
                nb = {s for s in [n for n in b["neighbours"][query] if n in candidates][:K]}
                values.append(len(na & nb) / K)
            rows.append(dict(cohort=a["cohort"], model_a=a["slug"], model_b=b["slug"],
                candidate_policy=policy, query_songs=len(values), candidates_per_query=len(candidates),
                k=K, mean_overlap_fraction=float(np.mean(values)),
                independent_uniform_expected_overlap_fraction=K / len(candidates)))
    write_csv("model_neighbor_overlap_k5.csv", rows)
    return rows


def cluster_association(models, annotations):
    """Local cluster membership and supported motif presence, once per song.

    A song may inhabit several clusters. -1 is retained as unassigned noise.
    The zero motif flag means no two-reader-supported presence, not true absence.
    Each expected joint count preserves the cluster's EN/ES song composition.
    """
    associations, memberships = [], []
    global_songs = set(annotations)
    language_songs = {lang: {s for s in global_songs if annotations[s]["language"] == lang}
                      for lang in ("en", "es")}
    motif_songs = {t: {s for s in global_songs if t in annotations[s]["consensus"]["subtopics"]}
                   for t in TOPICS}
    for model in models:
        if model["cohort"] != EXPANDED:
            continue
        cluster_songs = defaultdict(set)
        cluster_blocks = Counter()
        for assignment in model["assignments"].values():
            cluster = int(assignment["topic_id"])
            cluster_songs[cluster].add(assignment["song_id"])
            cluster_blocks[cluster] += 1
        for cluster, songs in sorted(cluster_songs.items()):
            lang_counts = {lang: len(songs & language_songs[lang]) for lang in ("en", "es")}
            for song in sorted(songs):
                memberships.append(dict(model_slug=model["slug"], model=model["label"],
                    local_cluster_id=cluster, unassigned_noise=cluster == -1,
                    song_id=song, language=annotations[song]["language"],
                    fear_consensus=annotations[song]["fear_reference_consensus"],
                    consensus_motifs="|".join(annotations[song]["consensus"]["subtopics"])))
            for motif, label in zip(TOPICS, TOPIC_LABELS):
                positives = motif_songs[motif]
                joint = len(songs & positives)
                expected = sum(lang_counts[lang] * len(positives & language_songs[lang])
                               / len(language_songs[lang]) for lang in ("en", "es"))
                outside_count = len(global_songs - songs)
                outside_positive = len(positives - songs)
                associations.append(dict(cohort=EXPANDED, model_slug=model["slug"], model=model["label"],
                    local_cluster_id=cluster, unassigned_noise=cluster == -1, motif=motif, motif_label=label,
                    unique_blocks_in_cluster=cluster_blocks[cluster],
                    cluster_song_memberships=len(songs), cluster_song_memberships_en=lang_counts["en"],
                    cluster_song_memberships_es=lang_counts["es"],
                    cluster_fear_positive_songs=sum(annotations[s]["fear_reference_consensus"] is True for s in songs),
                    cluster_fear_unresolved_songs=sum(annotations[s]["fear_reference_consensus"] is None for s in songs),
                    consensus_motif_songs_in_cluster=joint,
                    no_consensus_motif_songs_in_cluster=len(songs) - joint,
                    consensus_motif_fraction_in_cluster=joint / len(songs),
                    consensus_motif_songs_outside_cluster=outside_positive,
                    outside_cluster_songs=outside_count,
                    consensus_motif_fraction_outside_cluster=outside_positive / outside_count if outside_count else None,
                    consensus_motif_songs_in_120=len(positives), total_songs=120,
                    language_conditional_expected_joint_songs=expected,
                    language_conditional_expected_fraction=expected / len(songs),
                    observed_minus_language_baseline_fraction=(joint - expected) / len(songs),
                    observed_to_language_expected_ratio=joint / expected if expected else None))
    write_csv("cluster_motif_association.csv", associations)
    write_csv("cluster_song_membership.csv", memberships)
    return associations


def figure_comparison(models, checks):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 8.0))
    positions = np.array([0, 1, 2, 3, 5, 6], dtype=float)
    colors = ["#4063a6", "#269c89", "#8855b5", "#e2823d", "#4063a6", "#269c89"]
    labels = [m["label"] for m in models]
    specs = [("fear_binary_agreement", "Referencia al miedo: acuerdo binario"),
             ("fear_conditioned_motif_intersection", "Motivos: al menos uno compartido")]
    for ax, (metric, title) in zip(axes, specs):
        data = [next(r for r in checks if r["model_slug"] == m["slug"] and r["cohort"] == m["cohort"]
                     and r["metric"] == metric and r["language"] == "all_languages") for m in models]
        observed = [r["observed_fraction"] for r in data]
        expected = [r["uniform_conditional_expected_fraction"] for r in data]
        if any(v is None for v in observed + expected):
            raise ValueError("The comparison figure requires nonempty label denominators")
        ax.barh(positions, observed, color=colors, height=.5)
        ax.scatter(expected, positions, marker="D", s=70, facecolors="white",
                   edgecolors="#22394c", linewidths=1.5, zorder=5,
                   label="Referencia uniforme condicional")
        for y, value, row in zip(positions, observed, data):
            ax.text(value + .012, y, f'{row["observed_matching_edges"]}/{row["retained_labeled_directed_edges"]}',
                    fontsize=10, va="center", weight="bold")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, .25, .5, .75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.set_xlabel("Coincidencias / aristas con etiquetas utilizables")
        ax.set_title(title, fontsize=12.5, loc="left", pad=47)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(4, lw=1, color="#b7c5cf", ls="--")
        matched = data[0]["eligible_queries"]
        expanded = data[4]["eligible_queries"]
        ax.text(.02, 1.05, f"40 canciones · {matched} consultas idénticas entre cuatro encoders", transform=ax.transAxes,
                fontsize=9.5, color="#4b5563")
        ax.text(.02, .355, f"120 canciones · {expanded} consultas entre dos encoders locales", transform=ax.transAxes,
                fontsize=9.5, color="#4b5563", bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fig.suptitle("Vecindarios y lecturas uniformes, después de controlar el idioma", x=.025, ha="left",
                 fontsize=17, weight="bold")
    fig.text(.025, .145,
        "k = 5, ranking por coseno de centroides en el espacio original. Se recuperan primero cinco vecinos del mismo idioma\n"
        "(19 candidatos en 40; 59 en 120), y después se omiten etiquetas sin resolución. Texto: coincidencias/aristas retenidas.\n"
        "Rombo: referencia uniforme dentro de los candidatos etiquetados del mismo idioma, ponderada por aristas retenidas.",
        fontsize=9.5, color="#4b5563", ha="left", va="top")
    fig.text(.025, .015,
        "Consenso de dos lecturas del mismo asistente; no exactitud contra validación humana. Las cohortes y denominadores difieren.\n"
        "Los motivos con frecuencia alta pueden aumentar la intersección. El centroide pierde orden, negación y voces narrativas.",
        fontsize=9.5, color="#4b5563", ha="left", va="bottom")
    fig.subplots_adjust(top=.80, bottom=.24, wspace=.40, left=.11, right=.97)
    name = "annual-embeddings-neighbor-checks"
    for extension in ("svg", "png"):
        path = FIG / f"{name}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        shutil.copyfile(path, SITE / path.name)
    plt.close(fig)
    return name


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    annotations_list = projections.read_json(ANNOTATIONS)
    annotations = {r["song_id"]: r for r in annotations_list}
    if len(annotations) != len(annotations_list) or len(annotations) != 120:
        raise ValueError("Exactly 120 unique uniformly read song annotations are required")
    votes = projections.read_json(VOTES)
    votes_by_song = Counter(r["song_id"] for r in votes)
    if len(votes) != 240 or set(votes_by_song) != set(annotations) or set(votes_by_song.values()) != {2}:
        raise ValueError("Exactly two new votes per song are required")
    for row in annotations_list:
        if not set(row["consensus"]["subtopics"]) <= set(TOPICS):
            raise ValueError("The new motif ontology must have exactly the seven declared categories")
        if row["consensus"]["subtopics"] and row["fear_reference_consensus"] is not True:
            raise ValueError("A consensus motif requires two-reader-supported fear presence")
    matched_manifest_path = HERE / "pilot-40" / "corpus-manifest.json"
    expanded_manifest_path = HERE / "corpus-manifest.json"
    matched_manifest = {r["song_id"]: r for r in projections.read_json(matched_manifest_path)}
    expanded_manifest = {r["song_id"]: r for r in projections.read_json(expanded_manifest_path)}
    if set(expanded_manifest) != set(annotations) or not set(matched_manifest) <= set(annotations):
        raise ValueError("The new labels must cover both existing frozen embedding cohorts")
    for song, meta in expanded_manifest.items():
        if any(meta[k] != annotations[song][k] for k in ("language", "year", "title", "artist")):
            raise ValueError("Annotation metadata differs from the expanded public manifest")
    models = [previous.load_verified(*spec, matched_manifest) for spec in previous.MODELS]
    for model in models:
        model["cohort"] = MATCHED
    models.extend(load_expanded(slug, label, expanded_manifest) for slug, label, _ in previous.MODELS[:2])
    for cohort in (MATCHED, EXPANDED):
        group = [m for m in models if m["cohort"] == cohort]
        if (len({m["receipt"]["corpus_sha256"] for m in group}) != 1
                or len({m["receipt"]["document_order_sha256"] for m in group}) != 1
                or any(m["song_ids"] != group[0]["song_ids"] for m in group)):
            raise ValueError("Each comparison cohort must preserve document hashes, order and song IDs")
        if Counter(annotations[s]["language"] for s in group[0]["song_ids"]) != {
                "en": len(group[0]["song_ids"]) // 2, "es": len(group[0]["song_ids"]) // 2}:
            raise ValueError("Both existing cohorts must retain their balanced EN/ES composition")
    checks, diagnostics, query_sets = neighbour_checks(models, annotations)
    overlap = neighbour_overlap(models, annotations)
    associations = cluster_association(models, annotations)
    figure_name = figure_comparison(models, checks)
    primary_checks = [r for r in checks if r["language"] == "all_languages"]
    summary = dict(annotation_songs=120, paired_readings=240,
        fear_reference_counts={str(k): v for k, v in Counter(r["fear_reference_consensus"] for r in annotations_list).items()},
        consensus_motif_song_counts={t: sum(t in r["consensus"]["subtopics"] for r in annotations_list) for t in TOPICS},
        cohorts=[dict(cohort=c, query_corpus_songs=len(next(m for m in models if m["cohort"] == c)["song_ids"]),
            encoders=sum(m["cohort"] == c for m in models),
            metric_query_counts={metric: len(ids) for (cohort, metric), ids in query_sets.items() if cohort == c})
            for c in (MATCHED, EXPANDED)],
        primary_neighbor_checks=primary_checks, language_diagnostics=diagnostics,
        model_overlap=overlap,
        cluster_association_rows=len(associations),
        local_cluster_counts_including_noise={m["label"]: len({int(a["topic_id"]) for a in m["assignments"].values()})
            for m in models if m["cohort"] == EXPANDED},
        interpretive_limits=["Descriptive agreement with new paired assistant readings, not accuracy against human gold",
            "The four matched encoders share query IDs per metric; the two expanded encoders share a separate larger cohort",
            "Post-retrieval masks produce different usable edge counts, so fractions require their denominators",
            "No cross-encoder cosine magnitudes or universal encoder ranking",
            "No consensus motif support does not establish the motif's true absence",
            "A song can inhabit several BERTopic clusters; topic IDs are local to each encoder/fit",
            "Repeated destinations and overlapping memberships are dependent; no population or causal inference"])
    write_json(DEST / "summary.json", summary)
    receipts = [dict(cohort=m["cohort"], model_slug=m["slug"], model=m["label"], **m["receipt"]) for m in models]
    receipt = dict(schema_version=1, status="executed", computed_at=datetime.now(timezone.utc).isoformat(),
        annotation_sha256=projections.file_digest(ANNOTATIONS), vote_sha256=projections.file_digest(VOTES),
        annotation_scope="New uniform 120-song, 240-vote paired reading; prior posthoc40 panel is not used",
        script_sha256=projections.file_digest(Path(__file__)),
        reused_ranking_source_sha256=projections.file_digest(HERE / "visualize_encoder_comparison.py"),
        matched_manifest_sha256=projections.file_digest(matched_manifest_path),
        expanded_manifest_sha256=projections.file_digest(expanded_manifest_path), model_receipts=receipts,
        primary_k=K, original_full_dimension_space=True, pca_or_umap_used_for_neighbors=False,
        song_centroid_method="L2-normalize each original unique-block vector; mean within song; L2-normalize centroid; rank cosine; self excluded; song ID resolves ties",
        candidate_policy="Filter full ranking to all other same-language songs, retrieve first five, then mask unavailable comparison labels; 19 alternatives in matched40 and 59 in expanded120",
        query_ids_by_cohort_and_metric={f"{c}:{metric}": ids for (c, metric), ids in query_sets.items()},
        uniform_conditional_reference="For each query, matching eligible candidates / all eligible candidates within its same-language candidate universe; weight by that query's retained labeled edge count",
        neighbor_aggregation="Micro-average directed labeled edges; EN/ES strata also exported; unresolved fear and unsupported motif intersections are masked after top-five retrieval",
        fear_binary_definition="True fear_reference_consensus (including explicit_or_implied) versus False; None is omitted",
        motif_intersection_definition="Both songs have fear-reference consensus and at least one two-reader-supported motif; intersection of the seven consensus motif sets is nonempty",
        cluster_membership="At least one unique nonrepeated block in a local BERTopic cluster, counted once per song/cluster; overlapping memberships allowed; -1 is unassigned noise",
        cluster_motif_denominator="All 120 songs for global supported-presence counts; cluster song memberships for within-cluster fractions. No consensus support is not a resolved psychological absence; fear-unresolved counts are explicit",
        cluster_uniform_reference="Hold cluster song counts in each language fixed. Sum n_cluster_language * n_consensus_motif_language / n_total_language. Each local cluster is compared with the same seven motif categories, never matched to another encoder's numeric topic ID",
        inference="Descriptive corpus-conditional associations and neighbour agreement; no human gold, accuracy ranking, significance test or population inference",
        api_calls=0, new_embeddings=0, changed_private_caches=False,
        private_exports="No lyrics, vectors, pairwise cosine matrices, model objects, keys or private filesystem paths",
        software={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
        figure_sha256={f"{figure_name}.{ext}": projections.file_digest(FIG / f"{figure_name}.{ext}") for ext in ("svg", "png")},
        output_sha256={p.name: projections.file_digest(p) for p in sorted(DEST.glob("*")) if p.is_file() and p.name != "receipt.json"})
    write_json(DEST / "receipt.json", receipt)
    print(json.dumps(dict(status="executed", annotation_songs=120, paired_readings=240, encoders=6,
                         cluster_association_rows=len(associations), api_calls=0,
                         metric_query_counts=summary["cohorts"], figure=figure_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
