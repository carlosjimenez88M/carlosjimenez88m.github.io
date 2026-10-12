"""Fit one global PCA per model from verified LOCAL private embedding caches.

Only three-dimensional coordinates, public metadata, aggregate centroids, and
receipts are exported. No lyrics, original vectors, PCA loadings or fitted PCA
objects are public. This module does not load encoders or make network calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_matches(actual, expected):
    return actual == expected or (expected.startswith("openai/") and actual == expected.split("/", 1)[1])


def locate_embeddings(model_id, model_slug, cache, bundles):
    """Resolve an indexed local matrix or a verified external NPZ sidecar."""
    # A frozen-cohort bundle deliberately overrides an index that may now point
    # at an expanded corpus. A bad explicit bundle fails; it never falls back.
    if model_id in bundles:
        array_path = bundles[model_id]
        sidecar = array_path.with_suffix(".json")
        metadata = read_json(sidecar)
        if not model_matches(metadata.get("model_id"), model_id):
            raise ValueError("Explicit embedding bundle model does not match the analysis manifest")
        with np.load(array_path, allow_pickle=False) as bundle:
            matrix = np.asarray(bundle["documents"], dtype=np.float64)
        return matrix, metadata, array_path, sidecar
    index_path = cache / "embeddings" / model_slug / "document-index.json"
    if index_path.exists():
        metadata = read_json(index_path)
        array_path = Path(metadata["array_path"])
        if not array_path.is_absolute():
            array_path = index_path.parent / array_path
        if not array_path.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Indexed embeddings must remain inside the private cache")
        if not model_matches(metadata.get("model_id"), model_id):
            raise ValueError("Embedding index model does not match the analysis manifest")
        matrix = np.load(array_path, allow_pickle=False)
        return np.asarray(matrix, dtype=np.float64), metadata, array_path, index_path
    candidates = sorted(cache.glob("*.npz"))
    for array_path in candidates:
        sidecar = array_path.with_suffix(".json")
        if not sidecar.exists():
            continue
        metadata = read_json(sidecar)
        if not model_matches(metadata.get("model_id"), model_id):
            continue
        with np.load(array_path, allow_pickle=False) as bundle:
            matrix = np.asarray(bundle["documents"], dtype=np.float64)
        return matrix, metadata, array_path, sidecar
    raise FileNotFoundError("No verified local embedding cache exists for this model")


def unique_assignments(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unique = [row for row in rows if row["document_id"] == row["canonical_id"]
              and row.get("exact_repeat", "False").lower() not in ("true", "1")]
    if len({row["document_id"] for row in unique}) != len(unique):
        raise ValueError("Unique assignments contain duplicate document IDs")
    return {row["document_id"]: row for row in unique}


def weighted_centroid(matrix, rows, language="balanced_available_languages"):
    """Mean blocks within song, songs within language, then available languages.

    Individual input embeddings are L2-normalized. Intermediate centroids are
    deliberately NOT renormalized, so this is an arithmetic mixture of the
    original normalized embedding space, before the single PCA transform.
    """
    selected = [row for row in rows if language == "balanced_available_languages"
                or row["language"] == language]
    if not selected:
        return None, []
    song_rows = defaultdict(list)
    for row in selected:
        song_rows[(row["language"], row["song_id"])].append(row["index"])
    languages = defaultdict(list)
    for (lang, _), indices in song_rows.items():
        languages[lang].append(matrix[indices].mean(axis=0))
    centroid = np.mean([np.mean(vectors, axis=0) for vectors in languages.values()], axis=0)
    return centroid, selected


def temporal_centroids(matrix, rows, pca, minimum_line_songs=3):
    centroids = []
    for granularity in ("year", "decade"):
        groups = defaultdict(list)
        for row in rows:
            period = row["year"] if granularity == "year" else row["year"] // 10 * 10
            groups[(row["topic_id"], period)].append(row)
        for (topic_id, period), group in sorted(groups.items()):
            for language in ("balanced_available_languages", *sorted({r["language"] for r in group})):
                centroid, selected = weighted_centroid(matrix, group, language)
                if centroid is None:
                    continue
                # This is exactly PCA.transform(...)[0, :3] for whiten=False.
                # Elementwise reduction avoids platform BLAS floating-status
                # warnings when projecting a single centroid against 20 PCs.
                coords = np.sum((centroid - pca.mean_)[None, :] * pca.components_[:3], axis=1)
                # Rank only observed members of this topic-period in the ORIGINAL
                # normalized embedding space, never from the 3D display distance.
                representatives = []
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-12:
                    # Elementwise reduction avoids platform BLAS floating-status
                    # warnings for otherwise finite, bounded one-vector products.
                    cosine = {row["index"]: float(np.sum(matrix[row["index"]] * centroid) / norm)
                              for row in selected}
                    if not all(np.isfinite(value) for value in cosine.values()):
                        raise ValueError("A centroid representative cosine is nonfinite")
                    ranked = sorted(selected, key=lambda row: (
                        -cosine[row["index"]], row["document_id"]))
                    represented_songs = set()
                    for row in ranked:
                        if row["song_id"] in represented_songs:
                            continue
                        represented_songs.add(row["song_id"])
                        representatives.append(dict(document_id=row["document_id"],
                            song_id=row["song_id"], title=row["title"], artist=row["artist"],
                            year=row["year"], language=row["language"],
                            cosine_to_centroid=cosine[row["index"]]))
                        if len(representatives) == 3:
                            break
                song_ids = sorted({row["song_id"] for row in selected})
                language_counts = {lang: len({r["song_id"] for r in selected if r["language"] == lang})
                                   for lang in sorted({r["language"] for r in selected})}
                centroids.append(dict(granularity=granularity, period=period, topic_id=topic_id,
                    language=language, x=float(coords[0]), y=float(coords[1]), z=float(coords[2]),
                    songs=len(song_ids), artists=len({r["artist"] for r in selected}),
                    blocks=len(selected), languages_present=len(language_counts),
                    language_counts=language_counts, song_ids=song_ids,
                    representatives=representatives,
                    representative_method="original_embedding_cosine_to_weighted_centroid_max_3_distinct_observed_songs",
                    line_eligible=len(song_ids) >= minimum_line_songs,
                    support_status="adequate_for_descriptive_line" if len(song_ids) >= minimum_line_songs
                    else "sparse_observed_group_no_line",
                    weighting="equal_block_within_song_then_equal_song_within_language_then_equal_available_language",
                    observed=True, imputed=False))
    return centroids


def project_model(directory, manifest, cache, songs, manifest_path, bundles, minimum_line_songs=3):
    model_manifest = read_json(directory / "model_manifest.json")
    model_id = model_manifest["encoder"]["model_id"]
    assignments = unique_assignments(directory / "section_assignments.csv")
    matrix, metadata, array_path, sidecar = locate_embeddings(model_id, directory.name, cache, bundles)
    document_ids = metadata.get("document_ids", [])
    hashes = metadata.get("document_hashes", [])
    if matrix.ndim != 2 or matrix.shape[0] != len(document_ids) or len(hashes) != len(document_ids):
        raise ValueError("Embedding matrix and its verified document sidecar disagree")
    if metadata.get("shape") and list(matrix.shape) != metadata["shape"]:
        raise ValueError("Embedding matrix shape does not match its private index")
    expected_dimensions = model_manifest["encoder"].get("dimensions")
    if expected_dimensions is not None and matrix.shape[1] != int(expected_dimensions):
        raise ValueError("Embedding dimensions do not match the executed model manifest")
    if len(set(document_ids)) != len(document_ids) or set(document_ids) != set(assignments):
        raise ValueError("Embedding documents do not exactly match the unique analysis assignments")
    if not np.isfinite(matrix).all() or min(matrix.shape) < 3:
        raise ValueError("PCA requires at least three finite documents and dimensions")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Embedding matrix contains a zero vector")
    matrix = matrix / norms
    rows = []
    for index, document_id in enumerate(document_ids):
        assignment = assignments[document_id]
        meta = songs.get(assignment["song_id"])
        if meta is None:
            raise ValueError("A document has no public song metadata")
        rows.append(dict(index=index, document_id=document_id, song_id=assignment["song_id"],
                         topic_id=int(assignment["topic_id"]), year=int(meta["year"]),
                         language=meta["language"], artist=meta["artist"], title=meta["title"]))
    # No yearly refits, stochastic UMAP, or unverified z coordinate is used here.
    # A single global full-SVD fit supplies both the first three display axes
    # and the scree spectrum. Later PCs never contribute to displayed xyz.
    fit_components = min(20, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=fit_components, svd_solver="full", whiten=False)
    coordinates = pca.fit_transform(matrix)
    if not np.isfinite(coordinates).all() or not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError("PCA produced nonfinite coordinates or undefined variance")
    points = [dict(**{k: v for k, v in row.items() if k != "index"},
                   x=float(coords[0]), y=float(coords[1]), z=float(coords[2]))
              for row, coords in zip(rows, coordinates)]
    centroids = temporal_centroids(matrix, rows, pca, minimum_line_songs)
    spectrum = [float(value) for value in pca.explained_variance_ratio_]
    explained = spectrum[:3]
    receipt = dict(model_id=model_id, input_documents=len(rows),
        input_songs=len({row["song_id"] for row in rows}), input_dimensions=matrix.shape[1],
        corpus_sha256=digest(list(zip(document_ids, hashes))), document_order_sha256=digest(document_ids),
        public_manifest_sha256=file_digest(manifest_path), embedding_file_sha256=file_digest(array_path),
        embedding_sha256=hashlib.sha256(matrix.astype("<f8").tobytes(order="C")).hexdigest(),
        metadata_sha256=file_digest(sidecar), topic_assignments_sha256=file_digest(directory / "section_assignments.csv"),
        pca_components_sha256=hashlib.sha256(pca.components_.astype("<f8").tobytes(order="C")).hexdigest(),
        projection_components_sha256=hashlib.sha256(pca.components_[:3].astype("<f8").tobytes(order="C")).hexdigest(),
        explained_variance_ratio=explained, explained_variance_total=sum(explained),
        pca_fit_components=fit_components, projection_components=3,
        scree_components=list(range(1, fit_components + 1)),
        scree_explained_variance_ratio=spectrum,
        scree_cumulative_variance_ratio=[float(value) for value in np.cumsum(spectrum)],
        maximum_possible_components=min(matrix.shape),
        fit_scope="all_unique_blocks_global_per_model", fit_weighting="equal_unique_blocks",
        input_normalization="L2 per original document vector", centroid_normalization="none after arithmetic averaging",
        solver="full", seed=None, seed_note="Deterministic full SVD; no random seed is used",
        whiten=False, center=True, minimum_line_songs=minimum_line_songs,
        line_rule="Connect only adjacent observed periods when BOTH endpoints have at least minimum_line_songs; no interpolation across absent or sparse periods",
        software={key: importlib.metadata.version(key) for key in ("numpy", "scikit-learn")},
        computed_at=datetime.now(timezone.utc).isoformat(),
        interpretation="PCA distances retain only a fraction of embedding variance; axes are not fear, intensity, or historical prevalence scales. Temporal centroids describe changing selected-song composition, not motion of the same songs.")
    output = dict(schema_version=1, status="executed", method="PCA_global_3D", receipt=receipt,
                  points=points, centroids=centroids,
                  privacy="Only low-dimensional coordinates, public metadata and aggregate receipts; no lyrics, high-dimensional vectors, PCA loadings or fitted object.")
    destination = directory / "projections.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)
    return dict(model=model_id, documents=len(points), songs=receipt["input_songs"],
                explained_variance_total=receipt["explained_variance_total"], centroids=len(centroids),
                annual_groups_with_line_support=sum(r["line_eligible"] for r in centroids if r["granularity"] == "year" and r["language"] == "balanced_available_languages"),
                output=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--cache", type=Path, default=ROOT / ".research-cache" / "fear-lyrics")
    parser.add_argument("--manifest", type=Path, default=HERE / "corpus-manifest.json")
    parser.add_argument("--model", action="append", default=[], help="Model ID or result directory slug; repeatable")
    parser.add_argument("--bundle", action="append", default=[], help="MODEL_ID=LOCAL_NPZ_PATH; optional explicit local bundle")
    parser.add_argument("--minimum-line-songs", type=int, default=3)
    args = parser.parse_args()
    if args.minimum_line_songs < 3:
        parser.error("At least three distinct songs are required for a descriptive temporal line")
    manifest = read_json(args.manifest)
    songs = {row["song_id"]: row for row in manifest}
    bundles = {key: Path(value) for key, value in (item.split("=", 1) for item in args.bundle)}
    success, failed = [], []
    for directory in sorted(args.results.iterdir()):
        if not directory.is_dir() or not (directory / "model_manifest.json").exists():
            continue
        model_id = read_json(directory / "model_manifest.json")["encoder"]["model_id"]
        if args.model and model_id not in args.model and directory.name not in args.model:
            continue
        try:
            success.append(project_model(directory, manifest, args.cache, songs, args.manifest,
                                         bundles, args.minimum_line_songs))
        except (ValueError, FileNotFoundError, KeyError) as error:
            failed.append(dict(model=model_id, status="failed", reason=str(error)))
    print(json.dumps(dict(executed=success, failed=failed), ensure_ascii=False))
    return 1 if failed or not success else 0


if __name__ == "__main__":
    raise SystemExit(main())
