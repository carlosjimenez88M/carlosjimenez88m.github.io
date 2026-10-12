"""Index completed phases from their existing receipts; no inference or network."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(relative):
    return {"path": relative, "sha256": sha(ROOT / relative)}


def encoders(base, expected_songs, expected_documents):
    rows = []
    for path in sorted((ROOT / base).glob("*/model_manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("results_status") != "executed":
            continue
        folder = path.parent.relative_to(ROOT).as_posix()
        projection = read(folder + "/projections.json")["receipt"]
        assert manifest["unique_documents"] == expected_documents
        assert projection["input_documents"] == expected_documents
        assert projection["input_songs"] == expected_songs
        stability = folder + "/seed_stability.csv"
        rows.append({
            "model": manifest["encoder"]["model_id"],
            "dimensions": manifest["encoder"]["dimensions"],
            "songs": projection["input_songs"],
            "unique_documents": projection["input_documents"],
            "corpus_sha256": projection["corpus_sha256"],
            "document_order_sha256": projection["document_order_sha256"],
            "pca": {key: projection[key] for key in [
                "pca_fit_components", "projection_components", "solver",
                "fit_scope", "explained_variance_total"]},
            "receipts": [artifact(folder + "/model_manifest.json"),
                         artifact(folder + "/projections.json"),
                         artifact(stability)],
        })
    assert len({row["corpus_sha256"] for row in rows}) == 1
    assert len({row["document_order_sha256"] for row in rows}) == 1
    return rows


def cohort(base, count, documents):
    diagnostics = read(base + "/corpus_diagnostics.json")
    assert diagnostics["songs_with_eligible_blocks"] == count
    assert diagnostics["unique_blocks"] == documents
    protocol = read(base + "/analysis_protocol.json")
    models = encoders(base, count, documents)
    rag = []
    for folder in ["rag", "rag-e5", "rag-openai-large", "rag-openai-small"]:
        rel = base + "/" + folder
        if not (ROOT / rel / "runs.json").exists():
            continue
        recipe = read(rel + "/protocol.json")
        runs = read(rel + "/runs.json")
        assert len(runs) == 192
        rag.append({"model": recipe["model"], "traces": len(runs),
                    "receipts": [artifact(rel + "/protocol.json"),
                                 artifact(rel + "/runs.json")]})
    return {
        "songs": count, "unique_blocks": documents,
        "all_blocks": diagnostics["blocks"],
        "decade_language_cells": [
            {key: row[key] for key in ["decade", "language", "songs"]}
            for row in diagnostics["decade_language_cells"]],
        "topic_seeds": protocol["seeds"], "encoders": models,
        "rag": rag,
        "bayes": {key: read(base + "/bayes/summary.json")[key]
                  for key in ["status", "cell_topic_rows", "posterior_rows",
                              "pooled_eb_boundary_fits", "pooled_eb_fits"]},
        "receipts": [artifact(base + "/corpus_diagnostics.json"),
                     artifact(base + "/bayes/summary.json")],
        "last_cli_invocation_only": {
            "models": protocol["models"],
            "note": "This invocation is not a receipt for all accumulated encoders.",
            "receipts": [artifact(base + "/analysis_protocol.json"),
                         artifact(base + "/run_status.json")]},
    }


def main():
    notebook_rel = "anexo-auditoria-metodologica.ipynb"
    notebook = read(notebook_rel)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    errors = [output for cell in code for output in cell.get("outputs", [])
              if output.get("output_type") == "error"]
    assert len(code) == 22 and not errors
    assert all(cell.get("execution_count") is not None for cell in code)
    stable_code_outputs = [{key: cell.get(key) for key in [
        "source", "execution_count", "outputs"]} for cell in code]
    code_outputs_sha = hashlib.sha256(json.dumps(
        stable_code_outputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    source = read("source-audit.json")
    coverage_rows = list(csv.DictReader((ROOT / "provider-coverage.csv").open()))
    panel = read("pilot-comparison/results/arc-panel/summary.json")
    fields = read("pilot-comparison/results/arc-panel/fieldwise-posthoc/summary.json")
    contexts = read("pilot-comparison/results/llm-comparison-context/summary.json")
    synthetic = read("results/llm-comparison-synthetic/summary.json")
    matched = cohort("results", 40, 386)
    expanded = cohort("expanded/results", 120, 1137)
    assert len(matched["encoders"]) == 4 and len(expanded["encoders"]) == 2
    assert len(coverage_rows) == 240
    article = ROOT.parents[1] / "content/post/2026-10-11-miedo-embeddings-bertopic-rag.md"
    report = {
        "schema_version": 1,
        "status": "completed_with_uniform_annual_revision_and_separate_optional_hosted_extension",
        "study_cutoff": "2026-10-11",
        "origin": "Aggregated from existing executed receipts; not a new experiment.",
        "historical_pilot": {"songs": 40, "unique_blocks": 386,
                             "freeze_receipt": artifact("pilot-40/freeze.json"),
                             "policy": "Frozen records are preserved unchanged."},
        "matched_comparison": matched,
        "nested_local_expansion": expanded,
        "corpus_provenance": {
            "songs": source["integrity"]["selected_songs"],
            "private_response_files": source["integrity"]["provider_response_files"],
            "public_coverage_rows": len(coverage_rows),
            "source_audit": artifact("source-audit.json")},
        "narrative_panel": {
            "songs": panel["songs"], "votes": panel["votes"],
            "models": panel["protocol"]["models"],
            "third_reviews": panel["third_reviews"],
            "primary_status_counts": panel["status_counts"],
            "fieldwise_posthoc": fields["fields"],
            "receipts": [artifact("pilot-comparison/results/arc-panel/summary.json"),
                         artifact("pilot-comparison/results/arc-panel/fieldwise-posthoc/summary.json")]},
        "matched_context_readers": {
            "contexts": contexts["protocol"]["test_queries"],
            "decisions": sum(row["valid_reviews"] for row in contexts["models"]),
            "models": contexts["protocol"]["models"],
            "receipt": artifact("pilot-comparison/results/llm-comparison-context/summary.json")},
        "synthetic_reader_comparison": {
            "models": [row["model_id"] for row in synthetic["models"]],
            "triplets_per_model": [row["requested"] for row in synthetic["models"]],
            "valid_responses": sum(row["valid"] for row in synthetic["models"]),
            "receipt": artifact("results/llm-comparison-synthetic/summary.json")},
        "notebook": {"executed_code_cells": len(code), "errors": len(errors),
                     "code_outputs_sha256": code_outputs_sha,
                     "receipt": artifact(notebook_rel)},
        "article": {"path": article.relative_to(ROOT.parents[1]).as_posix(),
                    "sha256": sha(article), "state": "prepared_editorial_draft"},
        "costs": read("cost-summary.json"),
        "visual_and_interpretive_revision": {
            "remote_calls": 0,
            "methods": "Derived from cached matched embeddings, existing votes, exact posterior moments and saved LangGraph traces.",
            "fear_focus": read("results/historical-map/fear-focus-summary.json"),
            "receipts": [artifact(rel) for rel in [
                "results/historical-map/receipt.json", "results/encoder-comparison-new/receipt.json",
                "figures/bayes-readable-receipt.json", "results/langgraph-process-new/receipt.json"]],
        },
        "uniform_annual_revision": {
            "primary_scope": "120 songs, two title/artist/date-masked readings each; three processes of the same configured assistant.",
            "summary": read("results/annual-120/summary.json"),
            "project_api_calls": 0,
            "receipts": [artifact(rel) for rel in [
                "annual-reading-protocol.md", "results/annual-120/receipt.json",
                "results/annual-120/visualization-receipt.json",
                "results/annual-120/embeddings/receipt.json",
                "results/annual-120/bayes/annual-bayes-receipt.json"]],
            "previous_panel_policy": "Preserved as sensitivity, never pooled into new paired votes.",
        },
        "optional_hosted_expansion": {
            "songs": 120, "status": "not_executed_pending_user_authorization",
            "note": "Not counted as a completed experiment or required for the original study phases."},
        "privacy": "No lyric text, secrets, embedding arrays or model weights in this metadata index.",
    }
    (ROOT / "phase-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"matched_encoders": len(matched["encoders"]),
                      "expanded_encoders": len(expanded["encoders"]),
                      "notebook_code_cells": len(code), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
