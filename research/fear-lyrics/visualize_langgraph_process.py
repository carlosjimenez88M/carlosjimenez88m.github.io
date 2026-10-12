#!/usr/bin/env python3
"""Visualize executed LangGraph traces without rerunning retrieval or an API.

Two experiments remain distinct: the primary four-encoder retrieval comparison
and a matched mini/nano reader of contexts frozen from a separate ablation arm.
Only public questions, IDs, counters and decisions are read/exported. Figures
are diagnostic views of existing runs, not a new evaluation or a causal claim.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fear-lyrics-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results/langgraph-process-new"
FIG = HERE / "figures"
SITE = ROOT / "static/research/fear-lyrics"
BG, INK, MUTED = "#fbfaf7", "#263341", "#647180"
TEAL, ORANGE, RED, GRAY = "#147d92", "#d46b40", "#b84f5a", "#d7dce1"
MODELS = [
    ("MiniLM multilingüe", "rag", "#147d92"),
    ("E5 small multilingüe", "rag-e5", "#d46b40"),
    ("OpenAI 3-large", "rag-openai-large", "#8e6ba5"),
    ("OpenAI 3-small", "rag-openai-small", "#4c956c"),
]
CASE_IDS = ["intruder_identity_missing_es", "mutual_fear_es"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK,
    "savefig.facecolor": BG, "svg.fonttype": "none",
})


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def receipt(path: Path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def box(ax, x, y, w, h, text, color=TEAL, *, fontsize=10, fill=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.3, edgecolor=color, facecolor=color if fill else BG,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="white" if fill else INK,
            linespacing=1.4)


def arrow(ax, left, right, *, color=MUTED, bend=0):
    ax.add_patch(FancyArrowPatch(
        left, right, arrowstyle="-|>", mutation_scale=11,
        color=color, linewidth=1.3,
        connectionstyle=f"arc3,rad={bend}", shrinkA=4, shrinkB=4,
    ))


def save(fig, name, caption):
    fig.text(.045, .024, caption, color=MUTED, fontsize=9,
             va="bottom", linespacing=1.5)
    files = []
    for ext in ("svg", "png"):
        path = FIG / f"{name}.{ext}"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        shutil.copy2(path, SITE / path.name)
        files.append(receipt(path))
    plt.close(fig)
    return files


def audit_trace(row: dict):
    """Check counters against saved events; never repair a primary row."""
    trace = row["trace"]
    retrieves = [e for e in trace if e.startswith("retrieve:")]
    rewrites = [e for e in trace if e.startswith("rewrite:")]
    assesses = [e for e in trace if e.startswith("assess:")]
    assert len(retrieves) == row["retrieval_calls"]
    assert len(rewrites) == row["rewrites"]
    assert len(assesses) == len(retrieves)
    assert sum(int(e.split("slots=")[1]) for e in retrieves) == row["candidate_slots"]
    assert row["candidate_slots"] <= 30 and row["rewrites"] <= 2
    assert 0 <= row["context_words"] <= 320 and len(row["context_ids"]) <= 5
    assert row["citation_ids_valid"]
    assert row["abstained"] == any(e.startswith("abstain:") for e in trace)
    if row["abstained"]:
        assert not row["context_ids"] and row["context_words"] == 0
    else:
        assert any(e.startswith("context:") for e in trace)
        assert any(e.startswith("answer:deterministic") for e in trace)
    assert row["api_calls"] == 0


def aggregate(rows: list[dict], questions: dict, model_name: str):
    result = []
    for method in ("bm25", "dense", "hybrid"):
        for arm in ("one_pass", "bounded_contextual"):
            selected = [r for r in rows if r["split"] == "test"
                        and r["method"] == method and r["arm"] == arm]
            assert len(selected) == 24
            positive = [r for r in selected if questions[r["query_id"]]["answerable"]]
            negative = [r for r in selected if not questions[r["query_id"]]["answerable"]]
            assert len(positive) == 16 and len(negative) == 8
            passing = [r for r in selected if not r["abstained"]]
            rewrites = Counter(r["rewrites"] for r in selected)
            result.append({
                "encoder": model_name, "method": method, "arm": arm,
                "queries": len(selected), "positive_queries": len(positive),
                "negative_queries": len(negative),
                "positive_families": len({r["family"] for r in positive}),
                "negative_families": len({r["family"] for r in negative}),
                "recall_at_5": float(np.mean([r["recall_at_5"] for r in positive])),
                "context_support_recall": float(np.mean([r["context_support_recall"] for r in positive])),
                "positive_abstentions": sum(r["abstained"] for r in positive),
                "negative_safe_abstentions": sum(r["abstained"] for r in negative),
                "negative_packages_emitted": sum(not r["abstained"] for r in negative),
                "evidence_packages": len(passing),
                "rule_abstentions": sum(r["abstained"] for r in selected),
                "zero_rewrites": rewrites[0], "one_rewrite": rewrites[1], "two_rewrites": rewrites[2],
                "retrieval_calls_total": sum(r["retrieval_calls"] for r in selected),
                "candidate_slots_total": sum(r["candidate_slots"] for r in selected),
                "context_words_total": sum(r["context_words"] for r in selected),
                "context_words_mean_when_emitted": float(np.mean([r["context_words"] for r in passing])) if passing else 0,
                "all_context_ids_valid": all(r["citation_ids_valid"] for r in selected),
                "api_calls": sum(r["api_calls"] for r in selected),
            })
    return result


def overview_figure(rows):
    fig = plt.figure(figsize=(15, 10.8))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 1), hspace=.5, wspace=.35)
    ax = fig.add_subplot(grid[0, :])
    ax.set(xlim=(0, 15), ylim=(-.5, 4.2)); ax.axis("off")
    ax.text(0, 4.0, "LangGraph: el camino realmente ejecutado", fontsize=20, weight="bold")
    ax.text(0, 3.53, "Recuperación y reglas sobre 386 bloques únicos · cuatro encoders · mismo test y presupuesto",
            fontsize=11, color=MUTED)
    y, h, w = 2.10, .75, 1.76
    xs = [.05, 2.2, 4.35, 6.50, 8.65, 10.8, 13.03]
    texts = ["Consulta\noriginal EN / ES", "route\ncorpus congelado",
             "retrieve\nBM25 / dense / RRF", "assess\ngate léxico",
             "context\n≤ 5 bloques / 320 pal.", "answer\npaquete de IDs", "validation\nID en contexto"]
    for x, text in zip(xs, texts):
        box(ax, x, y, w, h, text, fontsize=9.7)
    for left, right in zip(xs, xs[1:]):
        arrow(ax, (left+w, y+h/2), (right, y+h/2))
    ax.text(8.37, 3.04, "gate pasa", color=TEAL, fontsize=9, ha="center")
    box(ax, 6.53, .65, 2.20, .80, "rewrite · hasta dos\nexpansiones bilingües", ORANGE, fontsize=10)
    arrow(ax, (7.38, y), (7.58, 1.48), color=ORANGE)
    arrow(ax, (6.5, 1.06), (5.22, y), color=ORANGE, bend=-.2)
    ax.text(4.78, .86, "gate falla\nqueda presupuesto", color=ORANGE, fontsize=9, ha="center")
    box(ax, 10.78, .65, 2.08, .80, "abstain\nsin paquete de evidencia", RED, fontsize=10)
    arrow(ax, (8.13, y+.02), (11.4, 1.49), color=RED, bend=-.16)
    arrow(ax, (12.9, 1.08), (13.70, 2.08), color=RED)
    ax.text(9.28, 1.61, "gate falla + sin reintento", color=RED, fontsize=9)
    ax.text(.05, -.12, "Una pasada: 1 × 30 candidatos. Acotado: hasta 3 × 10; vecinos antes del paquete.\n"
            "El gate comprueba coincidencia de términos; no demuestra que la evidencia sostenga todos los detalles pedidos.",
            fontsize=10, color=MUTED, va="bottom", linespacing=1.5)
    dense = [r for r in rows if r["method"] == "dense"]
    yloc = np.arange(4)
    names = [m[0] for m in MODELS]
    a = fig.add_subplot(grid[1, 0])
    b = fig.add_subplot(grid[1, 1])
    c = fig.add_subplot(grid[1, 2])
    for i, (name, _, color) in enumerate(MODELS):
        single = next(r for r in dense if r["encoder"] == name and r["arm"] == "one_pass")
        bounded = next(r for r in dense if r["encoder"] == name and r["arm"] == "bounded_contextual")
        a.plot([single["recall_at_5"], bounded["recall_at_5"]], [i, i], color=color, lw=2.5)
        a.scatter(single["recall_at_5"], i, s=60, facecolors=BG, edgecolors=color, linewidths=2, zorder=3)
        a.scatter(bounded["recall_at_5"], i, s=70, color=color, zorder=3)
        a.text(.50, i-.18, f"{single['recall_at_5']:.3f} → {bounded['recall_at_5']:.3f}", color=color, fontsize=9)
        b.plot([single["negative_safe_abstentions"], bounded["negative_safe_abstentions"]], [i, i], color=color, lw=2.5)
        b.scatter(single["negative_safe_abstentions"], i, s=60, facecolors=BG, edgecolors=color, linewidths=2, zorder=3)
        b.scatter(bounded["negative_safe_abstentions"], i, s=70, color=color, zorder=3)
        b.text(3.8, i-.18, f"{single['negative_safe_abstentions']}/8 → {bounded['negative_safe_abstentions']}/8", color=color, fontsize=9)
        left = 0
        for field, shade in [("zero_rewrites", TEAL), ("one_rewrite", "#dc9e5d"), ("two_rewrites", "#b84f5a")]:
            n = bounded[field]
            c.barh(i, n, left=left, height=.5, color=shade)
            if n: c.text(left+n/2, i, str(n), ha="center", va="center", fontsize=10, color="white")
            left += n
    a.set_title("Encuentra más soporte anotado", loc="left", weight="bold", pad=18)
    a.set(xlim=(.48, 1.02), ylim=(-.55, 3.65), yticks=yloc, yticklabels=names,
          xlabel="Recall@5 · media de 16 preguntas positivas")
    b.set_title("Pero acepta preguntas sin respuesta", loc="left", weight="bold", pad=18)
    b.set(xlim=(0, 8.6), ylim=(-.55, 3.65), yticks=yloc, yticklabels=[], xticks=(0, 2, 4, 6, 8),
          xlabel="Abstenciones · 8 preguntas negativas")
    c.set_title("Reintentos efectivamente usados", loc="left", weight="bold", pad=18)
    c.set(xlim=(0, 24), ylim=(-.55, 3.65), yticks=yloc, yticklabels=[], xticks=(0, 8, 16, 24),
          xlabel="24 preguntas test · brazo acotado")
    handles = [Patch(facecolor=shade, label=f"{label} reformulaciones")
               for shade, label in [(TEAL, "0"), ("#dc9e5d", "1"), ("#b84f5a", "2")]]
    c.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -.24), frameon=False, ncol=1, fontsize=9)
    for axis in (a, b, c): axis.invert_yaxis(); axis.grid(axis="x", alpha=.15)
    fig.text(.045, .135, "○ Una pasada   ● Acotado   |   Recuperación dense; 8 familias positivas y 4 negativas, cada una en EN y ES.",
             fontsize=10, color=MUTED)
    fig.subplots_adjust(left=.15, right=.98, bottom=.22, top=.98)
    return save(fig, "langgraph-process-overview",
                "192 trazas test dense = 4 encoders × 24 consultas × 2 brazos. Los otros 576 registros incluyen dev, BM25 e híbrido.\n"
                "El paquete de IDs es determinista: este grafo no genera una interpretación literaria. Recall y abstención son diagnósticos distintos.")


def event_label(event, row):
    if event.startswith("route:"):
        return "route\ncorpus fijo", "#647180"
    if event.startswith("retrieve:"):
        return "retrieve\n10 candidatos", TEAL
    if event.startswith("assess:"):
        good = event.endswith("True")
        return "assess\n" + ("pasa" if good else "falla"), TEAL if good else RED
    if event.startswith("rewrite:"):
        return f"rewrite {event.split(':')[-1]}\nexpansión EN/ES", ORANGE
    if event.startswith("context:"):
        return f"context\n{row['context_words']} palabras", TEAL
    if event.startswith("answer:"):
        return f"answer\n{len(row['context_ids'])} IDs", TEAL
    if event.startswith("abstain:"):
        return "abstain\n0 contexto", RED
    if event.startswith("citation_validation:"):
        return "validación\nsolo IDs", "#647180"
    raise ValueError(event)


def traces_figure(raw, questions):
    fig, axes = plt.subplots(2, 1, figsize=(17.3, 10.7))
    titles = [
        "Caso negativo: encontrar el motivo no prueba la identidad ni la fecha de detención",
        "Caso positivo: soporte recuperado que el gate léxico puede rechazar",
    ]
    selected = []
    for ax, query_id, title in zip(axes, CASE_IDS, titles):
        ax.set(xlim=(-2.45, 13.1), ylim=(-.95, 4.35)); ax.axis("off")
        ax.text(-2.45, 4.23, title, fontsize=14, weight="bold", va="top")
        query = questions[query_id]["query"]
        ax.text(-2.45, 3.65, textwrap.fill(query, width=155), fontsize=10, color=MUTED, va="top")
        for i, (name, slug, color) in enumerate(MODELS):
            row = next(r for r in raw[slug] if r["query_id"] == query_id and r["method"] == "dense"
                       and r["arm"] == "bounded_contextual")
            selected.append({"encoder": name, "query": query, **row})
            y = 2.56 - i*.83
            ax.text(-2.44, y+.26, name, color=color, weight="bold", fontsize=9.5, va="center")
            metric = "Sin soporte solicitado" if row["recall_at_5"] is None else f"Recall@5: {row['recall_at_5']:.2f}"
            ax.text(-2.44, y-.05, metric, fontsize=9, va="center", color=MUTED)
            for j, event in enumerate(row["trace"]):
                label, shade = event_label(event, row)
                box(ax, j*1.1, y-.13, .98, .57, label, shade, fontsize=7.8)
                if j: arrow(ax, (j*1.1-.11, y+.16), (j*1.1, y+.16), color=GRAY)
        note = ("Los cuatro pasan el gate y emiten un paquete, aunque la identidad legal y la fecha pedidas no están en el corpus."
                if query_id.startswith("intruder") else
                "MiniLM y ambos OpenAI recuperan todo el soporte anotado; los cuatro se abstienen por el gate. E5 no recupera ese soporte.")
        ax.text(-2.45, -.83, note, fontsize=10, color=RED, va="bottom")
    fig.suptitle("Cada caja es un evento guardado: hasta dos reescrituras; ninguna llamada nueva", x=.043,
                 ha="left", weight="bold", fontsize=18, y=.99)
    fig.subplots_adjust(left=.035, right=.99, top=.93, bottom=.11, hspace=.16)
    files = save(fig, "langgraph-process-traces",
                 "Selección ilustrativa posterior al experimento; resultados agregados en el gráfico anterior. Preguntas originales del benchmark, sin versos.\n"
                 "Orden y contadores provienen de runs.json. El grafo valida la existencia de IDs, no la verdad de una respuesta.")
    return files, selected


def reader_figure(reader_rows, protocol, questions):
    models = ["openai/gpt-5.4-mini", "openai/gpt-5.4-nano"]
    grouped = []
    for model in models:
        rows = [r for r in reader_rows if r["model"] == model]
        assert len(rows) == 24
        for category in ("positive_complete_context", "positive_incomplete_context", "negative"):
            if category == "negative":
                subset = [r for r in rows if not r["assistant_answerable"]]
            else:
                complete = category == "positive_complete_context"
                subset = [r for r in rows if r["assistant_answerable"] and r["assistant_all_support_in_context"] == complete]
            assert len(subset) == 8
            grouped.append({"model": model, "category": category, "contexts": len(subset),
                            "abstained": sum(r["final_abstained"] for r in subset),
                            "emitted": sum(not r["final_abstained"] for r in subset),
                            "invalid_citation_reviews": sum(not r["citation_ids_valid"] for r in subset)})
    pairs = {}
    for row in reader_rows:
        assert row["status"] == "executed" and row["citation_ids_valid"]
        pairs.setdefault(row["query_id"], {})[row["model"]] = row
    assert len(pairs) == 24
    for pair in pairs.values():
        left, right = [pair[m] for m in models]
        assert left["context_identity"] == right["context_identity"]
        assert left["context_ids"] == right["context_ids"]
    matching_decisions = sum(pair[models[0]]["support_decision"] == pair[models[1]]["support_decision"] for pair in pairs.values())
    matching_abstentions = sum(pair[models[0]]["final_abstained"] == pair[models[1]]["final_abstained"] for pair in pairs.values())
    differences = [{"query_id": key, "question": questions[key]["query"],
                    "mini": pair[models[0]]["support_decision"], "nano": pair[models[1]]["support_decision"],
                    "mini_abstained": pair[models[0]]["final_abstained"], "nano_abstained": pair[models[1]]["final_abstained"]}
                   for key, pair in pairs.items() if pair[models[0]]["final_abstained"] != pair[models[1]]["final_abstained"]]
    fig = plt.figure(figsize=(14.5, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.2), hspace=.34, wspace=.28)
    ax = fig.add_subplot(grid[0, :]); ax.axis("off"); ax.set(xlim=(0, 14.4), ylim=(-.65, 3.7))
    ax.text(0, 3.56, "Después: dos lectores sobre exactamente el mismo contexto", fontsize=19, weight="bold")
    ax.text(0, 3.08, "Experimento separado · 24 contextos congelados × mini / nano · no continúa las trazas dense anteriores",
            fontsize=10.5, color=MUTED)
    xs = [.03, 3.04, 6.05]
    labels = ["attach_frozen_context\nconsulta original\n+ ≤ 5 bloques", "support_reader\nmini o nano · JSON acotado", "citation_validation\nIDs solo dentro del contexto"]
    for x, label in zip(xs, labels): box(ax, x, 1.78, 2.61, .82, label, fontsize=10)
    for left, right in zip(xs, xs[1:]): arrow(ax, (left+2.61, 2.19), (right, 2.19))
    box(ax, 10.2, 2.22, 3.37, .75, "answer · afirmaciones privadas\nIDs + conteos públicos", TEAL, fontsize=10)
    box(ax, 10.2, .96, 3.37, .75, "abstain · faltan detalles\no falta apoyo", RED, fontsize=10)
    arrow(ax, (8.67, 2.18), (10.2, 2.59), color=TEAL)
    arrow(ax, (8.67, 2.14), (10.2, 1.33), color=RED)
    ax.text(.02, .94, "Contexto de otra fase: MiniLM híbrido, brazo rewrite_1_neighbors_0.\n"
            "El lector no elige herramientas ni modifica la búsqueda; responde o se abstiene con ese contexto fijo.",
            color=MUTED, fontsize=10.2, linespacing=1.5)
    ax.text(.02, .0, "Guard de respuesta: al menos una afirmación y una cita suministrada, sin detalles solicitados pendientes.\n"
            "Que la cita resuelva un ID es condición necesaria; no comprueba por sí sola la interpretación.",
            color=MUTED, fontsize=10.2, linespacing=1.5)
    chart = fig.add_subplot(grid[1, 0])
    categories = ["positive_complete_context", "positive_incomplete_context", "negative"]
    labels = ["Positiva · soporte completo", "Positiva · soporte incompleto", "Negativa · detalle ausente"]
    for j, model in enumerate(models):
        for i, category in enumerate(categories):
            row = next(r for r in grouped if r["model"] == model and r["category"] == category)
            y = i*2.1 + j*.61
            chart.barh(y, row["emitted"], height=.47, color=TEAL)
            chart.barh(y, row["abstained"], left=row["emitted"], height=.47, color=RED)
            if row["emitted"]: chart.text(row["emitted"]/2, y, str(row["emitted"]), ha="center", va="center", color="white")
            if row["abstained"]: chart.text(row["emitted"]+row["abstained"]/2, y, str(row["abstained"]), ha="center", va="center", color="white")
            chart.text(-.12, y, "mini" if j == 0 else "nano", ha="right", va="center", fontsize=9)
    chart.set(yticks=[i*2.1+.305 for i in range(3)], yticklabels=labels,
              xlim=(-.05, 8.1), xlabel="8 contextos por grupo · mismo conjunto para ambos lectores", xticks=(0, 2, 4, 6, 8))
    chart.tick_params(axis="y", pad=36, length=0)
    chart.invert_yaxis()
    chart.set_title("Acuerdo agregado; dos desacuerdos concretos", loc="left", weight="bold", fontsize=12)
    chart.legend(handles=[Patch(facecolor=TEAL, label="Respuesta emitida"), Patch(facecolor=RED, label="Abstención")],
                 frameon=False, bbox_to_anchor=(0, -.18), loc="upper left", fontsize=9)
    note = fig.add_subplot(grid[1, 1]); note.axis("off")
    note.text(0, 1, f"Decisión de apoyo idéntica: {matching_decisions}/24\nAbstención final idéntica: {matching_abstentions}/24",
              weight="bold", fontsize=15, va="top", linespacing=1.7)
    note.text(0, .73, "Los mismos totales no implican las mismas respuestas.", fontsize=10.5, color=MUTED)
    start = .63
    for difference in differences:
        note.text(0, start, difference["query_id"], weight="bold", fontsize=10)
        status = lambda key: "se abstiene" if difference[key] else "emite respuesta"
        note.text(0, start-.07, f"mini: {status('mini_abstained')} · nano: {status('nano_abstained')}", fontsize=10)
        start -= .20
    example = pairs["intruder_identity_missing_es"]
    assert all(example[m]["final_abstained"] for m in models)
    note.text(0, .12, "Identidad del intruso: ambos se abstienen.\nEn este contexto distinto, reconocen que faltan los detalles.\nEl contexto anterior y este no se enlazan como una sola ejecución.",
              fontsize=10, color=RED, linespacing=1.5)
    fig.subplots_adjust(left=.19, right=.98, bottom=.17, top=.98)
    files = save(fig, "langgraph-process-reader",
                 "48 revisiones guardadas. Soporte completo/incompleto se refiere a los IDs de referencia del benchmark, no a una anotación exhaustiva.\n"
                 "24 preguntas = 12 familias EN/ES. La pertenencia de las citas fue válida en 48/48; el acuerdo entre lectores no mide verdad.")
    return files, {"categories": grouped, "same_support_decision": matching_decisions,
                   "same_final_abstention": matching_abstentions, "paired_contexts": len(pairs),
                   "final_abstention_disagreements": differences,
                   "context_origin": {k: protocol[k] for k in ("retrieval_source", "method", "arm", "context_pool_identity")}}


def main():
    for path in (OUT, FIG, SITE): path.mkdir(parents=True, exist_ok=True)
    sources = [HERE / "rag.py", Path(__file__).resolve()]
    all_aggregates, raw, questions, traces = [], {}, None, 0
    for name, slug, _ in MODELS:
        folder = HERE / "results" / slug
        files = [folder / "runs.json", folder / "questions.json", folder / "protocol.json"]
        sources.extend(files)
        rows = read(files[0]); raw[slug] = rows
        current_questions = {q["query_id"]: q for q in read(files[1])}
        if questions is None: questions = current_questions
        assert current_questions == questions, "Primary encoder comparison must use identical questions."
        assert len(rows) == 192
        for row in rows: audit_trace(row)
        traces += len(rows)
        all_aggregates.extend(aggregate(rows, questions, name))
    reader_path = HERE / "pilot-comparison/results/llm-comparison-context"
    sources += [reader_path / "runs.json", reader_path / "protocol.json", reader_path / "summary.json"]
    reader_rows, reader_protocol = read(reader_path / "runs.json"), read(reader_path / "protocol.json")
    # This origin must remain separate from the primary dense four-encoder graph.
    assert reader_protocol["arm"] == "rewrite_1_neighbors_0" and reader_protocol["method"] == "hybrid"
    figures = overview_figure(all_aggregates)
    files, cases = traces_figure(raw, questions); figures += files
    files, readers = reader_figure(reader_rows, reader_protocol, questions); figures += files
    with (OUT / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_aggregates[0]))
        writer.writeheader(); writer.writerows(all_aggregates)
    write(OUT / "illustrative-traces.json", {
        "selection": "Posthoc illustrative cases, not a new held-out evaluation.",
        "primary_method": "dense", "primary_arm": "bounded_contextual", "cases": cases,
    })
    write(OUT / "reader-comparison.json", readers)
    write(OUT / "receipt.json", {
        "status": "Rendered offline from immutable executed traces; no retrieval, model or API calls.",
        "primary_cohort": {"songs": 40, "unique_blocks": 386, "encoders": 4, "saved_runs": traces,
                           "visualized_test_dense_runs": 192, "test_queries_per_arm_encoder": 24,
                           "positive_families": 8, "negative_families": 4, "languages": ["en", "es"]},
        "context_reader": {"saved_reviews": len(reader_rows), "matched_contexts": 24,
                           "models": ["openai/gpt-5.4-mini", "openai/gpt-5.4-nano"],
                           "origin": readers["context_origin"], "separate_experiment": True},
        "trace_integrity": {"audited": traces, "counter_mismatches": 0, "budget_violations": 0,
                            "invalid_citation_membership": 0},
        "interpretation": [
            "Bounded dense retrieval improves mean annotated Recall@5 for all four encoders in this test.",
            "Safe negative abstention decreases to 4/8 for each encoder: lexical gating accepts relevant motifs with unsupported requested details.",
            "Some fully recovered positive support is rejected by the lexical gate; retrieval rank and accepted context are separate outcomes.",
            "Mini and nano share aggregate counts but differ on two final abstention decisions; agreement is not correctness.",
            "Reader contexts originated in a different hybrid ablation arm; no dense trace was continued into these reviews.",
        ],
        "privacy": "Public source questions, IDs, counters and decisions only; no lyric text, embeddings, credentials or generated private claims.",
        "sources": [receipt(p) for p in sources], "figures": figures,
        "api_calls": 0, "model_calls": 0, "new_cost_usd": 0,
    })
    print(json.dumps({"status": "rendered", "trace_integrity": traces,
                      "figures": [p["path"] for p in figures], "reader_contexts": 24,
                      "api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__": main()
