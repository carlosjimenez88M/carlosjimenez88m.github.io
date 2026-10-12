"""Render scientific figures and a metadata-only interactive research atlas."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram
import plotly
from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
FIG = HERE / "figures"
SITE = ROOT / "static/research/fear-lyrics"
COLORS = ["#147d92", "#d46b40", "#8e6ba5", "#4c956c", "#bb8d22", "#c25279", "#5877b8", "#6e8b87"]
NAMES = {"paraphrase_multilingual_MiniLM_L12_v2": "MiniLM multilingüe", "multilingual_e5_small": "E5 small multilingüe", "text_embedding_3_large": "OpenAI embedding 3 large", "text_embedding_3_small": "OpenAI embedding 3 small"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                    "axes.spines.right": False, "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7",
                    "text.color": "#263341", "axes.labelcolor": "#263341", "savefig.facecolor": "#fbfaf7"})

def read(path):
    return json.loads(path.read_text())

def label(path):
    key = path.name
    return next((v for k, v in NAMES.items() if k.lower() in key.lower().replace("-", "_")), key)

def save(fig, name, caption):
    fig.text(.06, .015, caption, fontsize=9, color="#647180", va="bottom", wrap=True)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

def title(topic):
    if topic["topic_id"] == -1:
        return "Sin asignación (-1)"
    # Terms describe the algorithm output; these are deliberately not human fear labels.
    terms = " · ".join(t["term"] for t in topic["terms"][:3])
    return f"T{topic['topic_id']}  {terms}"

def hierarchy_linkage(topics, hierarchy):
    leaves = [t["topic_id"] for t in topics if t["topic_id"] != -1]
    nodes = {str(t): i for i, t in enumerate(leaves)}
    if hierarchy.get("status") == "fallback":
        nodes = {x["node_id"]: leaves.index(x["topic_id"]) for x in hierarchy["leaves"]}
    sizes = {i: 1 for i in range(len(leaves))}
    links = []
    edges = hierarchy.get("edges", [])
    if hierarchy.get("status") == "ok":
        edges = sorted(edges, key=lambda e: int(e["parent_id"]))
    for edge in edges:
        a, b = str(edge["child_left_id"]), str(edge["child_right_id"])
        if a not in nodes or b not in nodes:
            return None, leaves
        ai, bi = nodes[a], nodes[b]
        count = sizes[ai] + sizes[bi]
        links.append([ai, bi, edge["distance"], count])
        node = len(leaves) + len(links) - 1
        nodes[str(edge["parent_id"])] = node
        sizes[node] = count
    return np.array(links) if len(links) == len(leaves) - 1 else None, leaves

def build():
    FIG.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    manifest = read(HERE / "corpus-manifest.json")
    if (RESULTS / 'section_inventory.csv').exists():
        cohort_ids = set(pd.read_csv(RESULTS / 'section_inventory.csv').song_id)
        manifest = [r for r in manifest if r['song_id'] in cohort_ids]
    songs = {s["song_id"]: s for s in manifest}
    diag = read(RESULTS / "corpus_diagnostics.json")
    cards_file = HERE / 'interpretive-cards.json'
    cards = read(cards_file)['cards'] if cards_file.exists() else []
    coverage = pd.read_csv(HERE / "provider-coverage.csv")
    agreements = pd.read_csv(HERE / "provider-agreement.csv")
    coverage = coverage[coverage.song_id.isin(songs)]
    agreements = agreements[agreements.song_id.isin(songs)]
    providers = sorted(coverage.provider.unique())
    counts = [int(((coverage.provider == p) & (coverage.status.astype(str) == "200")).sum()) for p in providers]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    cells = pd.DataFrame(diag["decade_language_cells"])
    for language, color, offset in [("en", COLORS[0], -.17), ("es", COLORS[1], .17)]:
        rows = cells[cells.language == language].sort_values("decade")
        axes[0].bar(np.arange(5) + offset, rows.songs, width=.32, color=color, label=language.upper())
    axes[0].set(xticks=range(5), xticklabels=["1980s", "1990s", "2000s", "2010s", "2020–2026*"], ylabel="Canciones", ylim=(0, max(cells.songs)*1.3))
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title(f"{diag['songs_with_lyrics']} canciones · {diag['artists']} artistas", loc="left", weight="bold")
    axes[1].barh(providers, counts, color=COLORS[:len(providers)])
    axes[1].set(xlim=(0, diag["songs_with_lyrics"]*1.13), xlabel=f"Letras recuperadas / {diag['songs_with_lyrics']}")
    for i, n in enumerate(counts): axes[1].text(n + .7, i, str(n), va="center")
    axes[1].set_title("Cobertura de proveedores", loc="left", weight="bold")
    fig.tight_layout(rect=(0, .10, 1, .97))
    save(fig, "corpus-providers", "*Corte: 11 de octubre de 2026. Selección intencional; equilibrio de idioma no implica representatividad.")

    models = []
    for path in sorted(RESULTS.iterdir(), key=lambda p: (0 if p.name.startswith("intfloat") else 1 if p.name.startswith("sentence-transformers") else 2, p.name)):
        if not path.is_dir() or not (path / "model_manifest.json").exists(): continue
        m = {"slug": path.name, "name": label(path), "manifest": read(path / "model_manifest.json"),
             "probes": read(path / "semantic_probes.json")["probes"]}
        if not (path / "topics.json").exists():
            models.append(m); continue
        m["topics"] = read(path / "topics.json")["topics"]
        if (path / "projections.json").exists(): m["projections"] = read(path / "projections.json")
        m["hierarchy"] = read(path / "topic_hierarchy.json")
        for field, filename in [("map", "exploratory_map.csv"), ("composition", "decade_composition.csv"),
                ("stability", "seed_stability.csv"), ("concentration", "artist_concentration.csv"),
                ("paths", "narrative_decade_paths.csv"), ("order", "narrative_order_tests.csv"),
                ("refrains", "repeated_refrain_sensitivity.csv")]:
            m[field] = json.loads(pd.read_csv(path / filename).to_json(orient="records"))
        for point in m["map"]:
            meta = songs[point["song_id"]]
            point.update(title=meta["title"], artist=meta["artist"], year=meta["year"], language=meta["language"])
        languages = pd.DataFrame(m['map'])
        for t in m['topics']:
            count = languages[languages.topic_id == t['topic_id']].language.value_counts()
            t['language_counts'] = {lang:int(count.get(lang,0)) for lang in ('en','es')}
        # Static map preserves original unsupervised topic IDs. No post hoc emotion names.
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        data = pd.DataFrame(m["map"])
        for j, topic in enumerate(m["topics"]):
            color = "#b9bec5" if topic["topic_id"] == -1 else COLORS[topic["topic_id"] % len(COLORS)]
            for language, marker in [("en", "o"), ("es", "^")]:
                d = data[(data.topic_id == topic["topic_id"]) & (data.language == language)]
                ax.scatter(d.x, d.y, s=22, marker=marker, alpha=.68, color=color,
                           label=title(topic)[:48] if language == "en" else None)
        ax.set_title(f"BERTopic · {m['name']}\n{diag['unique_blocks']} bloques únicos en una proyección exploratoria", loc="left", weight="bold")
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .99))
        save(fig, f"{path.name}-map", "Círculo: EN; triángulo: ES. Clustering en 5D; mapa en 2D con ajuste separado. La distancia visual no valida un subtema.")

        z, leaves = hierarchy_linkage(m["topics"], m["hierarchy"])
        if z is not None and len(leaves) > 1:
            lookup = {t["topic_id"]: t for t in m["topics"]}
            fig, ax = plt.subplots(figsize=(11, max(4.7, len(leaves) * .42)))
            dendrogram(z, labels=[title(lookup[t])[:55] + f"  [{lookup[t]['songs']} canciones]" for t in leaves],
                       orientation="right", color_threshold=0, above_threshold_color=COLORS[0], ax=ax)
            ax.set_title(f"Jerarquía de tópicos · {m['name']}", loc="left", weight="bold")
            ax.set_xlabel("Distancia de fusión c-TF-IDF" if m["hierarchy"]["status"] == "ok" else "Distancia coseno entre centroides")
            fig.tight_layout(rect=(0, .11, 1, .98))
            save(fig, f"{path.name}-hierarchy", "Palabras automáticas; agrupación por similitud. Los nodos no constituyen una taxonomía literaria ni categorías de miedo validadas.")

        composition = pd.DataFrame(m["composition"])
        d = composition[(composition.variant == "unique_first_occurrence") & (composition.language == "balanced_available_languages")]
        displayed = [t["topic_id"] for t in sorted(m["topics"], key=lambda t: -t["unique_blocks"])[:6]]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.1), sharex=True, sharey=True)
        for ax, topic_id in zip(axes.flat, displayed):
            t = next(t for t in m["topics"] if t["topic_id"] == topic_id)
            v = d[d.topic_id == topic_id].sort_values("decade")
            ax.errorbar(range(len(v)), v.song_normalized_share, yerr=[v.song_normalized_share - v.ci_low, v.ci_high - v.song_normalized_share],
                        fmt="o-", capsize=3, lw=1.5, color="#7d8793" if topic_id == -1 else COLORS[topic_id % len(COLORS)])
            ax.set_title(title(t)[:44], loc="left", fontsize=10)
            ax.set(xticks=range(5), xticklabels=["80s", "90s", "00s", "10s", "20–26*"], ylim=(0, 1))
        for ax in list(axes.flat)[len(displayed):]: ax.set_visible(False)
        fig.suptitle(f"Composición de los casos por período · {m['name']}", x=.06, ha="left", weight="bold")
        fig.supylabel("Proporción media por canción")
        fig.tight_layout(rect=(0, .11, 1, .94))
        save(fig, f"{path.name}-decades", "Canciones con igual peso; EN y ES equilibrados; bootstrap de 1.000 muestras dentro de idioma.")
        models.append(m)

    if models:
        # Refresh all model-specific shares from this run, preserving the original
        # interpretive card text and evidence IDs. Topic IDs never transfer runs.
        for card in cards:
            assignments=[]
            for m in models:
                directory=RESULTS/m['slug']
                shares=pd.read_csv(directory/'song_topic_shares_unique_first_occurrence.csv')
                shares=shares[shares.song_id==card['song_id']]
                sections=pd.read_csv(directory/'section_assignments.csv')
                sections=sections[(sections.song_id==card['song_id']) & (~sections.exact_repeat)]
                assignments.append(dict(model_id=m['manifest']['encoder']['model_id'],seed=17,
                    assignment_file=str((directory/'section_assignments.csv').relative_to(ROOT)),
                    topic_ids_present=sorted(sections.topic_id.unique().tolist()),
                    unique_block_counts={str(k):int(v) for k,v in sections.topic_id.value_counts().items()},
                    unique_block_shares={str(r.topic_id):float(r.share) for r in shares.itertuples()},
                    evidence_document_assignments=[dict(document_id=r.document_id,topic_id=int(r.topic_id)) for r in sections.itertuples()]))
            card['primary_bertopic_assignments']=assignments
        if cards:
            fig, axes = plt.subplots(1,len(models),figsize=(13,6),gridspec_kw={'width_ratios':[max(2,len(m['topics'])) for m in models]},squeeze=False)
            for m, ax in zip(models, axes[0]):
                ids=[t['topic_id'] for t in m['topics']]
                values=[]
                for c in cards:
                    assignment=next(a for a in c['primary_bertopic_assignments'] if a['model_id']==m['manifest']['encoder']['model_id'])
                    values.append([assignment['unique_block_shares'].get(str(t),0) for t in ids])
                im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
                ax.set(xticks=range(len(ids)),xticklabels=[str(t)for t in ids],yticks=range(len(cards)),
                       yticklabels=[c['title'] for c in cards] if m==models[0] else [],xlabel='Tópico (-1: sin asignación)')
                ax.set_title(m['name'],loc='left',weight='bold')
                for r,row in enumerate(values):
                    for j,value in enumerate(row):
                        if value>=.1:ax.text(j,r,f'{value:.0%}',ha='center',va='center',fontsize=7,color='white' if value>.65 else '#263341')
            fig.subplots_adjust(left=.23,right=.91,bottom=.16,top=.91,wspace=.10)
            cax=fig.add_axes([.94,.3,.018,.45]);fig.colorbar(im,cax=cax,label='Proporción de bloques únicos')
            save(fig,'reading-cases','Fichas de lectura y particiones automáticas: dos formas distintas de examinar los mismos casos.')
        fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 9), squeeze=False)
        for m, ax in zip(models, axes[0]):
            topics = m['topics']
            en = np.array([t['language_counts']['en'] for t in topics])
            es = np.array([t['language_counts']['es'] for t in topics])
            total = en + es
            y = np.arange(len(topics))
            ax.barh(y, en/total, color=COLORS[0], label='EN')
            ax.barh(y, es/total, left=en/total, color=COLORS[1], label='ES')
            ax.set(yticks=y, yticklabels=[f"T{t['topic_id']} · n={n}" for t,n in zip(topics,total)],
                   xlim=(0,1), xlabel='Composición lingüística del tópico')
            ax.invert_yaxis(); ax.set_title(m['name'],loc='left',weight='bold')
            ax.legend(frameon=False,ncol=2)
        fig.tight_layout(rect=(0,.10,1,.97))
        save(fig,'language-confound','Fracciones de bloques EN/ES en cada tópico; no exactitud temática. E5 agrupa principalmente por idioma en este ajuste.')
        fig, axes = plt.subplots(1, len(models), figsize=(4.5*len(models), 8), sharey=True, squeeze=False)
        for i, (m, ax) in enumerate(zip(models, axes[0])):
            values = np.array([[p["related_cosine"], p["contrast_cosine"]] for p in m["probes"]])
            im = ax.imshow(values, vmin=-.1, vmax=1, cmap="YlGnBu", aspect="auto")
            ax.set(xticks=[0, 1], xticklabels=["Candidato A", "Candidato B"], yticks=range(len(values)),
                   yticklabels=[p["id"] for p in m["probes"]])
            ax.set_title(m["name"], loc="left", weight="bold")
            for r in range(len(values)):
                for c in range(2): ax.text(c, r, f"{values[r,c]:.2f}", ha="center", va="center", fontsize=8,
                                          color="white" if values[r,c] > .72 else "#263341")
        fig.subplots_adjust(left=.25, right=.82, bottom=.12, wspace=.15)
        cax = fig.add_axes([.86,.25,.024,.55])
        fig.colorbar(im, cax=cax, label="Coseno dentro de cada modelo")
        save(fig, "semantic-probes", "20 contrastes sintéticos originales. A no es siempre una respuesta correcta: negación y roles prueban cercanía con significado distinto.")

        fig, axes = plt.subplots(1, len(models), figsize=(5.8*len(models), 4.9), squeeze=False)
        for m, ax in zip(models, axes[0]):
            d = pd.DataFrame(m.get("paths", []))
            if d.empty: continue
            d = d[d.metric == "semantic_contrast"]
            for j, decade in enumerate(sorted(d.decade.unique())):
                v = d[(d.decade == decade) & (d.variant == "unique_first_occurrence")].sort_values("normalized_position")
                ax.plot(v.normalized_position, v['mean'], lw=2, color=COLORS[j], label=str(decade) + "s")
                ax.fill_between(v.normalized_position, v.ci_low, v.ci_high, color=COLORS[j], alpha=.09)
            ax.axhline(0, color="#b9bec5", lw=1)
            ax.set_title(m["name"], loc="left", weight="bold")
            ax.set(xlabel="Posición relativa de los bloques", ylabel="Afrontamiento − (miedo + amenaza) / 2")
            ax.legend(frameon=False, fontsize=8, ncol=3)
        fig.tight_layout(rect=(0, .13, 1, .98))
        save(fig, "semantic-trajectories", "Similitud a prototipos originales, no intensidad emocional. Bandas puntuales bootstrap; orden de bloques, no tiempo de audio.")

    if models:
        # Shared corpus and parameters; each model retains its own coordinate system.
        columns = 2
        rows = (len(models)+1)//2
        fig, axes = plt.subplots(rows, columns, figsize=(13, 5.2*rows), squeeze=False)
        for m, ax in zip(models, axes.flat):
            points = pd.DataFrame(m['map'])
            for language, marker in [('en', 'o'), ('es', '^')]:
                v = points[points.language == language]
                ax.scatter(v.x, v.y, c=v.topic_id, cmap='tab20', s=13, alpha=.65, marker=marker,
                           label=f'{language.upper()} · n={len(v)}')
            ax.set(title=m['name'], xlabel='UMAP 1', ylabel='UMAP 2')
            ax.legend(frameon=False, fontsize=8)
        for ax in list(axes.flat)[len(models):]: ax.axis('off')
        fig.tight_layout(rect=(0,.065,1,.98))
        save(fig, 'embeddings-umap-all', 'Cada punto es un bloque único; color: tópico del modelo, forma: idioma. UMAP muestra vecindarios; sus ejes no miden miedo.')
        with_pca = [m for m in models if m.get('projections')]
        if with_pca:
            fig = plt.figure(figsize=(13, 5.4*((len(with_pca)+1)//2)))
            for i, m in enumerate(with_pca):
                ax = fig.add_subplot((len(with_pca)+1)//2, 2, i+1, projection='3d')
                points = pd.DataFrame(m['projections']['points'])
                if 'year' not in points:
                    points = points.merge(pd.DataFrame(manifest)[['song_id','year']], on='song_id', validate='many_to_one')
                scatter = ax.scatter(points.x, points.y, points.z, c=points.year, cmap='viridis', vmin=1980, vmax=2026, s=6, alpha=.5)
                receipt = m['projections']['receipt']
                ax.set(title=f"{m['name']} · varianza 3 PC {receipt['explained_variance_total']:.1%}", xlabel='PC 1', ylabel='PC 2', zlabel='PC 3')
                ax.view_init(elev=20, azim=40)
            fig.colorbar(scatter, ax=fig.axes, shrink=.5, pad=.02, label='Año de publicación')
            fig.subplots_adjust(left=.03,right=.87,bottom=.09,top=.95,wspace=.08,hspace=.18)
            save(fig, 'embeddings-pca3-all', 'PCA global por modelo, con ejes fijos entre años. Color: publicación. No se ajusta un PCA distinto para cada período.')
            fig, axes = plt.subplots(1,2,figsize=(12,4.9))
            for i,m in enumerate(with_pca):
                receipt=m['projections']['receipt']
                spectrum=receipt.get('scree_explained_variance_ratio', receipt['explained_variance_ratio'])
                components=np.arange(1,len(spectrum)+1)
                axes[0].plot(components,np.asarray(spectrum)*100,'o-',ms=3,label=m['name'],color=COLORS[i])
                axes[1].plot(components,np.cumsum(spectrum)*100,'o-',ms=3,label=m['name'],color=COLORS[i])
                axes[1].scatter(3,receipt['explained_variance_total']*100,s=50,color=COLORS[i])
            axes[0].set(xlabel='Componente principal',ylabel='Varianza explicada (%)',title='Qué conserva cada componente')
            axes[1].set(xlabel='Número de componentes',ylabel='Varianza acumulada (%)',title='Cuánta información conserva la proyección')
            axes[1].axvline(3,color='#a5adb6',ls='--',lw=1)
            axes[1].legend(frameon=False,fontsize=8)
            fig.tight_layout(rect=(0,.12,1,.97))
            save(fig,'embedding-variance','Varianza de los bloques L2 normalizados, centrados globalmente. Más varianza en tres ejes no implica mejor lectura temática.')

    # Original diagram: retrieval context is explicit and total evidence budgets constrain retries.
    fig, ax = plt.subplots(figsize=(12, 5.1)); ax.set(xlim=(0, 12), ylim=(0, 5)); ax.axis("off")
    boxes = [(0.2, 3, "1 · Pregunta", "Idioma, período y alcance"), (3.2, 3, "2 · Recuperación", "BM25 + denso → RRF"),
             (6.2, 3, "3 · Suficiencia", "Evidencia · presupuesto"), (9.2, 3, "4 · Contexto", "Bloques vecinos · voz"),
             (6.2, .9, "Reformular / abstenerse", "Máximo 3 búsquedas"), (9.2, .9, "5 · Respuesta y revisión", "Afirmación ↔ IDs ↔ fuente")]
    for x, y, head, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.45, 1.15, boxstyle="round,pad=.08", facecolor="#eaf1f2", edgecolor="#90b3ba"))
        ax.text(x+.15, y+.80, head, weight="bold", fontsize=10)
        ax.text(x+.15, y+.33, sub, fontsize=9)
    for a, b in [((2.75,3.58),(3.1,3.58)),((5.75,3.58),(6.1,3.58)),((8.75,3.58),(9.1,3.58)),
                 ((7.4,2.9),(7.4,2.15)),((10.4,2.9),(10.4,2.15)),((8.75,1.45),(9.1,1.45)),((6.1,1.45),(4.4,2.9))]:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle":"->", "color":"#147d92", "lw":1.6})
    ax.text(.2,4.55,"LangGraph · recuperación con contexto y límites verificables",fontsize=16,weight="bold")
    save(fig, "langgraph-workflow", "Grafo de recuperación ejecutado; lectura LLM sobre contexto congelado, validación de IDs y salida estructurada.")

    rag = []
    for directory in ['rag','rag-e5','rag-openai-large','rag-openai-small']:
        ragfile = RESULTS / directory / 'summary.json'
        if ragfile.exists():
            rag.extend([dict(**row,model={'rag':'MiniLM','rag-e5':'E5','rag-openai-large':'OpenAI large','rag-openai-small':'OpenAI small'}[directory]) for row in read(ragfile)])
    if rag:
        table = pd.DataFrame(rag)
        table = table[table.split=='test'].groupby(['model','method','arm'],as_index=False).mean(numeric_only=True)
        fig, axes = plt.subplots(1,2,figsize=(11.8,5.2))
        records = table.sort_values(['model','method','arm'])
        labels = [f"{r.model} · {r.method}\n{'Contexto' if r.arm=='bounded_contextual' else 'Una pasada'}" for r in records.itertuples()]
        for ax, metric, heading in [(axes[0],'recall_at_5','Evidencia recuperada · 16 preguntas positivas'),
                                   (axes[1],'negative_safe_abstention_rate','Abstención · 8 preguntas negativas')]:
            x=np.arange(len(records))
            ax.bar(x,records[metric],color=[COLORS[0] if a=='one_pass' else COLORS[1] for a in records.arm])
            ax.set(xticks=x,xticklabels=labels,ylim=(0,1.05));ax.tick_params(axis='x',labelrotation=90,labelsize=8)
            ax.set_title(heading,loc='left',weight='bold',fontsize=11)
            ax.set_ylabel('Proporción')
        fig.tight_layout(rect=(0,.16,1,.98))
        save(fig,'rag-recall-safety','Test del analista; media EN/ES. Más recall puede coexistir con menos abstención correcta. Controlador de reglas, sin generación LLM.')
    bayesfile = RESULTS / 'bayes/posterior_rates.csv'
    bayes = json.loads(pd.read_csv(bayesfile).to_json(orient='records')) if bayesfile.exists() else []
    for ext in ['svg', 'png']:
        source = RESULTS / 'bayes' / f'beta_binomial_presence.{ext}'
        if source.exists(): shutil.copy2(source, FIG / source.name)
    labelfile = RESULTS / 'topic_labels.json'
    labels = read(labelfile) if labelfile.exists() else []
    arc_directory=HERE/'pilot-comparison/results/arc-panel'
    if not (arc_directory/'song_annotations.json').exists(): arc_directory=RESULTS/'arc-panel'
    arc_panel={}
    if (arc_directory/'song_annotations.json').exists() and {r['song_id'] for r in read(arc_directory/'song_annotations.json')} == set(songs):
        for key,filename in [('annotations','song_annotations.json'),('votes','panel_votes.json'),('summary','summary.json'),('ontology','ontology.json')]:
            if (arc_directory/filename).exists(): arc_panel[key]=read(arc_directory/filename)
        fieldwise=arc_directory/'fieldwise-posthoc'
        if (fieldwise/'song_annotations.json').exists():
            arc_panel['primary_annotations']=arc_panel['annotations']
            arc_panel['primary_summary']=arc_panel['summary']
            arc_panel['annotations']=read(fieldwise/'song_annotations.json')
            arc_panel['summary']=read(fieldwise/'summary.json')
            arc_panel['categorical_diagnostic']=True
            for row in arc_panel['annotations']:
                row['valid_votes']=min(row.get('valid_votes_by_field',{}).values(),default=0)
            if (fieldwise/'anchored_cases.json').exists(): arc_panel['anchored_cases']=read(fieldwise/'anchored_cases.json')
    payload = dict(expanded_atlas_url='/research/fear-lyrics/expanded/atlas.html' if RESULTS==HERE/'results' else None,
                   collection_songs=read(HERE/'corpus-manifest.json'), arc_panel=arc_panel, bayes=bayes, topic_labels=labels, corpus=diag, songs=manifest, cards=cards, coverage=json.loads(coverage.to_json(orient="records")),
                   agreement=json.loads(agreements.to_json(orient="records")), models=models, rag=rag)
    (SITE / "atlas-data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    (SITE / "plotly.min.js").write_text(get_plotlyjs())
    (SITE / "atlas.html").write_text(HTML.replace("384 bloques únicos", f"{diag['unique_blocks']} bloques únicos"), encoding="utf-8")
    for pattern in ('*.svg','*.png'):
        for file in FIG.glob(pattern): shutil.copy2(file, SITE / file.name)
    print(json.dumps({"models":len(models), "static_figures":len(list(FIG.glob('*.svg'))), "atlas":str(SITE / 'atlas.html')}))

HTML = r'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Atlas del miedo en las letras</title>
<style>body{margin:0;background:#fbfaf7;color:#263341;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}h1{font-size:clamp(28px,5vw,46px);line-height:1.12;max-width:900px}h2{margin-top:38px;line-height:1.3}p{max-width:850px}select{font:inherit;padding:9px;background:white;border:1px solid #97a5b2;border-radius:6px}.plot{height:520px;border:1px solid #dce1e4;border-radius:10px;margin-top:15px;background:white}.note{font-size:14px;color:#667181}.stats{display:flex;flex-wrap:wrap;gap:10px}.stats b{padding:12px 18px;background:#eaf1f2;border-radius:8px}a{color:#147d92}.probe{font-size:14px;max-width:950px;min-height:70px;padding:14px;background:#edf1f3}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}@media(max-width:600px){.plot{height:440px}main{padding:20px 14px}}</style>
<main><a href="/post/2026-10-11-miedo-embeddings-bertopic-rag/">Artículo del estudio</a><h1>Qué conservan los embeddings cuando una canción habla de miedo</h1>
<p>Explora 40 canciones en inglés y español, de 1980 al 11 de octubre de 2026. Cambiar de representación puede cambiar los grupos. Este atlas muestra resultados del corpus seleccionado; no estima cómo ha evolucionado toda la música popular.</p>
<div class="stats"><b>40 canciones</b><b>34 artistas</b><b>384 bloques únicos</b><b>4 EN + 4 ES por período</b></div>
<h2>Una representación cada vez</h2><select id="model" aria-label="Modelo de embeddings"></select><p id="status" class="note"></p>
<h2>Mapa de tópicos y canciones</h2><p class="note">Cada punto es un bloque. Color: tópico; símbolo: idioma. El clustering se ajusta en 5D y este mapa usa otra proyección a 2D. La forma de las islas no demuestra una taxonomía del miedo.</p><div id="map" class="plot"></div>
<h2>Subgrupos con procedencia</h2><p class="note">Descriptores automáticos c-TF-IDF. Selecciona una barra para identificar el vocabulario del grupo. La concentración en pocos artistas limita la generalización.</p><div id="topics" class="plot"></div>
<img id="hierarchy" style="max-width:100%;margin-top:24px" alt="Jerarquía de similitud de tópicos"><h2>Comparar períodos sin contar el estribillo varias veces</h2><p class="note">Media de proporciones por canción, equilibrada EN/ES. Los intervalos remuestrean estas canciones, no una población de oyentes. El bloque final llega solo hasta octubre de 2026.</p><select id="topic" aria-label="Tópico para comparación temporal"></select><div id="decades" class="plot"></div>
<h2>Pruebas de significado</h2><p class="note">Frases originales. La cercanía a una negación no demuestra que el modelo haya interpretado su polaridad. Haz clic en una fila para leer el contraste.</p><div id="probes" class="plot" style="height:700px"></div><p class="probe" id="probeText">Selecciona una fila del mapa.</p>
<h2>Leer situaciones y contralecturas</h2><p class="note">Diez fichas propuestas por el asistente, sin validación humana independiente. Los subtemas son lecturas tentativas; no son etiquetas descubiertas por BERTopic.</p><select id="card" aria-label="Canción para leer"></select><div id="cardText"></div><h2>Trayectorias de representación</h2><p class="note">Afrontamiento menos la media de miedo y amenaza. Es una diferencia entre similitudes a prototipos, no una escala de emoción ni un arco argumental validado.</p><div id="arcs" class="plot"></div>
<h2>Workflow de recuperación</h2><img src="langgraph-workflow.svg" alt="Pregunta, recuperación híbrida, suficiencia, contexto, reformulación o abstención, respuesta y revisión" style="width:100%"><div id="rag"></div>
<p class="note">No se distribuyen las letras. El manifiesto, los hashes, las fuentes y los resultados permiten auditar procedencia y cálculos. Las etiquetas iniciales de selección son hipótesis del analista.</p></main>
<script src="plotly.min.js"></script><script>
const colors=['#147d92','#d46b40','#8e6ba5','#4c956c','#bb8d22','#c25279','#5877b8','#6e8b87'];
const base={paper_bgcolor:'#fbfaf7',plot_bgcolor:'#fbfaf7',font:{family:'system-ui',color:'#263341',size:13},margin:{l:60,r:30,t:30,b:65},hoverlabel:{font:{size:13}},legend:{orientation:'h',y:-.2}};
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};let data,m;
const name=t=>t.topic_id===-1?'Sin asignación (-1)':'T'+t.topic_id+' · '+t.terms.slice(0,3).map(x=>x.term).join(' · ');
fetch('atlas-data.json').then(r=>r.json()).then(d=>{data=d;document.getElementById('model').innerHTML=d.models.map((x,i)=>`<option value="${i}">${x.name}</option>`).join('');document.getElementById('model').onchange=render;document.getElementById('card').innerHTML=(d.cards||[]).map((c,i)=>`<option value="${i}">${c.title} · ${c.proposed_subtopic.label}</option>`).join('');document.getElementById('card').onchange=showCard;showCard();render();});
function render(){m=data.models[+document.getElementById('model').value];document.getElementById('hierarchy').src=m.slug+'-hierarchy.svg';const nonnoise=m.topics.filter(t=>t.topic_id!==-1),noise=m.topics.find(t=>t.topic_id===-1);document.getElementById('status').textContent=`${nonnoise.length} tópicos; ${noise?noise.unique_blocks:0} bloques sin asignación. ${m.manifest.truncated_documents} bloques exceden la ventana del modelo. Semilla principal: 17.`;
const traces=[];for(const t of m.topics)for(const lang of ['en','es']){const pts=m.map.filter(p=>p.topic_id===t.topic_id&&p.language===lang);traces.push({x:pts.map(p=>p.x),y:pts.map(p=>p.y),type:'scatter',mode:'markers',name:name(t)+' '+lang.toUpperCase(),marker:{size:8,opacity:.75,color:t.topic_id===-1?'#bcc2c9':colors[t.topic_id%colors.length],symbol:lang==='en'?'circle':'triangle-up'},text:pts.map(p=>`${p.title} · ${p.artist}<br>${p.year} · ${p.language.toUpperCase()}<br>${p.document_id}`),hovertemplate:'%{text}<extra></extra>'});}
Plotly.react('map',traces,{...base,showlegend:false,xaxis:{title:'UMAP 1',zeroline:false},yaxis:{title:'UMAP 2',zeroline:false}},config);
Plotly.react('topics',[{type:'bar',orientation:'h',x:m.topics.map(t=>t.unique_blocks),y:m.topics.map(name),marker:{color:m.topics.map(t=>t.topic_id===-1?'#bcc2c9':colors[t.topic_id%colors.length])},customdata:m.topics.map(t=>{let c=m.concentration.find(x=>x.topic_id===t.topic_id);return [t.songs,c?.contributing_artists,c?.largest_artist,c?.largest_artist_share,t.terms.map(x=>x.term).join(' · ')];}),hovertemplate:'%{x} bloques · %{customdata[0]} canciones<br>%{customdata[1]} artistas · mayor: %{customdata[2]}<br>Cuota del mayor: %{customdata[3]:.1%}<br>%{customdata[4]}<extra></extra>'}],{...base,margin:{l:220,r:30,t:30,b:60},xaxis:{title:'Bloques únicos'},yaxis:{autorange:'reversed'}},config);
document.getElementById('topic').innerHTML=m.topics.map(t=>`<option value="${t.topic_id}">${name(t)}</option>`).join('');document.getElementById('topic').onchange=decades;decades();
Plotly.react('probes',[{type:'heatmap',z:m.probes.map(p=>[p.related_cosine,p.contrast_cosine]),x:['Candidato A','Candidato B'],y:m.probes.map(p=>p.id),zmin:-.1,zmax:1,colorscale:'YlGnBu',hovertemplate:'%{y} · %{x}<br>coseno %{z:.3f}<extra></extra>'}],{...base,margin:{l:200,r:30,t:30,b:60},yaxis:{autorange:'reversed'},xaxis:{side:'top'}},config);
document.getElementById('probes').removeAllListeners?.('plotly_click');document.getElementById('probes').on('plotly_click',e=>{const p=m.probes.find(p=>p.id===e.points[0].y);document.getElementById('probeText').textContent=`Consulta: ${p.query.slice(0,600)} | A: ${p.related.slice(0,300)} | B: ${p.contrast.slice(0,300)}. ${p.interpretation}`;});
const arcs=[];for(const decade of [1980,1990,2000,2010,2020]){const v=m.paths.filter(p=>p.decade===decade&&p.variant==='unique_first_occurrence'&&p.metric==='semantic_contrast').sort((a,b)=>a.normalized_position-b.normalized_position);arcs.push({type:'scatter',mode:'lines+markers',name:decade===2020?'2020–2026*':decade+'s',x:v.map(p=>p.normalized_position),y:v.map(p=>p.mean),error_y:{type:'data',symmetric:false,array:v.map(p=>p.ci_high-p.mean),arrayminus:v.map(p=>p.mean-p.ci_low),thickness:.8,width:2},hovertemplate:'Posición %{x:.2f}<br>Contraste %{y:.3f}<extra>%{fullData.name}</extra>'});}Plotly.react('arcs',arcs,{...base,xaxis:{title:'Posición relativa del bloque'},yaxis:{title:'Contraste entre similitudes'}},config);
const rows=(data.rag||[]).filter(x=>x.split==='test');document.getElementById('rag').innerHTML='<table><tr><th>Modelo</th><th>Método</th><th>Brazo</th><th>Idioma</th><th>Recall@5</th><th>Recall contexto</th><th>Abstención negativas</th></tr>'+rows.map(r=>`<tr><td>${r.model}</td><td>${r.method}</td><td>${r.arm==='one_pass'?'Una pasada':'Contexto'}</td><td>${r.language.toUpperCase()}</td><td>${r.recall_at_5.toFixed(3)}</td><td>${r.context_support_recall.toFixed(3)}</td><td>${r.negative_safe_abstention_rate.toFixed(3)}</td></tr>`).join('')+'</table><p class=note>8 positivas y 4 negativas test por idioma; controller de reglas, sin LLM. Recuperación y supervivencia en contexto son métricas distintas.</p>'; 
}
function showCard(){const c=data.cards?.[+document.getElementById('card').value];if(!c)return;const box=document.getElementById('cardText');box.replaceChildren();for(const [label,value] of [['Situación',c.situation],['Voz',c.voice],['Postura',c.stance],['Movimiento',c.narrative_movement],['Límite',c.interpretive_limit]]){const p=document.createElement('p');p.textContent=label+': '+(typeof value==='string'?value:Object.values(value).join(' · '));box.appendChild(p);}for(const item of [...c.support,...c.counterevidence]){const p=document.createElement('p');p.className='note';p.textContent=(item.interpretation||item.reason||JSON.stringify(item))+' · '+(item.document_ids||[]).join(', ');box.appendChild(p);}}
function decades(){const topic=+document.getElementById('topic').value;const lines=[];for(const [variant,label]of [['unique_first_occurrence','Bloques únicos'],['all_blocks_with_repeats','Con repeticiones']]){const v=m.composition.filter(x=>x.topic_id===topic&&x.language==='balanced_available_languages'&&x.variant===variant).sort((a,b)=>a.decade-b.decade);lines.push({type:'scatter',mode:'lines+markers',name:label,x:v.map(x=>x.decade===2020?'2020–2026*':x.decade+'s'),y:v.map(x=>x.song_normalized_share),error_y:{type:'data',symmetric:false,array:v.map(x=>x.ci_high-x.song_normalized_share),arrayminus:v.map(x=>x.song_normalized_share-x.ci_low)},hovertemplate:'%{x}<br>Proporción %{y:.1%}<extra>%{fullData.name}</extra>'});}Plotly.react('decades',lines,{...base,yaxis:{title:'Proporción media por canción',range:[0,1],tickformat:'.0%'}},config);}
</script></html>'''

if __name__ == "__main__": build()
