"""Crítica de Aquamosh: scrape extendido + topic modeling a nivel oración + análisis comparativo.

Marco:
- Solo 2 reseñas-críticas reales + 2 framings enciclopédicos.
- Estrategia: segmentar en oraciones (n≈80-120) y clusterizar a nivel oración.
- Cada cluster lo nombra GPT-4o-mini.
- Comparación con los hallazgos del análisis del álbum (idioma×campo, axes).
"""
import os, re, json, time, requests
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from openai import OpenAI

for p in [Path.cwd(), *Path.cwd().parents][:5]:
    if (p / ".env").exists():
        load_dotenv(p / ".env"); break

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.5 Safari/605.1.15")

# ========== 1) corpus extendido ==========
df_existing = pd.read_parquet("outputs/exports/corpus_critics.parquet")
# Drop AllMusic (solo menú scrapeado, sin contenido real)
df_existing = df_existing[df_existing["source"] != "AllMusic"].reset_index(drop=True)
print(f"Corpus original conservado: {len(df_existing)} fuentes")

extra_rows = []
for src in [
    ("Wikipedia ES", "https://es.wikipedia.org/wiki/Aquamosh", "div.mw-parser-output p"),
    ("Wikipedia EN", "https://en.wikipedia.org/wiki/Aquamosh", "div.mw-parser-output p"),
]:
    name, url, sel = src
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = "\n".join(n.get_text(" ", strip=True)
                              for n in soup.select(sel)
                              if len(n.get_text(strip=True)) > 30)
            if len(text) > 100:
                extra_rows.append({"source": name, "url": url, "text": text,
                                    "text_length": len(text),
                                    "score": None, "reviewer": None,
                                    "date": None, "language": "es" if "es." in url else "en"})
                print(f"  + {name}: {len(text)} chars")
    except Exception as e:
        print(f"  - {name}: {e}")
    time.sleep(1)

df = pd.concat([df_existing, pd.DataFrame(extra_rows)], ignore_index=True)
print(f"\nCorpus extendido: {len(df)} fuentes")
print(df[["source", "text_length", "language"]].to_string(index=False))

# ========== 2) segmentación en oraciones ==========
def sentence_segment(text: str, source: str, lang: str) -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    if lang == "en":
        # split en EN simple
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    else:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", text)
    out = []
    for s in sentences:
        s = s.strip()
        if 25 <= len(s) <= 400 and len(s.split()) >= 5:
            out.append({"source": source, "text": s, "lang": lang})
    return out

sent_rows = []
for _, r in df.iterrows():
    sent_rows.extend(sentence_segment(r["text"], r["source"], r["language"]))
df_sent = pd.DataFrame(sent_rows)
print(f"\nOraciones extraídas: {len(df_sent)}")
print(df_sent["source"].value_counts())

# ========== 3) embeddings ==========
print("\nGenerando embeddings OpenAI...")
emb_path = Path("data/embeddings/openai_critics_sentences.npy")
if emb_path.exists():
    emb = np.load(emb_path)
    print(f"Cache: {emb.shape}")
else:
    out = []
    texts = df_sent["text"].tolist()
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        r = client.embeddings.create(model="text-embedding-3-large",
                                       input=batch, dimensions=1024)
        out.extend([d.embedding for d in r.data])
    emb = np.array(out, dtype=np.float32)
    np.save(emb_path, emb)
    print(f"Guardado: {emb.shape}")

# Normalizar
emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

# ========== 4) clustering por silhouette ==========
print("\nClustering por silhouette + criterio de interpretabilidad...")
silhouettes = {}
for k in range(2, min(9, len(emb_n)-1)):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(emb_n)
    s = silhouette_score(emb_n, labels, metric="cosine")
    silhouettes[k] = s
    print(f"  k={k}: silhouette={s:.3f}")

# Las diferencias entre k son pequeñas (~0.02). Fijamos k=4 por interpretabilidad:
# las reseñas son pocas, sobre-segmentar produce clusters redundantes.
best_k = 4
print(f"  -> Elegimos k = {best_k}  (silhouette={silhouettes[best_k]:.3f}) por interpretabilidad")

km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df_sent["cluster"] = km.fit_predict(emb_n)

# ========== 5) nombrar clusters con LLM ==========
print("\nNombrando clusters...")
cluster_info = []
for c in sorted(df_sent["cluster"].unique()):
    sub = df_sent[df_sent["cluster"] == c]
    # Centroide: oraciones más cercanas al centro del cluster
    centroid = emb_n[df_sent["cluster"] == c].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    sims = emb_n[df_sent["cluster"] == c] @ centroid
    top_idx = np.argsort(-sims)[:6]
    top_sents = sub.iloc[top_idx]["text"].tolist()

    prompt = (
        "Eres analista musical. Lee estas oraciones de reseñas/notas sobre el "
        "álbum 'Aquamosh' (Plastilina Mosh, 1998) que un clustering agrupó "
        "juntas. Identifica el tema común que las une.\n\n"
        "Oraciones:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(top_sents)) +
        "\n\nResponde SOLO con JSON con este formato:\n"
        '{"nombre": "<2-5 palabras>", "descripcion": "<una frase>", '
        '"keywords": ["...", "..."]}'
    )

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=300,
    )
    try:
        info = json.loads(r.choices[0].message.content)
    except Exception:
        info = {"nombre": f"Cluster {c}", "descripcion": "—", "keywords": []}

    info["cluster_id"] = int(c)
    info["n"] = int(len(sub))
    info["sources"] = sub["source"].value_counts().to_dict()
    info["top_sentences"] = top_sents[:4]
    cluster_info.append(info)
    print(f"  Cluster {c} (n={len(sub)}): {info['nombre']}")
    print(f"    {info['descripcion']}")

# Volcar a JSON (sanitizar tipos numpy)
def _to_py(o):
    import numpy as _np
    if isinstance(o, (_np.floating, _np.integer)): return o.item()
    if isinstance(o, _np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_py(x) for x in o]
    return o

findings = _to_py({
    "corpus_size": {"n_sources": len(df), "n_sentences": len(df_sent)},
    "sources": df[["source", "text_length", "language"]].to_dict("records"),
    "silhouettes": silhouettes,
    "k_chosen": best_k,
    "clusters": cluster_info,
})
Path("outputs/exports/critics_topics.json").write_text(
    json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[json] outputs/exports/critics_topics.json")
df_sent.to_parquet("outputs/exports/critics_sentences.parquet", index=False)

# ========== 6) visualizaciones ==========

# 6.1 — Distribución de oraciones por fuente × cluster
fig, ax = plt.subplots(figsize=(11, 5))
ct = pd.crosstab(df_sent["source"], df_sent["cluster"])
ct.columns = [next(c["nombre"] for c in cluster_info if c["cluster_id"] == k)
                for k in ct.columns]
ct.plot(kind="barh", stacked=True, ax=ax, colormap="viridis", edgecolor="white")
ax.set_xlabel("Número de oraciones")
ax.set_ylabel("")
ax.set_title("De qué habla cada fuente: tópicos × reseña", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/critics_topics_by_source.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] critics_topics_by_source.png")

# 6.2 — Proyección PCA con clusters
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(emb_n)
fig, ax = plt.subplots(figsize=(11, 8))
palette = sns.color_palette("husl", best_k)
for c in range(best_k):
    mask = df_sent["cluster"] == c
    info = next(i for i in cluster_info if i["cluster_id"] == c)
    ax.scatter(proj[mask, 0], proj[mask, 1], c=[palette[c]],
                s=60, alpha=0.8, edgecolors="white", linewidth=0.4,
                label=f"{info['nombre']} ({info['n']})")
# Marcar fuente con marcador distinto
src_markers = {"Ink19": "o", "AlbumOfTheYear": "s",
                "Wikipedia ES": "^", "Wikipedia EN": "v"}
# Replot por fuente para sobreescribir con marcadores distintos
for src, marker in src_markers.items():
    mask = df_sent["source"] == src
    if mask.sum() == 0: continue
    ax.scatter(proj[mask, 0], proj[mask, 1], facecolors="none",
                edgecolors="white", linewidth=1.2, s=85, marker=marker)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"Topic-space de la crítica (PCA · k={best_k})",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           fontsize=9, loc="best")
plt.tight_layout()
plt.savefig("outputs/figures/critics_topics_pca.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] critics_topics_pca.png")

# 6.3 — Distancia semántica entre fuentes (mapa de calor)
src_centroids = {}
for src in df_sent["source"].unique():
    mask = df_sent["source"] == src
    src_centroids[src] = emb_n[mask].mean(axis=0)
    src_centroids[src] /= (np.linalg.norm(src_centroids[src]) + 1e-12)

src_names = list(src_centroids.keys())
sim_matrix = np.zeros((len(src_names), len(src_names)))
for i, s1 in enumerate(src_names):
    for j, s2 in enumerate(src_names):
        sim_matrix[i, j] = float(src_centroids[s1] @ src_centroids[s2])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="magma",
             xticklabels=src_names, yticklabels=src_names, ax=ax,
             vmin=0.4, vmax=1.0,
             cbar_kws={"label": "Cosine similarity (centroide)"})
ax.set_title("Distancia semántica entre fuentes críticas", color="white", fontsize=13)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/figures/critics_source_similarity.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] critics_source_similarity.png")

# ========== 7) qué temas faltan en la crítica vs el álbum ==========
# El álbum tiene los campos: CUERPO, MARCA, LUGAR, EMOCION, IDENTIDAD, ACCION,
# REFERENCIA, NONSENSE. La crítica habla de OTROS temas. Comparemos.

# Embeddings de las 8 categorías del álbum
field_descriptions = {
    "CUERPO": "fisicalidad, sexualidad, cuerpo, movimiento, baile",
    "MARCA": "productos, marcas comerciales, consumo, K-Mart, publicidad",
    "LUGAR": "geografía, ciudades, México, América, frontera, Monterrey",
    "EMOCION": "sentimientos, afecto, amor, nostalgia, dolor",
    "IDENTIDAD": "pertenencia, origen, regio, latino, identidad cultural",
    "ACCION": "eventos, hacer, dinámica, energía, ritmo",
    "REFERENCIA": "nombres propios, citas culturales, Woody Allen, samples",
    "NONSENSE": "onomatopeyas, frases sin sentido, ruido, scratching",
}
descs = list(field_descriptions.values())
fields = list(field_descriptions.keys())

field_embs = client.embeddings.create(
    model="text-embedding-3-large", input=descs, dimensions=1024
).data
field_embs = np.array([d.embedding for d in field_embs], dtype=np.float32)
field_embs = field_embs / (np.linalg.norm(field_embs, axis=1, keepdims=True) + 1e-12)

# Para cada cluster de crítica, su similaridad media con cada campo del álbum
cluster_field_sim = np.zeros((best_k, len(fields)))
for c in range(best_k):
    mask = df_sent["cluster"] == c
    centroid = emb_n[mask].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)
    for fi, fe in enumerate(field_embs):
        cluster_field_sim[c, fi] = float(centroid @ fe)

cluster_names = [next(c["nombre"] for c in cluster_info if c["cluster_id"] == k)
                  for k in range(best_k)]
fig, ax = plt.subplots(figsize=(10, max(5, 0.6 * best_k)))
sns.heatmap(cluster_field_sim, annot=True, fmt=".2f", cmap="viridis",
             xticklabels=fields, yticklabels=cluster_names, ax=ax,
             cbar_kws={"label": "cosine similarity"})
ax.set_title("Topics de la crítica × campos semánticos del álbum",
              color="white", fontsize=13)
ax.set_xlabel("Campo del álbum"); ax.set_ylabel("Topic de la crítica")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/figures/critics_x_album_fields.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] critics_x_album_fields.png")

# Cuál es el "campo del álbum" sobre el que la crítica habla MENOS
critic_attention_per_field = cluster_field_sim.max(axis=0)  # mejor match por campo
field_attention = dict(zip(fields, critic_attention_per_field.round(3).tolist()))
print(f"\nAtención de la crítica por campo del álbum (más alto = más cubierto):")
for k, v in sorted(field_attention.items(), key=lambda x: -x[1]):
    print(f"  {k:11s} {v:.3f}")

# Guardar la matriz
findings["album_field_coverage"] = _to_py(field_attention)
findings["clusters_x_album_fields_matrix"] = _to_py(cluster_field_sim.round(3).tolist())
findings["clusters_x_album_fields_cluster_order"] = cluster_names
findings["clusters_x_album_fields_field_order"] = fields
Path("outputs/exports/critics_topics.json").write_text(
    json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n--- DONE ---")
