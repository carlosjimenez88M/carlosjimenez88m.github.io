"""Supervivencia diferencial: ¿es Aquamosh excepcional cuando lo comparamos
con sus contemporáneos regiomontanos y nacionales mexicanos de 1996-1999?

Cinco álbumes en el contrafactual:
  - Plastilina Mosh — Aquamosh (1998, Mty)
  - Café Tacuba — Revés/Yo Soy (1999, México)
  - Control Machete — Mucho Barato (1996, Mty)
  - Molotov — ¿Dónde Jugarán las Niñas? (1997, México)
  - Zurdok — Hombre Sintetizador (1999, Mty)

Métricas (todas públicas, gratuitas):
  - Discogs: community rating, # have, # want, num_for_sale, precio mínimo
  - Wikipedia: pageviews mensuales 2015-2026 (API REST)
  - Google Trends: interés del artista 2004-2026 (pytrends)
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.5 Safari/605.1.15 aquamosh-research/1.0")

# === álbumes del contrafactual =======================================
ALBUMS = [
    {"key": "aquamosh",
     "artist": "Plastilina Mosh", "album": "Aquamosh", "year": 1998,
     "discogs_master": None, "discogs_release": 98687,
     "wiki_en": "Aquamosh", "wiki_es": "Aquamosh",
     "trends_query": "Plastilina Mosh"},
    {"key": "reves",
     "artist": "Café Tacuba", "album": "Revés/Yo Soy", "year": 1999,
     "discogs_master": 14655, "discogs_release": None,
     "wiki_en": "Revés/Yo Soy", "wiki_es": "Revés/Yo Soy",
     "trends_query": "Café Tacuba"},
    {"key": "control_machete",
     "artist": "Control Machete", "album": "Mucho Barato", "year": 1996,
     "discogs_master": 178007, "discogs_release": None,
     "wiki_en": "Mucho Barato", "wiki_es": "Mucho Barato",
     "trends_query": "Control Machete"},
    {"key": "molotov",
     "artist": "Molotov", "album": "¿Dónde Jugarán las Niñas?", "year": 1997,
     "discogs_master": 23700, "discogs_release": None,
     "wiki_en": "¿Dónde Jugarán las Niñas?",
     "wiki_es": "¿Dónde jugarán las niñas?",
     "trends_query": "Molotov banda"},
    {"key": "zurdok",
     "artist": "Zurdok", "album": "Hombre Sintetizador", "year": 1999,
     "discogs_master": None, "discogs_release": None,
     "wiki_en": "Hombre Sintetizador",
     "wiki_es": "Hombre sintetizador",
     "trends_query": "Zurdok"},
]

# === 1) Discogs: community stats por release =========================
print("=" * 70)
print("1) DISCOGS — Community stats por álbum")
print("=" * 70)

def discogs_search_release(artist, album):
    """Busca un release_id por artist+album si no lo tenemos cacheado."""
    url = "https://api.discogs.com/database/search"
    r = requests.get(url, params={"artist": artist, "release_title": album,
                                    "type": "release"},
                      headers={"User-Agent": UA}, timeout=15)
    if r.status_code == 200:
        results = r.json().get("results", [])
        return results[0]["id"] if results else None
    return None

def discogs_release(release_id):
    cache = Path(f"data/raw/discogs_{release_id}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://api.discogs.com/releases/{release_id}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        time.sleep(1.5)
        return data
    return {}

def discogs_master(master_id):
    cache = Path(f"data/raw/discogs_master_{master_id}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://api.discogs.com/masters/{master_id}"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        time.sleep(1.5)
        return data
    return {}

discogs_rows = []
for alb in ALBUMS:
    print(f"\n  {alb['album']} ({alb['artist']}, {alb['year']})")
    # Buscar release o master
    release_id = alb.get("discogs_release")
    master_id = alb.get("discogs_master")
    if not release_id and not master_id:
        rid = discogs_search_release(alb["artist"], alb["album"])
        if rid:
            release_id = rid
            print(f"    encontrado release_id={rid}")
            time.sleep(1.5)

    data = {}
    if master_id:
        master = discogs_master(master_id)
        if master and master.get("main_release"):
            data = discogs_release(master["main_release"])
    elif release_id:
        data = discogs_release(release_id)

    if not data:
        print(f"    sin datos en Discogs")
        discogs_rows.append({"key": alb["key"], "discogs_have": None,
                              "discogs_want": None, "discogs_rating": None,
                              "discogs_rating_n": None, "discogs_lowest_price": None,
                              "discogs_for_sale": None})
        continue
    community = data.get("community", {})
    rating = community.get("rating", {})
    discogs_rows.append({
        "key": alb["key"],
        "discogs_have": community.get("have"),
        "discogs_want": community.get("want"),
        "discogs_rating": rating.get("average"),
        "discogs_rating_n": rating.get("count"),
        "discogs_lowest_price": data.get("lowest_price"),
        "discogs_for_sale": data.get("num_for_sale"),
        "discogs_release_id": data.get("id"),
    })
    print(f"    rating: {rating.get('average')}/5 (n={rating.get('count')})  "
          f"have: {community.get('have')}  want: {community.get('want')}  "
          f"for-sale: {data.get('num_for_sale')} @ ${data.get('lowest_price')}")

df_discogs = pd.DataFrame(discogs_rows)
print("\n" + df_discogs.to_string(index=False))

# === 2) Wikipedia pageviews ==========================================
print("\n" + "=" * 70)
print("2) WIKIPEDIA — Pageviews mensuales 2015-2026")
print("=" * 70)

WIKI_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "{project}/all-access/all-agents/{article}/monthly/{start}/{end}")

def wiki_pageviews(project: str, article: str, start="20150701", end="20260520"):
    cache = Path(f"data/raw/wiki_pageviews_{project}_{article.replace('/', '_')}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = WIKI_API.format(project=project, article=article, start=start, end=end)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    if r.status_code == 200:
        d = r.json()
        cache.write_text(json.dumps(d, ensure_ascii=False))
        time.sleep(0.5)
        return d
    elif r.status_code == 404:
        print(f"    NOT FOUND: {article} en {project}")
        return None
    print(f"    HTTP {r.status_code}: {url}")
    return None

wiki_records = []
for alb in ALBUMS:
    # Query Wikipedia con ambos idiomas
    print(f"\n  {alb['album']} ({alb['artist']})")
    # Página específica del álbum (es)
    for project, article_key in [("es.wikipedia", alb["wiki_es"]),
                                    ("en.wikipedia", alb["wiki_en"])]:
        # Artículo del álbum
        d_album = wiki_pageviews(project, article_key.replace(" ", "_"))
        # Artículo del artista (fallback / contexto)
        d_artist = wiki_pageviews(project, alb["artist"].replace(" ", "_"))
        for source, d in [("album", d_album), ("artist", d_artist)]:
            if not d or "items" not in d:
                continue
            for it in d["items"]:
                wiki_records.append({
                    "key": alb["key"],
                    "project": project,
                    "page_type": source,
                    "month": it["timestamp"][:6],
                    "year": int(it["timestamp"][:4]),
                    "views": it["views"],
                })

df_wiki = pd.DataFrame(wiki_records)
if not df_wiki.empty:
    print("\nResumen Wikipedia pageviews TOTAL desde 2015:")
    total = (df_wiki.groupby(["key", "project", "page_type"])["views"]
              .sum().unstack([1, 2]).fillna(0).astype(int))
    print(total.to_string())

    # Pageviews promedio mensual del ARTISTA (proxy de interés cultural)
    artist_monthly = (df_wiki[df_wiki["page_type"] == "artist"]
                       .groupby(["key", "project"])["views"]
                       .agg(["mean", "median", "sum"]))
    print("\nPageviews mensual del ARTIST:")
    print(artist_monthly.round(0).to_string())

# === 3) Google Trends ================================================
print("\n" + "=" * 70)
print("3) GOOGLE TRENDS — Interés del artista 2004-2026")
print("=" * 70)

trends_cache = Path("data/raw/google_trends.parquet")
if trends_cache.exists():
    df_trends = pd.read_parquet(trends_cache)
    print(f"Cache cargada: {df_trends.shape}")
else:
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="es-MX", tz=360)
        # Hasta 5 queries en una sola petición
        queries = [a["trends_query"] for a in ALBUMS]
        pt.build_payload(queries, cat=0, timeframe="2004-01-01 2026-05-01",
                           geo="", gprop="")
        df_trends = pt.interest_over_time()
        if not df_trends.empty:
            df_trends = df_trends.drop(columns=["isPartial"], errors="ignore")
            df_trends.to_parquet(trends_cache)
            print(f"Trends descargados: {df_trends.shape}")
        else:
            print("⚠️  Google Trends devolvió vacío (rate limit o sin datos)")
            df_trends = pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Google Trends falló: {e}")
        df_trends = pd.DataFrame()

if not df_trends.empty:
    print("\nMedia de interés 2004-2026 por query:")
    print(df_trends.mean().round(1).to_string())
    print("\nÚltimos 12 meses (proxy interés contemporáneo):")
    print(df_trends.tail(12).mean().round(1).to_string())

# === 4) consolidar tabla comparativa ================================
print("\n" + "=" * 70)
print("4) TABLA COMPARATIVA")
print("=" * 70)

# Sumar pageviews de Wikipedia del ARTISTA (proxy más robusto, no depende de
# que exista el artículo específico del álbum)
if not df_wiki.empty:
    artist_total = (df_wiki[df_wiki["page_type"] == "artist"]
                     .groupby(["key", "project"])["views"]
                     .sum().unstack().fillna(0).astype(int))
    if "es.wikipedia" in artist_total.columns:
        artist_es = artist_total["es.wikipedia"]
    else:
        artist_es = pd.Series(0, index=artist_total.index)
    if "en.wikipedia" in artist_total.columns:
        artist_en = artist_total["en.wikipedia"]
    else:
        artist_en = pd.Series(0, index=artist_total.index)

    # Pageviews del álbum específico
    album_total = (df_wiki[df_wiki["page_type"] == "album"]
                    .groupby(["key", "project"])["views"]
                    .sum().unstack().fillna(0).astype(int))
    if "es.wikipedia" in album_total.columns:
        alb_es = album_total["es.wikipedia"]
    else:
        alb_es = pd.Series(0, index=album_total.index)
    if "en.wikipedia" in album_total.columns:
        alb_en = album_total["en.wikipedia"]
    else:
        alb_en = pd.Series(0, index=album_total.index)
else:
    artist_es = artist_en = alb_es = alb_en = pd.Series(dtype=int)

# Trends (último año vs media histórica)
if not df_trends.empty:
    trends_recent = df_trends.tail(12).mean()
    trends_hist = df_trends.mean()
else:
    trends_recent = trends_hist = pd.Series(dtype=float)

rows = []
for alb in ALBUMS:
    k = alb["key"]
    rows.append({
        "key": k,
        "artist": alb["artist"], "album": alb["album"], "year": alb["year"],
        "wiki_es_artist_total": int(artist_es.get(k, 0)),
        "wiki_en_artist_total": int(artist_en.get(k, 0)),
        "wiki_es_album_total": int(alb_es.get(k, 0)),
        "wiki_en_album_total": int(alb_en.get(k, 0)),
        "trends_artist_recent_12mo": float(trends_recent.get(alb["trends_query"], 0)),
        "trends_artist_historic_avg": float(trends_hist.get(alb["trends_query"], 0)),
    })

df_summary = pd.DataFrame(rows)
df_summary = df_summary.merge(df_discogs, on="key", how="left")
print("\n" + df_summary.to_string(index=False))

# === 5) visualizaciones =============================================
# 5.1 — barras: Wikipedia pageviews del álbum por proyecto
fig, ax = plt.subplots(figsize=(11, 5))
y = np.arange(len(df_summary))
ax.barh(y - 0.2, df_summary["wiki_es_artist_total"], 0.4,
         color="#4ea1d3", edgecolor="white", label="ES Wikipedia (artista)")
ax.barh(y + 0.2, df_summary["wiki_en_artist_total"], 0.4,
         color="#ff6b6b", edgecolor="white", label="EN Wikipedia (artista)")
ax.set_yticks(y)
ax.set_yticklabels([f"{r['artist']}" for _, r in df_summary.iterrows()])
ax.set_xlabel("Pageviews totales (2015 → 2026)")
ax.set_title("Wikipedia pageviews del artista (ES vs EN, 2015-2026)",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
for i, r in df_summary.iterrows():
    es_v = r["wiki_es_artist_total"]
    en_v = r["wiki_en_artist_total"]
    if es_v > 0:
        ax.text(es_v, i - 0.2, f" {es_v:,}", va="center", color="white", fontsize=8)
    if en_v > 0:
        ax.text(en_v, i + 0.2, f" {en_v:,}", va="center", color="white", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/commercial_wiki_pageviews.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/commercial_wiki_pageviews.png")

# 5.2 — Google Trends evolución temporal
if not df_trends.empty:
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"Plastilina Mosh": "#ff6b6b", "Café Tacuba": "#6bcb77",
                "Control Machete": "#4ea1d3", "Molotov banda": "#ff9f43",
                "Zurdok": "#c084fc"}
    # Resamplear a anual para suavizar
    annual = df_trends.resample("Y").mean()
    for col in df_trends.columns:
        ax.plot(annual.index, annual[col], label=col,
                 linewidth=2, color=colors.get(col, "#fff"), marker="o")
    ax.set_xlabel("Año"); ax.set_ylabel("Google Trends (0-100, media anual)")
    ax.set_title("Interés histórico por artista en Google (2004-2026)",
                  color="white", fontsize=13)
    ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
               loc="best", fontsize=10)
    ax.set_facecolor("#0d0d1a")
    plt.tight_layout()
    plt.savefig("outputs/figures/commercial_trends.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("[fig] outputs/figures/commercial_trends.png")

# 5.3 — Bubble chart: rating × have × precio
fig, ax = plt.subplots(figsize=(11, 7))
sub = df_summary[df_summary["discogs_rating"].notna()]
xs = sub["discogs_rating"]
ys = sub["discogs_have"]
sizes = sub["discogs_want"] * 5 + 50
labels = sub["artist"]
sc = ax.scatter(xs, ys, s=sizes, c=range(len(sub)), cmap="plasma",
                  edgecolors="white", linewidths=1.5, alpha=0.85)
for x, y, lab in zip(xs, ys, labels):
    ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points",
                  color="white", fontsize=10)
ax.set_xlabel("Discogs community rating (1-5)")
ax.set_ylabel("Discogs # have (collectors)")
ax.set_title("Posición en Discogs · tamaño ∝ # want", color="white", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/commercial_discogs.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/commercial_discogs.png")

# === 6) supervivencia diferencial ====================================
# Una métrica compuesta: percentil del artista en cada métrica.
def rank_normalize(series, ascending=True):
    return series.rank(ascending=ascending, pct=True) * 100

if not df_summary.empty:
    df_summary["score_wiki_es"] = rank_normalize(df_summary["wiki_es_artist_total"])
    df_summary["score_wiki_en"] = rank_normalize(df_summary["wiki_en_artist_total"])
    df_summary["score_trends_recent"] = rank_normalize(df_summary["trends_artist_recent_12mo"])
    df_summary["score_discogs_have"] = rank_normalize(df_summary["discogs_have"].fillna(0))
    df_summary["score_discogs_rating"] = rank_normalize(df_summary["discogs_rating"].fillna(0))
    score_cols = [c for c in df_summary.columns if c.startswith("score_")]
    df_summary["survival_index"] = df_summary[score_cols].mean(axis=1)

    print("\n=== Índice de supervivencia diferencial (percentil promedio) ===")
    print(df_summary[["artist", "album"] + score_cols + ["survival_index"]]
            .round(1).to_string(index=False))

# === guardar ==========================================================
df_summary.to_csv("outputs/exports/commercial_success.csv", index=False)
df_wiki.to_parquet("outputs/exports/commercial_wiki_pageviews.parquet", index=False)
if not df_trends.empty:
    df_trends.to_parquet("outputs/exports/commercial_google_trends.parquet")

# JSON consolidado
out = {
    "albums": df_summary.to_dict("records"),
    "wiki_pageviews_summary": {
        "n_data_points": int(len(df_wiki)),
        "time_range": "2015-07 to 2026-05" if not df_wiki.empty else "no data",
    },
    "google_trends_summary": {
        "n_months": int(len(df_trends)) if not df_trends.empty else 0,
        "time_range": "2004-01 to 2026-05" if not df_trends.empty else "no data",
    },
}
Path("outputs/exports/commercial_success.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8")
print("\n[json] outputs/exports/commercial_success.json")
print("[csv] outputs/exports/commercial_success.csv")
print("\n--- DONE ---")
