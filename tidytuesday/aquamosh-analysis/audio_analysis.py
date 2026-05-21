"""Análisis de audio de Aquamosh con LAION-CLAP.

CLAP (Contrastive Language-Audio Pretraining) embebe audio y texto en el MISMO
espacio. Esto permite proyectar los 12 tracks del álbum sobre los mismos ejes
culturales Kozlowski (ORIGEN, SUPERFICIE, TIEMPO) que construí desde anchors
verbales, ahora con vectores DERIVADOS DEL SONIDO.

La pregunta empírica clave:

  ¿Las posiciones de los tracks en los ejes culturales coinciden cuando se
  miden desde las letras vs desde el sonido?

Si coinciden: la decisión semántica del álbum (qué dice cada track) y la
decisión sónica (cómo suena) están alineadas. Si discrepan: las letras y la
producción de Rothrock/Schnapf cuentan historias distintas, y la mezcla
"globalizó" el sonido independientemente del contenido lingüístico.

Adicional:
  - Features acústicos baseline (tempo, energía RMS, centroide espectral)
  - Identificación de samples sospechosos vía cross-similarity entre tracks
"""
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf
import torch

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# === 1) inventario de audio ===========================================
audio_files = sorted(Path("data/audio").glob("*.mp3"))
print(f"Tracks: {len(audio_files)}")

def parse_trackname(path: Path) -> dict:
    name = path.stem
    # Forma "01_Niño Bomba" o "01_Banano´s Bar"
    parts = name.split("_", 1)
    try:
        num = int(parts[0])
    except Exception:
        num = -1
    title = parts[1] if len(parts) > 1 else name
    return {"path": str(path), "track_num": num, "title": title}

audio_meta = [parse_trackname(p) for p in audio_files]
df_audio = pd.DataFrame(audio_meta).sort_values("track_num").reset_index(drop=True)
print(df_audio[["track_num", "title"]].to_string(index=False))

# === 2) features acústicos baseline ===================================
print("\nExtrayendo features acústicos básicos (tempo, RMS, centroide)...")
features = []
for _, r in df_audio.iterrows():
    y, sr = librosa.load(r["path"], sr=22050, mono=True)
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo_arr).item())
    rms = float(librosa.feature.rms(y=y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    duration = librosa.get_duration(y=y, sr=sr)
    features.append({
        "track_num": r["track_num"],
        "title": r["title"],
        "duration_s": duration,
        "tempo_bpm": tempo,
        "rms_energy": rms,
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "zcr": zcr,
        "spectral_bandwidth_hz": bandwidth,
    })
    print(f"  {r['track_num']:2d}  {r['title'][:30]:30s}  {tempo:5.1f}bpm  "
          f"RMS={rms:.3f}  centroid={centroid:.0f}Hz")

df_features = pd.DataFrame(features)
df_features.to_csv("outputs/exports/audio_baseline_features.csv", index=False)
print(f"\n[csv] outputs/exports/audio_baseline_features.csv")

# === 3) CLAP embeddings de audio ======================================
print("\nCargando LAION-CLAP...")
import laion_clap
# HTSAT-tiny es el que corresponde al checkpoint default (630k_audioset_best)
model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
model.load_ckpt()
# CLAP no convive bien con MPS por algunas operaciones, lo dejamos en CPU
print("  CLAP corre en CPU (compatibilidad con operaciones de modelo)")

def embed_audio_clap(path: str) -> np.ndarray:
    """Embedding CLAP del track. La librería expone get_audio_embedding_from_filelist."""
    emb = model.get_audio_embedding_from_filelist(x=[path], use_tensor=False)
    return emb[0]

print("Generando embeddings CLAP por track...")
audio_emb_cache = Path("data/embeddings/clap_audio_tracks.npy")
audio_titles_cache = Path("data/embeddings/clap_audio_titles.json")
if audio_emb_cache.exists() and audio_titles_cache.exists():
    audio_emb = np.load(audio_emb_cache)
    cached_titles = json.loads(audio_titles_cache.read_text())
    if cached_titles == df_audio["title"].tolist():
        print(f"  Cache OK: {audio_emb.shape}")
    else:
        audio_emb = None
else:
    audio_emb = None

if audio_emb is None:
    audio_emb = np.zeros((len(df_audio), 512), dtype=np.float32)
    for i, r in df_audio.iterrows():
        try:
            e = embed_audio_clap(r["path"])
            audio_emb[i] = e
            print(f"  {r['track_num']:2d} {r['title'][:30]:30s}  shape={e.shape}")
        except Exception as e:
            print(f"  {r['track_num']:2d} FAIL: {e}")
    np.save(audio_emb_cache, audio_emb)
    audio_titles_cache.write_text(json.dumps(df_audio["title"].tolist(),
                                              ensure_ascii=False))
print(f"audio_emb shape: {audio_emb.shape}")

# Normalizar
audio_emb_n = audio_emb / (np.linalg.norm(audio_emb, axis=1, keepdims=True) + 1e-12)

# === 4) CLAP text embeddings de los anchors Kozlowski =================
print("\nEmbeddings de los anchors verbales (text-side de CLAP)...")
AXES = {
    "ORIGEN": {
        "A": ["Monterrey", "norteño", "regio", "frontera norte",
              "Avanzada Regia", "barrio", "colonia"],
        "B": ["Hollywood", "MTV", "Sunset Boulevard", "Capitol Records",
              "Los Angeles", "mainstream americano", "radio inglesa"],
    },
    "SUPERFICIE": {
        "A": ["sentimiento", "amor", "dolor", "deseo",
              "nostalgia", "soledad"],
        "B": ["parodia", "sarcasmo", "kitsch", "pastiche",
              "referencia cultural", "cita"],
    },
    "TIEMPO": {
        "A": ["alternativo", "indie", "underground", "experimental",
              "trip-hop", "nuevo", "vanguardia"],
        "B": ["clásico", "nostalgia", "mítico", "histórico",
              "influyente", "referencia obligada"],
    },
    # Eje añadido específico para audio: instrumental ↔ vocal
    "VOCALIDAD": {
        "A": ["instrumental music", "no vocals", "purely instrumental"],
        "B": ["vocal music", "singing", "voice", "lyrical"],
    },
}

def embed_text_clap(texts: list[str]) -> np.ndarray:
    return model.get_text_embedding(texts, use_tensor=False)

def build_axis(polo_a: list[str], polo_b: list[str]) -> np.ndarray:
    a = embed_text_clap(polo_a).mean(axis=0)
    b = embed_text_clap(polo_b).mean(axis=0)
    v = b - a
    return v / (np.linalg.norm(v) + 1e-12)

axis_vectors = {}
for name, polos in AXES.items():
    axis_vectors[name] = build_axis(polos["A"], polos["B"])
    print(f"  Eje {name}: shape {axis_vectors[name].shape}")

# === 5) proyectar audio sobre los ejes Kozlowski ======================
print("\nProyecciones de tracks sobre los ejes...")
projections = {name: audio_emb_n @ v for name, v in axis_vectors.items()}

df_proj_audio = pd.DataFrame({"title": df_audio["title"]})
for name, p in projections.items():
    df_proj_audio[f"audio_{name}"] = p

print(df_proj_audio.round(3).to_string(index=False))

# === 6) cargar proyecciones lyrics-only para comparar =================
print("\nCargando proyecciones lyrics-only (Sección 5 del notebook)...")
lyrics_proj_path = Path("data/processed/semantic_axes.json")
df_proj_lyrics = pd.DataFrame()
if lyrics_proj_path.exists():
    data = json.loads(lyrics_proj_path.read_text())
    df_proj_lyrics = pd.DataFrame(data)
    print(df_proj_lyrics.round(3).to_string(index=False))

# Match titles entre audio (con apostrofo raro de YouTube) y lyrics
def norm_title(s):
    s = s.lower()
    s = s.replace("´", "'").replace("`", "'").replace("´", "'")
    s = s.replace(" feat. pocahontas freaky groove", "")
    s = s.replace(" (melancolic mix)", "")
    s = s.replace("´", "'")
    return s

if not df_proj_lyrics.empty:
    df_proj_audio["norm_title"] = df_proj_audio["title"].apply(norm_title)
    df_proj_lyrics["norm_title"] = df_proj_lyrics["title"].apply(norm_title)
    df_compare = df_proj_audio.merge(df_proj_lyrics, on="norm_title",
                                       how="inner", suffixes=("_audio", "_lyrics"))
    print(f"\nTracks empatados audio-lyrics: {len(df_compare)}")

    # Correlación por eje
    from scipy.stats import pearsonr, spearmanr
    print("\n=== Correlación entre proyección audio vs lyrics, por eje ===")
    correls = {}
    for axis in ["ORIGEN", "SUPERFICIE", "TIEMPO"]:
        a_col = f"audio_{axis}"
        l_col = {"ORIGEN": "ORIGEN_la", "SUPERFICIE": "SUPERFICIE_ironia",
                  "TIEMPO": "TIEMPO_retro"}[axis]
        if a_col in df_compare.columns and l_col in df_compare.columns:
            if df_compare[a_col].std() < 1e-9 or df_compare[l_col].std() < 1e-9:
                r, p = 0.0, 1.0
                rho, _ = 0.0, 1.0
            else:
                r, p = pearsonr(df_compare[a_col], df_compare[l_col])
                rho, _ = spearmanr(df_compare[a_col], df_compare[l_col])
            correls[axis] = {"pearson": float(r), "spearman": float(rho), "p": float(p)}
            print(f"  {axis:10s}  Pearson r = {r:+.3f}  Spearman ρ = {rho:+.3f}  p = {p:.3g}")

# === 7) cross-similarity entre tracks (¿sonido común?) ================
print("\n=== Cross-similarity ENTRE tracks en el espacio CLAP de audio ===")
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(audio_emb_n)
fig, ax = plt.subplots(figsize=(11, 9))
short_titles = [t[:22] for t in df_audio["title"]]
sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="magma", vmin=0.0, vmax=1.0,
             xticklabels=short_titles, yticklabels=short_titles, ax=ax,
             cbar_kws={"label": "Cosine similarity (CLAP audio)"})
ax.set_title("Similaridad sónica entre tracks de Aquamosh (CLAP audio embeddings)",
              color="white", fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("outputs/figures/audio_sim_matrix.png", dpi=130, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/audio_sim_matrix.png")

# Promedio inter-track (excluyendo diagonal): nivel de "sonido común"
np.fill_diagonal(sim_matrix, np.nan)
mean_inter = np.nanmean(sim_matrix)
print(f"Similaridad media inter-track (CLAP audio): {mean_inter:.3f}")
print(f"  Si fuera ~0.95 → el productor 'aplanó' todo a un sonido común.")
print(f"  Si fuera ~0.50 → cada track suena distinto.")

# === 8) visualización: comparación de proyecciones ====================
if not df_compare.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, axis_name in zip(axes, ["ORIGEN", "SUPERFICIE", "TIEMPO"]):
        a_col = f"audio_{axis_name}"
        l_col = {"ORIGEN": "ORIGEN_la", "SUPERFICIE": "SUPERFICIE_ironia",
                  "TIEMPO": "TIEMPO_retro"}[axis_name]
        ax.scatter(df_compare[l_col], df_compare[a_col], s=120,
                    c="#ff6b6b", edgecolors="white", linewidths=1.2)
        for _, r in df_compare.iterrows():
            ax.annotate(r["title_lyrics"][:18] if "title_lyrics" in df_compare.columns
                         else r.get("title_audio", "")[:18],
                         (r[l_col], r[a_col]), xytext=(5, 5),
                         textcoords="offset points", color="white", fontsize=8)
        # Diagonal de referencia (z-score-style)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        lo = min(xlim[0], ylim[0]); hi = max(xlim[1], ylim[1])
        # No diagonal porque las escalas no son comparables; reemplazamos por línea cero
        ax.axhline(0, color="#666", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="#666", linestyle="--", linewidth=0.5)
        r = correls.get(axis_name, {})
        ax.set_title(f"{axis_name}  ·  r={r.get('pearson',0):+.2f}  ρ={r.get('spearman',0):+.2f}",
                      color="white")
        ax.set_xlabel("Proyección lyrics")
        ax.set_ylabel("Proyección audio")
        ax.set_facecolor("#0d0d1a")
    plt.suptitle("¿Coinciden audio y letras en los ejes culturales?",
                  color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/figures/audio_vs_lyrics_axes.png", dpi=140,
                 bbox_inches="tight")
    plt.close()
    print("[fig] outputs/figures/audio_vs_lyrics_axes.png")

# === 9) rankings por eje (audio-only) =================================
print("\n=== Rankings por eje CLAP-audio ===")
for axis in ["ORIGEN", "SUPERFICIE", "TIEMPO", "VOCALIDAD"]:
    col = f"audio_{axis}"
    sub = df_proj_audio[["title", col]].sort_values(col, ascending=False)
    print(f"\n  {axis}:")
    for _, r in sub.iterrows():
        print(f"    {r[col]:+.3f}  {r['title']}")

# === 10) guardar todo =================================================
results = {
    "n_tracks": len(df_audio),
    "audio_features": df_features.to_dict("records"),
    "audio_projections_kozlowski": df_proj_audio.to_dict("records"),
    "correlation_audio_vs_lyrics": correls if not df_compare.empty else {},
    "mean_inter_track_similarity": float(mean_inter),
    "device": DEVICE,
    "model": "LAION-CLAP HTSAT-base, default checkpoint",
}
Path("outputs/exports/audio_analysis.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8")
print("\n[json] outputs/exports/audio_analysis.json")

# Plot de features acústicos por track
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
order = df_features.sort_values("track_num")
for ax, col, title in zip(axes.flat,
    ["tempo_bpm", "rms_energy", "spectral_centroid_hz", "spectral_bandwidth_hz"],
    ["Tempo (BPM)", "Energía RMS", "Centroide espectral (Hz)", "Ancho de banda espectral (Hz)"]):
    ax.barh(range(len(order)), order[col], color="#4ea1d3", edgecolor="white")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([t[:24] for t in order["title"]])
    ax.set_title(title, color="white")
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(labelsize=8)
plt.suptitle("Features acústicos baseline por track", color="white", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/audio_baseline_features.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/audio_baseline_features.png")

print("\n--- DONE ---")
