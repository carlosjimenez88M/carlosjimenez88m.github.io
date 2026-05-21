"""Deeper analysis pass: fix language detection, regenerate stats, find exemplars."""
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 42

# Style
plt.rcParams.update({"axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
                     "axes.edgecolor": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "axes.titlecolor": "white"})

lines = pd.read_parquet("outputs/exports/corpus_lines.parquet")
lyrics = pd.read_parquet("outputs/exports/corpus_lyrics.parquet")

# === 1) IMPROVED LANGUAGE DETECTION =====================================
SPANISH_MARKERS = {"que","de","la","el","y","en","un","una","con","por","los",
                    "las","como","es","te","mi","mis","tu","tus","sin","sus",
                    "para","pero","más","si","yo","me","se","lo","al","del",
                    "ya","no","nos","ni","muy","aquí","allá","esto","esta"}
ENGLISH_MARKERS = {"the","of","and","to","you","i","is","my","in","on","with",
                    "for","at","by","that","this","be","are","was","but","not",
                    "have","do","what","when","like","baby","just","go"}
FRENCH_MARKERS  = {"je","tu","le","la","et","de","les","un","une","ne","pas",
                    "ce","que","est","mais","ou","où","des","du","ses","mon",
                    "ma","mes","ton","ta","tes","son","sa","quand","comment"}
# JA: hiragana, katakana, kanji
JA_PATTERN = re.compile(r"[぀-ヿ一-鿿]")
# Words that are Spanish-only contractions / regionalisms langdetect misses
SPANISH_STRONG = {"pa'", "p'", "pa", "tá", "ta", "qué", "cómo", "dónde",
                   "sí", "está", "esté", "está", "soy", "eres", "fue",
                   "muy", "más", "hacer", "haces", "haciendo", "ese", "esa"}

def detect_lang_robust(text: str) -> tuple[str, float]:
    """Devuelve (lang, prob). Prioriza señales fuertes; usa langdetect como respaldo."""
    if not text or len(text.strip()) < 2:
        return ("OTHER", 0.0)

    # 1) Japanese: si hay caracteres japoneses, JA seguro.
    if JA_PATTERN.search(text):
        return ("JA", 1.0)

    tokens = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñüÜçÇ']+", text.lower())
    if not tokens:
        return ("OTHER", 0.0)
    token_set = set(tokens)

    # 2) Conteo de markers
    counts = {
        "ES": len(token_set & SPANISH_MARKERS) + 2*len(token_set & SPANISH_STRONG),
        "EN": len(token_set & ENGLISH_MARKERS),
        "FR": len(token_set & FRENCH_MARKERS),
    }
    # Caracteres acentuados ES
    if re.search(r"[ñÑáéíóúüÁÉÍÓÚÜ]", text):
        counts["ES"] += 1

    # Mixed: dos o más idiomas con marcas
    languages_present = [k for k, v in counts.items() if v >= 1]
    if len(languages_present) >= 2:
        # MIXED si las dos lenguas tienen ≥1 marker y la línea es ≥4 tokens
        if len(tokens) >= 4:
            return ("MIXED", 0.7)

    # Si una lengua domina por marcadores
    if any(v >= 2 for v in counts.values()):
        best = max(counts, key=counts.get)
        return (best, 0.85)

    # 3) Caer en langdetect para probabilidades
    try:
        langs = detect_langs(text)
        # langs es lista de Language objects (lang, prob)
        best = langs[0]
        prob = best.prob
        c = best.lang
    except Exception:
        return ("OTHER", 0.0)
    mapping = {"es": "ES", "en": "EN", "fr": "FR", "ja": "JA",
                # Mapear lenguas romance al español si la confianza es media (suelen confundirse)
                "ca": "ES", "gl": "ES", "pt": "ES", "it": "ES",
                # Mapear germánicas al inglés
                "de": "EN", "nl": "EN", "da": "EN", "sv": "EN",
                "no": "EN", "af": "EN"}
    return (mapping.get(c, "OTHER"), float(prob))

# Re-detectar en TODAS las líneas
lines["lang_v2"], lines["lang_prob"] = zip(*lines["line_text"].apply(detect_lang_robust))

print("=" * 70)
print("LANGUAGE DETECTION COMPARISON")
print("=" * 70)
print("Antes (langdetect crudo):")
print(lines["language"].value_counts())
print("\nDespués (detección robusta con markers + langdetect):")
print(lines["lang_v2"].value_counts())

# Confusion: cuántas líneas cambiaron
moves = lines[lines["language"] != lines["lang_v2"]]
print(f"\n{len(moves)} líneas cambiaron de idioma:")
print(pd.crosstab(moves["language"], moves["lang_v2"], margins=True))

# Ejemplo de movimientos
print("\nEjemplos OTHER -> ES:")
ex = moves[(moves["language"] == "OTHER") & (moves["lang_v2"] == "ES")].head(8)
for _, r in ex.iterrows():
    print(f"  [{r['title'][:18]:18s}] {r['line_text'][:70]}")

# === 2) RE-RUN CHI-SQUARED CON LANG_V2 ==================================
df_stat = lines[lines["confidence"] >= 0.5]
df_stat = df_stat[df_stat["lang_v2"].isin(["ES","EN","FR","JA","MIXED"])]
df_stat = df_stat[df_stat["campo"].isin(["CUERPO","MARCA","LUGAR","EMOCION",
                                          "IDENTIDAD","ACCION","REFERENCIA","NONSENSE"])]

ctab = pd.crosstab(df_stat["lang_v2"], df_stat["campo"])
print("\n" + "=" * 70)
print(f"CHI² CON DETECCIÓN MEJORADA (n={len(df_stat)} líneas)")
print("=" * 70)
print(ctab)
chi2, p, dof, expected = chi2_contingency(ctab)
print(f"\nχ² = {chi2:.2f}  dof = {dof}  p = {p:.3e}")

residuals = (ctab.values - expected) / np.sqrt(expected)
df_resid = pd.DataFrame(residuals, index=ctab.index, columns=ctab.columns)
print("\nResiduos estandarizados (|z| > 2 = asociación significativa):")
print(df_resid.round(2))

# === 3) EXEMPLAR LINES PER STRONG RESIDUAL ==============================
print("\n" + "=" * 70)
print("EJEMPLOS POR ASOCIACIÓN FUERTE (idioma × campo)")
print("=" * 70)
strong = []
for lang in df_resid.index:
    for campo in df_resid.columns:
        z = df_resid.loc[lang, campo]
        if abs(z) > 2.0:
            strong.append((lang, campo, z))
strong.sort(key=lambda x: -abs(x[2]))
for lang, campo, z in strong[:10]:
    sign = "+" if z > 0 else "-"
    print(f"\n  {sign}{abs(z):.2f}  {lang} × {campo}")
    examples = df_stat[(df_stat["lang_v2"] == lang) & (df_stat["campo"] == campo)]
    for _, r in examples.head(4).iterrows():
        print(f"    [{r['title'][:18]:18s}] {r['line_text'][:75]}")

# === 4) PER-TRACK AXIS POSITIONS ========================================
print("\n" + "=" * 70)
print("POSICIÓN DE CADA TRACK EN LOS 3 EJES")
print("=" * 70)
axes_path = Path("data/processed/semantic_axes.json")
df_axes = pd.DataFrame(json.loads(axes_path.read_text()))
df_axes_sorted = df_axes.set_index("title")
print(df_axes_sorted.round(3))
print("\nTracks más 'mainstream LA' (top en ORIGEN_la):")
print(df_axes_sorted.sort_values("ORIGEN_la", ascending=False)[["ORIGEN_la"]].head(3))
print("\nTracks más 'regio/local' (bottom en ORIGEN_la):")
print(df_axes_sorted.sort_values("ORIGEN_la").head(3)[["ORIGEN_la"]])
print("\nTracks más irónicos:")
print(df_axes_sorted.sort_values("SUPERFICIE_ironia", ascending=False).head(3)[["SUPERFICIE_ironia"]])
print("\nTracks más emocionales:")
print(df_axes_sorted.sort_values("SUPERFICIE_ironia").head(3)[["SUPERFICIE_ironia"]])
print("\nTracks que envejecieron más como 'clásico retro':")
print(df_axes_sorted.sort_values("TIEMPO_retro", ascending=False).head(3)[["TIEMPO_retro"]])

# === 5) PER-TRACK LANGUAGE DOMINANCE ====================================
print("\n" + "=" * 70)
print("DISTRIBUCIÓN DE IDIOMA POR TRACK (con detección mejorada)")
print("=" * 70)
lang_by_track = pd.crosstab(lines["title"], lines["lang_v2"])
lang_by_track["dominante"] = lang_by_track.idxmax(axis=1)
lang_by_track["entropia"] = lang_by_track.iloc[:, :-1].apply(
    lambda r: -sum((p/r.sum())*np.log2(p/r.sum()) for p in r if p > 0), axis=1)
print(lang_by_track.round(2))

# === 6) SAVE NEW ARTIFACTS ===============================================
out = {
    "language_detection_improved": {
        "method": "marker-based + langdetect probabilities, romance-language collapse",
        "lines_relabeled": int(len(moves)),
        "before": lines["language"].value_counts().to_dict(),
        "after":  lines["lang_v2"].value_counts().to_dict(),
    },
    "chi_squared_improved": {
        "chi2": float(chi2), "p": float(p), "dof": int(dof),
        "n_lines": int(len(df_stat)),
        "table": ctab.to_dict(),
        "residuals": df_resid.round(3).to_dict(),
    },
    "strong_associations": [
        {"lang": lang, "campo": campo, "z": round(float(z), 2)}
        for lang, campo, z in strong
    ],
    "per_track_axes": df_axes_sorted.round(3).reset_index().to_dict("records"),
    "per_track_language": lang_by_track.round(3).reset_index().to_dict("records"),
}
Path("outputs/exports/deep_findings.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8")
print("\nGuardado: outputs/exports/deep_findings.json")

# Guardar el dataframe de líneas con lang_v2 para futuras visualizaciones
lines.to_parquet("outputs/exports/corpus_lines_v2.parquet", index=False)
print("Guardado: outputs/exports/corpus_lines_v2.parquet")

# === 7) NUEVA VISUALIZACIÓN: heatmap mejorado ============================
fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(df_resid, annot=True, fmt=".2f", cmap="RdBu_r",
             center=0, vmin=-5, vmax=5, ax=ax,
             cbar_kws={"label": "Residuo estandarizado"},
             linewidths=0.6, linecolor="#0d0d1a")
ax.set_title(f"Asociación idioma × campo (mejorada)  ·  χ²={chi2:.1f}, p<1e-{int(-np.log10(p)):d}",
              color="white", fontsize=13, pad=12)
ax.set_xlabel("Campo semántico"); ax.set_ylabel("Idioma")
plt.tight_layout()
plt.savefig("outputs/figures/language_field_residuals_v2.png", dpi=140,
            facecolor="#0d0d1a", bbox_inches="tight")
print("Guardado: outputs/figures/language_field_residuals_v2.png")

# === 8) NUEVA VISUALIZACIÓN: distribución de idioma por track ============
fig, ax = plt.subplots(figsize=(12, 5))
plot_data = lang_by_track.iloc[:, :-2]
# Asegurar orden ES, EN, FR, MIXED, OTHER
order = [c for c in ["ES", "EN", "FR", "MIXED", "OTHER"] if c in plot_data.columns]
plot_data = plot_data[order]
colors_map = {"ES": "#4ea1d3", "EN": "#ff6b6b", "FR": "#6bcb77",
               "JA": "#ff9f43", "MIXED": "#a0a0a0", "OTHER": "#444"}
plot_data.plot(kind="barh", stacked=True, ax=ax,
                color=[colors_map.get(c, "#777") for c in plot_data.columns])
ax.set_title("Composición lingüística por track (líneas)", color="white", fontsize=13)
ax.set_xlabel("Número de líneas"); ax.set_ylabel("")
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           loc="lower right")
plt.tight_layout()
plt.savefig("outputs/figures/language_by_track.png", dpi=140,
            facecolor="#0d0d1a", bbox_inches="tight")
print("Guardado: outputs/figures/language_by_track.png")
