"""LLM-as-judge para validar las ventanas de atención.

Para cada par de líneas consecutivas en el mismo track, GPT-4o-mini juzga:
"¿el segundo verso continúa el tema del primero — sí o no — con una
razón breve?"

Luego comparamos el juicio del LLM contra la decisión binaria de
'ventana rota' (sim < θ) de los embeddings.

Métricas:
  - acuerdo global (Cohen's κ)
  - acuerdo estratificado por language-switch vs same-lang
  - matriz de confusión 2×2

Lo importante: si los embeddings dicen "ruptura" mientras el LLM dice
"continuidad" — y eso sucede DESPROPORCIONADAMENTE en transiciones de
idioma — entonces los embeddings están midiendo discontinuidad léxica
y no semántica. Esa es la validación dura.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from dotenv import load_dotenv
from openai import OpenAI

for p in [Path.cwd(), *Path.cwd().parents][:5]:
    if (p / ".env").exists():
        load_dotenv(p / ".env"); break

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# === datos =============================================================
lines = pd.read_parquet("outputs/exports/corpus_lines_v2.parquet")
lines = lines.sort_values(["track_num", "line_num"]).reset_index(drop=True)
emb_oa = np.load("data/embeddings/openai_lyrics_lines.npy")
emb_lb = np.load("data/embeddings/labse_lyrics_lines.npy")

def normalize(emb):
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    out = np.where(nrm > 1e-10, emb / np.maximum(nrm, 1e-12), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

emb_oa_n = normalize(emb_oa)
emb_lb_n = normalize(emb_lb)
theta_oa = 0.3201
theta_lb = 0.3230

# === construir pairs ==================================================
pair_rows = []
for tnum, g in lines.groupby("track_num", sort=False):
    idx = g.index.values
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k+1]
        s_oa = float(emb_oa_n[i] @ emb_oa_n[j]) if (np.isfinite(emb_oa_n[i]).all()
                                                       and np.isfinite(emb_oa_n[j]).all()) else np.nan
        s_lb = float(emb_lb_n[i] @ emb_lb_n[j]) if (np.isfinite(emb_lb_n[i]).all()
                                                       and np.isfinite(emb_lb_n[j]).all()) else np.nan
        if not (np.isfinite(s_oa) and np.isfinite(s_lb)):
            continue
        pair_rows.append({
            "track_num": int(tnum),
            "title": g["title"].iloc[k],
            "line_a": g["line_text"].iloc[k],
            "line_b": g["line_text"].iloc[k+1],
            "lang_a": g["lang_v2"].iloc[k],
            "lang_b": g["lang_v2"].iloc[k+1],
            "switch": int(g["lang_v2"].iloc[k] != g["lang_v2"].iloc[k+1]),
            "sim_openai": s_oa,
            "sim_labse": s_lb,
            "broken_openai": int(s_oa < theta_oa),
            "broken_labse": int(s_lb < theta_lb),
        })
df_pairs = pd.DataFrame(pair_rows)
print(f"Pares consecutivos a juzgar: {len(df_pairs)}")
print(f"  same-lang: {(df_pairs['switch']==0).sum()}")
print(f"  switch:    {(df_pairs['switch']==1).sum()}")

# === LLM judge ========================================================
JUDGE_PROMPT = """Eres un lector atento de letras de canciones. Lee dos líneas consecutivas y juzga si la segunda CONTINÚA el tema, la imagen o la acción de la primera, o si introduce un cambio de tema.

Importante: si las líneas están en idiomas distintos pero hablan de lo mismo, eso ES continuidad. Si están en el mismo idioma pero hablan de cosas no relacionadas, eso ES discontinuidad.

Línea A: "{LINE_A}"
Línea B: "{LINE_B}"

Responde SOLO en JSON estricto:
{"continuidad": true|false, "razon": "<10-25 palabras>"}
"""

def judge_pair(line_a: str, line_b: str) -> dict:
    prompt = JUDGE_PROMPT.replace("{LINE_A}", line_a).replace("{LINE_B}", line_b)
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150,
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"continuidad": None, "razon": f"[ERROR: {e}]"}

# Con cache
cache_path = Path("data/processed/llm_judge_cache.json")
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

print("\nLlamando al LLM judge (cache caliente acelera mucho)...")
judgments = []
for i, r in df_pairs.iterrows():
    key = f"{r['line_a']}|||{r['line_b']}"
    if key in cache:
        judgments.append(cache[key])
        continue
    j = judge_pair(r["line_a"], r["line_b"])
    cache[key] = j
    judgments.append(j)
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{len(df_pairs)}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                encoding="utf-8")
cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                        encoding="utf-8")

df_pairs["llm_continuity"] = [j.get("continuidad") for j in judgments]
df_pairs["llm_reason"] = [j.get("razon", "") for j in judgments]
df_pairs = df_pairs.dropna(subset=["llm_continuity"]).copy()
df_pairs["llm_broken"] = (~df_pairs["llm_continuity"].astype(bool)).astype(int)

print(f"\nJuicios válidos: {len(df_pairs)}/{len(judgments)}")
print(f"LLM dice continuidad: {df_pairs['llm_continuity'].sum()}")
print(f"LLM dice ruptura:     {(~df_pairs['llm_continuity']).sum()}")

# === métricas =========================================================
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def stratified_agreement(df, model_col):
    rows = []
    for stratum, sub in [("all", df),
                          ("same-lang", df[df["switch"] == 0]),
                          ("switch", df[df["switch"] == 1])]:
        if len(sub) < 5: continue
        # broken=1 ↔ llm_broken=1: acuerdo
        agree = (sub[model_col] == sub["llm_broken"]).mean()
        k = kappa(sub["llm_broken"], sub[model_col])
        # ¿cuándo el modelo dice 'ruptura' pero el LLM dice 'continuidad'?
        false_breaks = ((sub[model_col] == 1) & (sub["llm_broken"] == 0)).sum()
        false_breaks_rate = false_breaks / len(sub)
        # ¿cuándo el modelo dice 'continuidad' pero el LLM dice 'ruptura'?
        missed_breaks = ((sub[model_col] == 0) & (sub["llm_broken"] == 1)).sum()
        missed_breaks_rate = missed_breaks / len(sub)
        rows.append({
            "model": model_col.replace("broken_", "").upper(),
            "stratum": stratum,
            "n": len(sub),
            "agreement": agree,
            "kappa": k,
            "false_break_rate": false_breaks_rate,
            "missed_break_rate": missed_breaks_rate,
        })
    return pd.DataFrame(rows)

print("\n=== Acuerdo Modelo vs LLM-judge (estratificado) ===")
df_strat = pd.concat([
    stratified_agreement(df_pairs, "broken_openai"),
    stratified_agreement(df_pairs, "broken_labse"),
], ignore_index=True)
print(df_strat.round(3).to_string(index=False))

# === el hallazgo central ==============================================
# ¿La tasa de "false breaks" (modelo dice rompe, LLM dice continúa) es
# mayor en switches que en same-lang? Eso es la evidencia dura.
print("\n=== HALLAZGO CENTRAL ===")
print("Tasa de 'false break' (modelo rompe, LLM continúa):")
for model_col in ["broken_openai", "broken_labse"]:
    model_name = model_col.replace("broken_", "").upper()
    fb_same = ((df_pairs[model_col] == 1) & (df_pairs["llm_broken"] == 0) &
                (df_pairs["switch"] == 0)).sum() / (df_pairs["switch"] == 0).sum()
    fb_sw = ((df_pairs[model_col] == 1) & (df_pairs["llm_broken"] == 0) &
              (df_pairs["switch"] == 1)).sum() / (df_pairs["switch"] == 1).sum()
    print(f"  {model_name}:  same-lang false-break = {fb_same:.3f},  switch false-break = {fb_sw:.3f},  ratio = {fb_sw/fb_same if fb_same > 0 else float('inf'):.2f}×")

# === ejemplos ilustrativos ============================================
print("\n=== EJEMPLOS de 'false breaks' en switches ===")
fb_examples = df_pairs[
    (df_pairs["switch"] == 1) &
    (df_pairs["broken_openai"] == 1) &
    (df_pairs["llm_broken"] == 0)
].head(5)
for _, r in fb_examples.iterrows():
    print(f"\n  [{r['title'][:25]}]  ({r['lang_a']}→{r['lang_b']}) sim_oa={r['sim_openai']:.3f}")
    print(f"    A: {r['line_a']}")
    print(f"    B: {r['line_b']}")
    print(f"    LLM dice continuidad: {r['llm_reason']}")

# === guardar ==========================================================
df_pairs.to_parquet("outputs/exports/llm_judge_pairs.parquet", index=False)
df_strat.to_csv("outputs/exports/llm_judge_stratified.csv", index=False)

# JSON resumen
out = {
    "n_pairs": int(len(df_pairs)),
    "llm_continuity_pct": float(df_pairs["llm_continuity"].mean()),
    "stratified": df_strat.round(4).to_dict("records"),
    "false_break_rates": {
        "openai": {
            "same_lang": float(((df_pairs["broken_openai"] == 1)
                                  & (df_pairs["llm_broken"] == 0)
                                  & (df_pairs["switch"] == 0)).sum() / max((df_pairs["switch"] == 0).sum(), 1)),
            "switch": float(((df_pairs["broken_openai"] == 1)
                              & (df_pairs["llm_broken"] == 0)
                              & (df_pairs["switch"] == 1)).sum() / max((df_pairs["switch"] == 1).sum(), 1)),
        },
        "labse": {
            "same_lang": float(((df_pairs["broken_labse"] == 1)
                                  & (df_pairs["llm_broken"] == 0)
                                  & (df_pairs["switch"] == 0)).sum() / max((df_pairs["switch"] == 0).sum(), 1)),
            "switch": float(((df_pairs["broken_labse"] == 1)
                              & (df_pairs["llm_broken"] == 0)
                              & (df_pairs["switch"] == 1)).sum() / max((df_pairs["switch"] == 1).sum(), 1)),
        },
    },
    "examples_false_break_in_switches": [
        {"track": r["title"], "from_lang": r["lang_a"], "to_lang": r["lang_b"],
          "line_a": r["line_a"], "line_b": r["line_b"],
          "sim_openai": float(r["sim_openai"]), "llm_reason": r["llm_reason"]}
        for _, r in fb_examples.iterrows()
    ],
}
Path("outputs/exports/llm_judge.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n[json] outputs/exports/llm_judge.json")
print("[parquet] outputs/exports/llm_judge_pairs.parquet")
print("[csv] outputs/exports/llm_judge_stratified.csv")

# === figura ===========================================================
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, model_col, name, color in [
    (axes[0], "broken_openai", "OpenAI", "#ff6b6b"),
    (axes[1], "broken_labse", "LaBSE", "#4ea1d3"),
]:
    fb_same = ((df_pairs[model_col] == 1) & (df_pairs["llm_broken"] == 0) &
                (df_pairs["switch"] == 0)).sum() / max((df_pairs["switch"] == 0).sum(), 1)
    fb_sw = ((df_pairs[model_col] == 1) & (df_pairs["llm_broken"] == 0) &
              (df_pairs["switch"] == 1)).sum() / max((df_pairs["switch"] == 1).sum(), 1)
    bars = ax.bar(["same-lang", "lang switch"], [fb_same, fb_sw],
                    color=color, edgecolor="white")
    bars[0].set_alpha(0.55)
    bars[1].set_alpha(1.0)
    for j, v in enumerate([fb_same, fb_sw]):
        ax.text(j, v + 0.01, f"{v:.2f}", ha="center", color="white", fontsize=11)
    ax.set_ylim(0, max(fb_same, fb_sw) * 1.4 + 0.05)
    ax.set_ylabel("P( modelo rompe ∧ LLM dice continúa )")
    ax.set_title(name, color="white")
    ax.set_facecolor("#0d0d1a")
plt.suptitle("Tasa de 'falsa ruptura': el modelo rompe pero un lector lo lee como continuidad",
              color="white", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/llm_judge_false_breaks.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/llm_judge_false_breaks.png")
print("\n--- DONE ---")
