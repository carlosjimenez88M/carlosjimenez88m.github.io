"""Final synthesis: feed the deeper findings to GPT-4o for a substantive interpretation."""
import os, json
from pathlib import Path
from dotenv import load_dotenv

for p in [Path.cwd(), *Path.cwd().parents][:5]:
    if (p / ".env").exists():
        load_dotenv(p / ".env")
        break

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

deep = json.loads(Path("outputs/exports/deep_findings.json").read_text())
samples = Path("data/processed/samples_analysis.txt").read_text()
robust = Path("data/processed/robustness_interp.txt").read_text()

# Construir un contexto rico y conciso para el LLM
chi = deep["chi_squared_improved"]
strong = deep["strong_associations"][:8]
per_track_axes = deep["per_track_axes"]
per_track_lang = deep["per_track_language"]

ctx = f"""ANÁLISIS DE AQUAMOSH (Plastilina Mosh, 1998) — datos crudos.

== CORPUS ==
- 10 letras transcritas (faltan Ode to Mauricio Garcés y Encendedor en Genius)
- 392 líneas analizables
- Detección de idioma corregida (markers + langdetect) tras encontrar que
  langdetect crudo marcaba como OTHER 78 líneas que en realidad eran ES.

== DISTRIBUCIÓN FINAL DE IDIOMA ==
- MIXED: 147 líneas (code-switching)
- ES:    113 líneas
- EN:    100 líneas
- OTHER:  26 líneas
- FR:      6 líneas
- JA:      0 líneas (Genius no transcribe los samples vocales japoneses)

== CHI² IDIOMA × CAMPO SEMÁNTICO ==
χ² = {chi['chi2']:.1f}, p = {chi['p']:.1e}, dof = {chi['dof']}, n = {chi['n_lines']}
Resultado: rechazo extremadamente fuerte de independencia. Los idiomas NO son
intercambiables; cada uno se asocia con campos semánticos específicos.

== ASOCIACIONES SIGNIFICATIVAS (residuo |z| > 2) ==
"""
for s in strong:
    if abs(s["z"]) > 2:
        ctx += f"  {s['lang']:6s} × {s['campo']:11s}  z = {s['z']:+.2f}\n"

ctx += f"""
== POSICIÓN DE CADA TRACK EN EJES SEMÁNTICOS ==
(positivos = polo derecho del eje; negativos = polo izquierdo)
Ejes: ORIGEN(neg=Monterrey/regio, pos=LA/mainstream),
      SUPERFICIE(neg=emoción, pos=ironía),
      TIEMPO(neg=1998/underground, pos=clásico/retro)

"""
for r in per_track_axes:
    ctx += (f"  {r['title'][:38]:38s}  "
             f"ORIGEN={r['ORIGEN_la']:+.3f}  "
             f"SUPERFICIE={r['SUPERFICIE_ironia']:+.3f}  "
             f"TIEMPO={r['TIEMPO_retro']:+.3f}\n")

ctx += "\n== COMPOSICIÓN LINGÜÍSTICA POR TRACK ==\n"
for r in per_track_lang:
    parts = []
    for k in ["ES", "EN", "FR", "MIXED", "OTHER"]:
        if k in r and r[k] > 0:
            parts.append(f"{k}={int(r[k])}")
    ctx += f"  {r['title'][:38]:38s}  dom={r['dominante']:5s}  H={r['entropia']:.2f}  ({' '.join(parts)})\n"

ctx += f"""

== ANÁLISIS DE SAMPLES (preexistente) ==
{samples[:1500]}

== ROBUSTEZ A COMPRESIÓN DIMENSIONAL ==
{robust[:1000]}
"""

SYSTEM = """Eres crítico musical especializado en rock latinoamericano y producción
discográfica de los 90s. Conoces en detalle: la Avanzada Regia, el contexto post-NAFTA
en Monterrey, la producción de Tom Rothrock/Rob Schnapf, Beck's Odelay, y la relación
entre música alternativa y videojuegos en los 90s. Cada afirmación va respaldada por
evidencia del texto o los datos. No usas frases vacías como 'fusión cultural',
'ecléctico', 'innovador', 'único'. Escribes denso, sin relleno."""

QUESTIONS = """
Mirando los datos arriba, responde con precisión, en español, en este orden exacto:

1) **¿Confirman los datos la tesis original** ("Aquamosh fue un éxito porque resolvió
el problema de sonar global sin renunciar a lo local")? Sé honesto: los datos
muestran una división específica del trabajo entre idiomas, ¿es eso lo mismo que
"global + local" o es otra cosa? Sé concreto sobre el patrón de la tabla chi²
y los residuos. Máx 220 palabras.

2) **¿Qué hace cada idioma en este álbum?** Apóyate en las asociaciones fuertes:
EN×REFERENCIA (+4.62), ES×LUGAR (+3.13), FR×LUGAR (+3.68 pero solo 6 líneas),
MIXED×REFERENCIA (-4.41), EN×CUERPO (-2.78). Da una caracterización funcional
de cada idioma. Incluye la observación de que las únicas líneas en francés
son sobre Viena (Savage Sucker Boy). Máx 180 palabras.

3) **¿Qué dice la cartografía de tracks?** Observa que Monster Truck y Mr. P. Mosh
son los más "LA" del álbum; Niño Bomba y Banano's Bar son los más "regios";
Aquamosh (la canción título) es la más emocional. Esta distribución, ¿es
estratégica o accidental? Considera que Monster Truck y Encendedor entraron
en videojuegos americanos (Street Sk8er), Afroman entró en True Crime LA.
Máx 200 palabras.

4) **El hallazgo de robustez dimensional**: ORIGEN y SUPERFICIE son robustos
hasta dim=256, pero TIEMPO se degrada (ρ=0.37 en dim=256). ¿Qué nos dice esto
sobre la naturaleza de la "datación" semántica de un álbum? Máx 120 palabras.

5) **El párrafo de apertura del blog post** (máx 130 palabras, sin las palabras
fusión / ecléctico / innovador / único). Tiene que enganchar a alguien que
nunca escuchó Aquamosh.

6) **¿Qué pregunta no responde este análisis** y que sería decisiva responder?
Mencionar explícitamente al menos uno: el rol del JA (ausente en transcripciones),
la producción sonora, el contexto post-NAFTA. Máx 100 palabras.
"""

prompt = ctx + "\n\n" + QUESTIONS

# Una sola llamada a gpt-4o
print("Generando síntesis con gpt-4o...")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ],
    temperature=0.5,
    max_tokens=3500,
)
out = resp.choices[0].message.content
print()
print(out)
print()

# Guardar
Path("outputs/exports/deep_synthesis.md").write_text(out, encoding="utf-8")
print(f"\nGuardado en outputs/exports/deep_synthesis.md ({len(out)} chars)")

# Costo aproximado
usage = resp.usage
cost = (usage.prompt_tokens * 2.50 + usage.completion_tokens * 10.00) / 1_000_000
print(f"Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
print(f"Costo aproximado: ${cost:.4f}")
