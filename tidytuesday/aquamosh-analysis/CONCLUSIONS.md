# Aquamosh — Conclusiones empíricas del notebook

**Corpus efectivo:** 10/12 letras transcritas en Genius · 392 líneas analizables · 3 reseñas críticas scrapeadas. Faltan *Ode to Mauricio Garcés* y *Encendedor* (Genius no las tiene).

**Lo que NO se pudo hacer (y por qué):**

| Sección | Estado | Causa |
|---|---|---|
| YouTube playlist + comentarios | bloqueado | YouTube Data API v3 deshabilitada para la API key del proyecto GCP |
| Reddit (PRAW) | omitido | sin credenciales en `.env` |
| Análisis de drift temporal | omitido | depende de YouTube |
| Topic modeling con BERTopic | omitido | depende de YouTube |
| Embeddings Gemini (modelo B) | reemplazado | `GEMINI_API` expirado — pivote a *robustez por dimensión* con OpenAI |
| Detección de japonés | sin datos | Genius no transcribe los samples vocales japoneses del álbum |

---

## 1 · La prueba estadística central

**Idioma × campo semántico no son independientes:**

```
χ² = 104.50 · dof = 21 · p = 4.6 × 10⁻¹³ · n = 356 líneas
```

Rechazo extremadamente fuerte de la hipótesis nula. La elección de qué decir en qué idioma **no es aleatoria** en *Aquamosh*. Hay una división de trabajo lingüística.

**Residuos estandarizados (|z| > 2 = asociación significativa):**

| Combinación | z | Interpretación |
|---|---|---|
| **EN × REFERENCIA** | **+4.62** | El inglés es el idioma de los nombres propios y citas culturales ("Woody Allen's world", "Afroman", "Mr. P. Mosh") |
| **MIXED × REFERENCIA** | **−4.41** | Cuando referencian cultura, NO mezclan idiomas — eligen lado |
| **FR × LUGAR** | **+3.68** | El francés (6 líneas, todas en *Savage Sucker Boy*) marca lugar: Viena |
| **ES × LUGAR** | **+3.13** | El español ancla geografía: "Desde África querida", "Para América Latina" |
| **EN × CUERPO** | **−2.78** | El cuerpo NO se nombra en inglés puro — se mete en líneas mixtas |

**Pattern emergente:**

- **Inglés = registro cultural-global**: referencias, nombres, acciones.
- **Español = registro territorial**: lugares, identidad, geografía.
- **Francés = exotismo de marca**: una sola escena (Viena) en una sola canción.
- **Mixed/code-switching = registro íntimo**: cuerpo (z=+1.84), emoción (+1.87), marca (+1.70). Es el modo cuando hablan de lo personal y lo comercial — pero NUNCA cuando citan cultura.

---

## 2 · Cartografía del álbum (3 ejes culturales)

Proyección sobre ejes anclados (Kozlowski et al., *The Geometry of Culture*, ASR 2019):

| Track | ORIGEN_la (regio→LA) | SUPERFICIE (emoción→ironía) | TIEMPO (1998→retro) |
|---|---|---|---|
| **Monster Truck** | **+0.082** ←más LA | **+0.054** ←más irónico | −0.129 |
| I've Got That Milton Pacheco | +0.070 | +0.006 | −0.003 |
| Mr. P. Mosh | +0.010 | −0.014 | −0.011 |
| Aquamosh | −0.005 | **−0.149** ←más emocional | −0.002 |
| Bungaloo Punta Cometa | −0.021 | −0.084 | −0.017 |
| Pornoshop | −0.034 | −0.093 | **+0.029** ←más retro |
| Savage Sucker Boy | −0.042 | −0.035 | −0.040 |
| Afroman | −0.046 | −0.041 | −0.004 |
| Banano's Bar | −0.058 | −0.062 | −0.018 |
| **Niño Bomba** | **−0.060** ←más regio | +0.012 | −0.050 |

**Lectura:**

- **Polo "global"**: *Monster Truck*, *Mr. P. Mosh*, *I've Got That Milton Pacheco* — los tracks que más se "venden" a la audiencia anglo. Coincide con los tracks que **entraron a videojuegos americanos** (*Street Sk8er* tomó *Monster Truck*; *True Crime: Streets of LA* tomó *Afroman* — aunque éste último cae en el lado "regio" del eje).
- **Polo "regio"**: *Niño Bomba*, *Banano's Bar*, *Afroman*. Son los tracks con más densidad de español y con identidad mexicana explícita ("Para América Latina").
- **Más emocional**: *Aquamosh* (la canción título). Ironía bajísima.
- **Más irónico**: *Monster Truck* — el track del videojuego.

---

## 3 · Composición lingüística por track

| Track | Dominante | H | ES | EN | FR | MIXED |
|---|---|---|---|---|---|---|
| Afroman | ES | 1.31 | 67 | 8 | 0 | 12 |
| Monster Truck | EN | 0.85 | 2 | 31 | 0 | 4 |
| Mr. P. Mosh | EN | 1.68 | 11 | 24 | 0 | 12 |
| Pornoshop | ES | 1.58 | 10 | 2 | 0 | 6 |
| Savage Sucker Boy | MIXED | **1.73** | 5 | 12 | 6 | 17 |
| Aquamosh | EN | 1.60 | 2 | 9 | 0 | 18 |
| Bungaloo Punta Cometa | MIXED | 1.48 | 9 | 0 | 0 | 9 |
| Niño Bomba | MIXED | 1.27 | 3 | 11 | 0 | 19 |
| I've Got That Milton Pacheco | MIXED | 1.50 | 0 | 3 | 0 | 8 |
| Banano's Bar | MIXED | 1.10 | 4 | 0 | 0 | 9 |

*H = entropía de Shannon de la distribución de idiomas. Mayor = más multilingüe.*

**El track más multilingüe es *Savage Sucker Boy* (H=1.73)** — y es justamente el único que incluye francés. *Monster Truck* es el menos multilingüe (H=0.85, casi puramente EN) — y es el que se exportó a videojuegos.

> **Hallazgo no obvio**: los tracks más populares globalmente son lingüísticamente menos arriesgados. La "cuadrilingüidad" del álbum se concentra en los tracks de menor exposición comercial. Eso refuta la lectura ingenua de "el álbum es exitoso *porque* es multilingüe" — los tracks más multilingüe NO son los exportados.

---

## 4 · Robustez del análisis a compresión dimensional

Mismo `text-embedding-3-large` truncado vía matryoshka:

| Eje | ρ(256, 3072) | ρ(512, 3072) | ρ(1024, 3072) |
|---|---|---|---|
| ORIGEN (geo) | 0.78 | **0.98** | 0.98 |
| SUPERFICIE (afecto) | 0.89 | 0.95 | 0.95 |
| **TIEMPO (datación)** | **0.37** | 0.65 | 0.75 |

| Dim | Pearson r de matriz de similaridad vs 3072 |
|---|---|
| 256 | 0.974 |
| 512 | 0.986 |
| 1024 | 0.994 |

**Lectura:** los ejes espaciales y afectivos son **robustos** a la compresión (ρ≥0.78 hasta 256 dims). La datación semántica (¿es esto "1998" o "clásico"?) **colapsa** bajo compresión. Esto tiene sentido teórico: lugar/emoción son señales lexicales relativamente discretas; la temporalidad cultural vive en correlaciones más finas y distribuidas.

---

## 5 · ¿Confirma la tesis?

**Tesis original:** *Aquamosh fue un éxito porque resolvió el problema de sonar global sin renunciar a lo local.*

**Veredicto empírico:** **parcialmente, pero con matiz importante.**

Los datos NO muestran *fusión* (lo que sugiere el verbo "resolver"). Muestran **división del trabajo lingüístico**:

- Lo "global" se hace en inglés y se concentra en tracks específicos (Monster Truck, Mr. P. Mosh).
- Lo "local" se hace en español y se concentra en otros (Afroman, Niño Bomba).
- El cuerpo, la emoción y las marcas viven en MIXED — el código de la intimidad.
- Las citas culturales viven en EN puro — el código de la conexión global.

> El álbum no resolvió el dilema global/local — **lo separó por canales** y los puso a convivir en el mismo objeto. La estrategia no fue mezclar; fue **modular**.

Esa es una tesis más interesante porque es falsable y los datos la sostienen.

---

## 6 · Lo que NO puede responder este análisis

- **El rol del japonés**: el álbum es famoso por incluir JA, pero Genius transcribe letras y los samples vocales japoneses no entran. Para responder esto haría falta audio analysis (Whisper en pasajes específicos, o lyrics manualmente transcritas desde el booklet).
- **El efecto de la producción (Rothrock/Schnapf)**: los datos lyrics-first no capturan timbre, mezcla, ni gestos sónicos. La hipótesis de que la producción anglófona "globalizó" el sonido sin tocar el contenido lingüístico es testeable, pero no aquí.
- **El contexto post-NAFTA**: la diferencia entre lo que el álbum podía decir en 1998 vs lo que un álbum análogo diría hoy. Requiere comparar con álbumes Latin Alternative pre-2000 vs post-2010.
- **Recepción real**: sin YouTube ni Reddit, no medimos cómo los fans hablan del álbum hoy.

---

## 7 · Attention Windows — OpenAI vs Google LaBSE

Ver `THEORETICAL_FRAMEWORK.md` para el desarrollo completo. Resumen:

**Setup:** se generaron embeddings de las 392 líneas con dos modelos —
OpenAI `text-embedding-3-large` (3072d) y Google **LaBSE** (768d, multilingual-aware). Umbral
$\theta$ calibrado por modelo (mediana + 1 SD de pares aleatorios) para
geometrías comparables: $\theta_{OA}=0.32$, $\theta_{LB}=0.32$.

**Hallazgo central — tasa de ruptura de ventana según transición de idioma:**

| Tipo de transición | OpenAI | LaBSE | Δ entre modelos |
|---|---|---|---|
| same language | 0.36 | 0.41 | +0.05 |
| **lang switch** | **0.70** | **0.65** | −0.05 |
| Salto switch − same | **+0.34** | **+0.24** | — |

**Lectura:**
- **Ambos modelos confunden cambio de idioma con cambio de tema** — la tasa de ruptura casi se duplica en transiciones lingüísticas.
- **LaBSE es 30 % menos reactivo al switch lingüístico** que OpenAI. Su entrenamiento con corpus paralelo atenúa el sesgo distribucional, pero no lo elimina.
- Spearman OpenAI ↔ LaBSE = 0.64 (moderado, no fuerte) — los modelos coinciden en el ranking grueso pero discrepan donde hay mezcla.
- *Savage Sucker Boy* — el único track con francés y el más multilingüe — es donde LaBSE produce ventanas más del doble de largas que OpenAI (2.45 vs 1.13). Eso valida que el modelo multilingüe-aware "atraviesa" los switches con menos rupturas.

**Implicación teórica:** Aquamosh, por su naturaleza cuadrilingüe, **convierte el problema señalado en el post Beatles vs Floyd en falsable**. La distinción "continuidad léxica vs continuidad conceptual" deja de ser retórica — se mide directamente como un Δ de 34 puntos en la tasa de ruptura. El álbum funciona como dispositivo crítico de la herramienta de medición, no solo como objeto de estudio.

---

## 8 · Resumen ejecutivo

| Hallazgo | Confianza | Visualización |
|---|---|---|
| Idiomas no son intercambiables (p<1e-12) | **alta** | `language_field_residuals_v2.png` |
| EN domina referencias culturales; ES domina lugar | **alta** | residuos +4.62 / +3.13 |
| Code-switching es para cuerpo, emoción, marca | media-alta | residuos +1.84/+1.87/+1.70 |
| Tracks exportados (videojuegos) son los menos multilingüe | media | `language_by_track.png` |
| Aquamosh (track título) es el más emocional | media | `semantic_axes_map.png` |
| Análisis robusto a dim=256 excepto eje TIEMPO | meta | `dimension_robustness.png` |
| **Switch lingüístico duplica tasa de ruptura de ventana** | **alta** | `aw_break_rates.png` |
| **LaBSE 30% menos reactivo al switch que OpenAI** | media-alta | `aw_by_language.png` |
| Savage Sucker Boy: máxima divergencia entre modelos | media | `aw_per_track.png` |

---

## Archivos clave producidos

- **Notebook ejecutado**: `aquamosh_analysis.executed.ipynb` (71 celdas, cero errores)
- **Datos**: `outputs/exports/corpus_lyrics.parquet`, `corpus_lines.parquet`, `corpus_lines_v2.parquet`
- **Hallazgos**: `outputs/exports/findings_summary.json`, `deep_findings.json`
- **Síntesis**: `outputs/exports/deep_synthesis.md` (escrita por gpt-4o sobre datos)
- **Figuras**: `outputs/figures/{language_mosaic, language_field_residuals_v2, semantic_axes_map, dimension_robustness, language_by_track}.png`
- **Token streams**: `outputs/html/tokenstream_*.html` (10 archivos)

---

## Próximos pasos sugeridos

1. **Renovar la `GEMINI_API`** en https://aistudio.google.com/apikey y volver a correr el modelo-dueling real OpenAI vs Gemini.
2. **Habilitar la YouTube Data API v3** en Google Cloud Console para el proyecto del usuario, o quitar restricciones del API key. Eso desbloquea Secciones 4 y 7.
3. **Transcribir manualmente las partes en japonés** desde el booklet o usar Whisper en los segmentos identificados.
4. **Escribir el blog post** en `content/post/2026-05-20-aquamosh-anatomia-cuadrilingue.md` usando estas conclusiones y las figuras.
