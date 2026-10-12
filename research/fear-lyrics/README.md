# ¿Dónde vive el miedo? 120 canciones, 1980–2026

Estudio que acompaña al artículo `content/post/2026-10-11-miedo-embeddings-bertopic-rag.md`. Corpus: **120 canciones**, 12 en inglés y 12 en español por período. Corte: **11 de octubre de 2026**.

## Pregunta y respuesta

Cuando una canción habla de miedo, ¿de qué tiene miedo, quién lo siente, cómo responde y qué protege? Dos mediciones independientes contestan lo mismo:

- **Dónde.** La amenaza se muda de la sociedad, el Estado y el monstruo a la pareja y, después, a la propia mente. Lectores: amenaza colectiva −12,9 pts/década (p = 0,01), interior +8,2 (p = 0,02). Embeddings, sin etiquetas: eje afuera→adentro +0,23 DE/década (p = 0,01), estable en cuatro encoders.
- **Quién.** El miedo pasa de narrarse a confesarse: teme quien canta, 40 % → 100 % (2010–19), +12,7 pts/década (p = 0,005).
- **Arco.** Menos escaladas, más giros a la agencia; el bucle persiste.
- **Qué protege.** De estar a salvo, a la dignidad y a sanar; el vínculo amoroso permanece.
- **Lección técnica.** Sin corregir, un embedding agrupa por idioma (100 % de vecinos del mismo idioma). El clustering libre y los vecindarios no recuperan el lugar de la amenaza; un eje anclado con un encoder instruido sí (AUC 0,89).

Límites: las lecturas no son ciegas (el mismo modelo identifica 110 de las 120 letras anonimizadas, y todas las canciones con miedo de 1980 a 2019); selección intencional (sin las canciones elegidas por miedo, la dirección se mantiene pero p = 0,07–0,37); 18 pendientes probadas (con Benjamini–Hochberg, las cinco principales tienen q = 0,065–0,084); 6,5–13,5 canciones con miedo por período; solo letras.

## Leer

- `miedo-embeddings-bertopic-rag.ipynb`: **cuaderno principal**, narrativo y ejecutado sin errores; lee solo `results/story-v3`.
- `static/research/fear-lyrics/story/`: figuras del artículo (SVG y PNG) y `explorador.html`, explorador interactivo de las 56 canciones con miedo.
- `story-reading-protocol.md` (y su traducción `story-reading-protocol.en.md`): protocolo de la lectura anonimizada por pares.
- `semantic-anchors.json`: frases ancla bilingües de los ejes y conceptos, escritas antes de proyectar las letras.
- `anexo-auditoria-metodologica.ipynb`: auditoría de la versión anterior (cuatro encoders sobre 40 canciones, BERTopic, Bayes, RAG con LangGraph). La genera `build_notebook.py`.

## Reproducir

Python 3.12; versiones en `requirements.lock.txt`. Los pasos que leen letras necesitan la caché privada `.research-cache/fear-lyrics/` (ignorada por Git).

```bash
python embed_instruct.py --device mps   # embeddings locales: E5 large instruct (con y sin instrucción) y anclas
python fear_story.py                    # lecturas + geometría → results/story-v3 (solo agregados públicos)
python visualize_story.py               # figuras → static/research/fear-lyrics/story
python build_explorer.py                # explorador interactivo
python build_story_notebook.py          # cuaderno principal, ejecutado
```

Las lecturas se produjeron con seis procesos lectores del asistente sobre lotes ciegos (`.research-cache/fear-lyrics/story-review/batch-L*.json`); `validate_readings.py` comprueba el esquema, los IDs y el orden de los pasos. Ninguna letra se envía a APIs externas en esta versión.

## Resultados públicos (`results/story-v3`)

| Archivo | Contenido |
| --- | --- |
| `song_table.csv` | Una fila por canción: metadatos, votos fraccionales por categoría, puntuaciones en los ejes |
| `readings_public.csv` | Las 240 lecturas: categorías y paráfrasis breves, sin citas |
| `block_states.csv` | Estado de cada bloque por lectura (IDs, sin texto) |
| `reader_agreement.csv` | Acuerdo y κ por campo |
| `trends.csv`, `trend_sensitivity.csv`, `period_means.csv` | Pendientes, permutaciones, q de Benjamini–Hochberg, sensibilidad e intervalos bootstrap |
| `language_neighbors.csv`, `clustering_vs_readers.csv`, `knn_zone_check.csv` | Diagnósticos de idioma, agrupamiento libre y vecindarios |
| `axis_validation.csv`, `axis_anchor_loo.csv`, `concept_validation.csv` | Validación de ejes y conceptos frente a las lecturas |
| `axis_validation_ci.csv` | Intervalos bootstrap del 90 % de los AUC y diferencias pareadas entre encoders |
| `recognition_summary.csv`, `recognition_by_song.csv`, `recognition_agreement.json` | Prueba de reconocimiento: ¿identifica el modelo lector cada letra anonimizada? |
| `context_markers_by_period.csv`, `values_by_period.csv` | Subtemas y valor en juego por período |
| `umap_maps.csv` | Coordenadas UMAP (solo inspección) |

## Privacidad

Las letras completas, las respuestas de proveedores, los lotes de lectura y las matrices de embeddings permanecen en `.research-cache/`. Los materiales públicos contienen IDs, metadatos, categorías, puntuaciones y paráfrasis de hasta unas diez palabras; una comprobación automática descarta coincidencias de cinco palabras con las letras, salvo títulos.
