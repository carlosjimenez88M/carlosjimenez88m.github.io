# Auditoría editorial del estudio

Revisión: **11 de octubre de 2026**. Artículo revisado:
`content/post/2026-10-11-miedo-embeddings-bertopic-rag.md`.

Esta es una revisión del asistente de investigación sobre correspondencia entre
afirmaciones, salidas guardadas y alcance del diseño. No constituye adjudicación
humana de letras, anotación independiente ni revisión científica externa.
Las comprobaciones de cifras leen resultados existentes; no ajustan modelos,
umbrales ni preguntas después de observar el test.

## Correspondencia entre afirmación y resultado

### Revisión visual y foco del 12 de octubre de 2026

Se añadieron diagnósticos derivados de los vectores cacheados, votos y trazas existentes, sin llamadas remotas ni reentrenamiento de tópicos. El notebook ejecutado tiene ahora 21 celdas de código y cero errores. Los recibos acumulados de esta revisión están en `phase-manifest.json`.

| Pregunta y afirmación | Fuente vigente | Resultado y alcance |
|---|---|---|
| Qué letras refieren al miedo | `results/historical-map/fear-focus-summary.json` | 12 explícitas, 11 implícitas, 10 ausentes y 7 sin resolución; panel de 40, no 120 canciones anotadas. Referencia incluye negación y posibilidad. |
| Qué arcos tienen las 23 referencias al miedo | La misma fuente; votos originales en `pilot-comparison/results/arc-panel/fieldwise-posthoc/` | 6 repeticiones, 1 giro, 1 relación mixta y 15 sin resolución. Los 12/14 de repetición corresponden al conjunto completo y no se trasladan a este subconjunto. |
| Cómo comparan los embeddings las mismas canciones | `results/encoder-comparison-new/receipt.json`, `model_neighbor_overlap.csv` | Centroides de bloques únicos en espacios completos. Large/small comparten 75,5 % de cinco vecinos; es coincidencia de identidades, no comparación de cosenos entre modelos. |
| Si esa cercanía coincide con lecturas del miedo | `results/encoder-comparison-new/fear_neighbor_checks.csv`, `fear_label_check_method.json` | Candidatos del mismo idioma; etiquetas sin resolución omitidas después de recuperar. Large: 85/133 en relevancia, frente a 54,53 % de referencia uniforme condicionada; no exactitud ni superioridad estadística. |
| Qué varianza tiene un tópico | `figures/bayes-readable-receipt.json`, tablas de momentos y varianza temporal | Var(c/n) de cinco períodos y Var(θ\|datos) exacta de Beta/mezcla jerárquica se distinguen de la varianza de coordenadas PCA. Ningún ancho de intervalo sustituye la varianza. |
| Qué hace el proceso LangGraph | `results/langgraph-process-new/receipt.json` y eventos fuente | 768 trazas auditadas. Gráfico principal: 192 test dense. La fase mini/nano usa contextos híbridos separados; no se inventa una continuación del flujo. |
| Qué representa el valor subjetivo | `results/historical-map/evolution-data.json`: cinco casos | Interpretaciones editoriales de seguridad, vínculo y cuidado, acompañadas de pasos de un lector con anclajes válidos. Consenso de tipo de arco no implica consenso de todos sus estados ni novedad histórica del valor. |

### Auditoría histórica del piloto

Las rutas de la primera tabla que empiezan por `results/` se resuelven dentro de
`research/fear-lyrics/pilot-40/`. Las tablas de extensión posteriores utilizan
las rutas actuales explícitas, relativas a `research/fear-lyrics/`.
La primera tabla conserva la auditoría del piloto original de dos encoders. Las
filas de costes son recibos históricos de esa fase; la extensión al final registra
los cuatro encoders, el corpus ampliado y las llamadas posteriores. La copia en
`pilot-40/` permanece congelada.

| Afirmación del artículo | Archivo y medida comprobados | Resultado | Límite de interpretación |
|---|---|---|---|
| 40 canciones, cuatro EN y cuatro ES por cada período; 34 identidades de artista | `results/corpus_diagnostics.json`: `songs_with_lyrics`, `decade_language_cells`, `artists` | 40; diez celdas con cuatro canciones; 34 | Selección intencional; colaboraciones conservadas. No representa mercados ni prevalencia histórica. |
| Corpus hasta el 11/10/2026, período y año finales parciales | `results/corpus_diagnostics.json`: `cutoff`, `partial_final_decade`, `partial_final_year` | `2026-10-11`; ambos `true` | Una publicación de 2026 no caracteriza todo el año. No equiparar exposición temporal de un año con una década. |
| LRCLIB disponible en 39 consultas, Genius en 34 y lyrics.ovh en cero | `pilot-40/provider-coverage.csv`: filas con `status == 200`, agrupadas por proveedor | 39 / 34 / 0 de 40 | Cobertura del colector y consultas concretas. No mide calidad global del servicio ni condiciones de otras cuentas. |
| 33 coincidencias de proveedores y mediana de similitud 0,936 | `pilot-40/provider-agreement.csv`: número de filas y mediana de `sequence_similarity` | 33; 0,9358974359 | Acuerdo de textos normalizados; no exactitud ni independencia de procedencia. |
| Edición de Zombie y timestamps de Aperture corregidos antes de análisis | `pilot-40/source-audit.json`; `pilot-40/protocol.md`; `audit_sources.py` | Correcciones de fuente y limpieza registradas con hashes | Se conservan límites de edición y fuentes originales privadas. Disponibilidad no demuestra corrección editorial completa. |
| Corpus analítico final usa 38 registros LRCLIB y dos Genius | `results/corpus_diagnostics.json`: `providers` | `lrclib: 38`, `genius: 2` | No contradice la cobertura 39/34: disponibilidad y selección de fuente son medidas diferentes. |
| 461 bloques, 386 únicos y 75 repeticiones exactas | `results/corpus_diagnostics.json`: `blocks`, `unique_blocks`, `exact_repeated_blocks` | 461 / 386 / 75 | Deduplicación exacta dentro de canción. Ventanas de ocho líneas no son estrofas verificadas; bloques no son observaciones independientes. |
| Ventanas de 128 tokens MiniLM y 512 E5; truncamiento de dos bloques y cero | `results/<modelo>/model_manifest.json`: `encoder.max_seq_length`, `truncated_documents` | MiniLM 128/2; E5 512/0 | Límites de esta implementación y tokenizer. No generalizar a todos los modelos que contienen “MiniLM”. |
| Veinte tripletas sintéticas originales | `results/<modelo>/semantic_probes.json`: longitud de `probes` | 20 por encoder | Diagnósticos redactados por el asistente; no labels emocionales humanos ni réplica de ALIGN-SIM o Sentence Smith. |
| Inversión de roles mantiene coseno de aproximadamente 0,98–0,99 | `semantic_probes.json`: `role_en`, `role_es`, `related_cosine` | MiniLM 0,979600/0,985573; E5 0,989925/0,990926 | Relación temática compartida no equivale a fidelidad al experimentador. No se exige coseno bajo como gold universal de contradicción. |
| Control de orden: MiniLM 0,700/0,706 y E5 0,872/0,870 | `semantic_probes.json`: `order_en`, `related_cosine`, `contrast_cosine` | MiniLM 0,699892/0,706110; E5 0,872257/0,870207 | Un ejemplo, márgenes pequeños; no constituye una prueba general de comprensión narrativa. |
| Control largo empata candidatos con distinto final | `semantic_probes.json`: `long_ending_es`, `related_minus_contrast` | 0,0 en ambos encoders | El final queda fuera de la ventana. No atribuir el empate a una comprensión fallida de texto que el encoder no recibió. |
| BERTopic con MiniLM tiene 19 topics y 75 bloques sin asignar; E5 dos y cero | `results/<modelo>/topics.json`, `section_assignments.csv`, semilla 17 | 19/75 y 2/0 | “Topic” identifica cluster de esa ejecución. `-1` es ruido, no una emoción; cantidad no mide calidad. |
| ARI entre semillas 0,516–0,585 MiniLM y 1,000 E5 | `results/<modelo>/seed_stability.csv`: `ari_including_noise` | MiniLM 0,515860–0,584913; E5 1,0 | Estabilidad de partición no valida pertinencia cultural ni semántica. Los ARI incluyen ruido. |
| Asociación topic–idioma NMI 0,113 y 0,863 | `results/topic_language_association.csv`: `nmi_topic_language` | MiniLM 0,112719; E5 0,863072 | Diagnóstico posterior al ajuste, no clase supervisada ni causalidad del idioma. |
| Acuerdo entre encoders ARI 0,033 | `results/embedding_model_cluster_agreement.csv`: `ari_including_noise` | 0,032632 | Particiones de los mismos bloques; IDs arbitrarios no alinean topics automáticamente. |
| E5 agrupa 196 bloques EN frente a 181 ES y nueve EN | `results/intfloat--multilingual-e5-small/section_assignments.csv`, unido a inventario/manifiesto | 196 EN; 181 ES + 9 EN | Factor de confusión concreto del corpus, modelo y clustering. No significa que E5 siempre separe idiomas. |
| OpenAI embeddings usa 1.024 dimensiones y el mismo corpus | `results/openai--text-embedding-3-large/model_manifest.json`: `encoder.dimensions`, `unique_documents`, `provider_metadata.usage` | 1.024; 386; 470 requests, 23.419 tokens y USD 0,00304447 | Se solicitó dimensionalidad reducida; no es la dimensión nativa por defecto. Modelo alojado sin revisión de pesos fijada; caché conserva la ejecución concreta. |
| OpenAI produce 11 topics, 14 bloques de ruido y ARI 0,942–0,946 | `topics.json`, `seed_stability.csv`: semilla 17 y `ari_including_noise` | 11/14; mínimo 0,942085 y máximo 0,946182 | Tres particiones sobre los mismos casos; estabilidad no valida miedo, arcos ni pertinencia de categorías. |
| OpenAI obtiene preferencia prevista en 14/14 controles elegibles | `semantic_probes.json`: `preferred_context_ranks_first` distinto de null | OpenAI 14/14, MiniLM 11/14, E5 13/14 | Controles sintéticos pequeños; seis casos no tienen preferencia binaria prevista. No exactitud emocional ni comparación de proveedores de generación. |
| OpenAI conserva el desenlace fuera de ventana local | `semantic_probes.json`: `long_ending_es`, `related_cosine`, `contrast_cosine`, conteos de tokenizer | 0,574142 frente a 0,525621; 2.058/2.062 tokens de entrada y sin truncamiento | Presencia del desenlace y un margen favorable no prueban comprensión general de arcos; magnitudes de coseno son internas a cada encoder. |
| OpenAI sigue asociado al idioma y concentra un grupo grande | `results/topic_language_association.csv`; `topics.json` topic 0 | NMI 0,508323; 173 bloques de 20 canciones | El mismo balance del corpus no implica representación sin confusión de idioma; tamaño no equivale a tema coherente. |
| El orden de trayectorias OpenAI no supera el umbral ajustado | `narrative_order_tests.csv`: `unique_first_occurrence`, `all_selected_songs`; `narrative_song_order_tests.csv` | Global q 0,107892; ninguna canción q < 0,05 dentro de variante | Prueba condicionada a bloques, prototipos y permutabilidad. No ausencia demostrada de arcos narrativos. |
| Juanes concentra aproximadamente 71 % de topic 13 y Linkin Park 67 % de topic 12 | `results/sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/artist_concentration.csv`: `largest_artist_share` | 0,708661 / 0,670290 | Masa normalizada por canción dentro del corpus; no porcentaje bruto de versos ni prevalencia de un estilo. |
| Máxima sensibilidad temporal equilibrada a repeticiones, unos cuatro puntos porcentuales; intervalo incluye cero | `repeated_refrain_sensitivity.csv`: `language == balanced_available_languages`, máximo absoluto de `repeated_minus_unique` | 0,040388; IC −0,007802 a 0,100271 | Es topic `-1` en 2000 y ambos grupos fijados. No es una diferencia causal histórica ni permiso para ignorar el estribillo. |
| Ninguna canción alcanza q < 0,05 en la prueba de orden | `results/<modelo>/narrative_song_order_tests.csv`: filas `tested`, `permutation_q_bh_within_variant` | Cero en ambos modelos y variantes | Ausencia de señal ajustada en pendiente geométrica; no prueba ausencia de trama o arcos. |
| Agregado de bloques únicos q=0,096 MiniLM y 0,196 E5 | `results/<modelo>/narrative_order_tests.csv`: `group == all_selected_songs`, `variant == unique_first_occurrence` | 0,095904 / 0,195804 | Contraste condicionado a textos, prototipos y supuesto de permutabilidad; ajuste dentro de variante. |
| 32 preguntas; test 16 positivas y ocho negativas | `results/rag/questions.json`: `split`, `answerable`, `family` | Desarrollo 6 positivas/2 negativas; test 16/8 | Test tiene ocho familias positivas y cuatro negativas. Traducciones EN/ES son parejas, no réplicas independientes. |
| Recall@5 BM25 0,365→0,542; MiniLM denso 0,771→0,833; híbrido 0,625→0,792 | `results/rag/runs.json`: medias `recall_at_5` entre test positivos por método/brazo | 0,364583→0,541667; 0,770833→0,833333; 0,625000→0,791667 | Ranking anterior al gate, incluso ante abstención; no generación, contexto final ni ablation aislada de vecinos. |
| Recall@5 E5 denso 0,573→0,719; híbrido 0,521→0,719 | `results/rag-e5/runs.json`: misma agrupación | 0,572917→0,718750; 0,520833→0,718750 | No clasificación universal de modelos. Mismo corpus/preguntas; sensibilidad a formulación y qrels provisionales. |
| Todos los brazos contextuales aceptan las cuatro negativas difíciles | `results/rag*/runs.json`: familias `intruder_identity_missing`, `bedtime_clinical_missing`, `abstained` | Cero abstenciones de cuatro por modelo y método | Aceptación de un paquete candidato por reglas; no alucinación emitida por un generador LLM. |
| MiniLM denso: abstención negativa correcta 8/8→4/8 | `results/rag/runs.json`: test negativo, `method == dense`, `negative_safe_abstention` | 1,0→0,5 | Ocho preguntas de cuatro familias. Las cuatro negativas fáciles y cuatro difíciles tienen alcances diferentes. |
| Intervalos de diferencia en recall del contexto final incluyen cero | `results/rag_family_sensitivity.csv`: `metric == context_support_recall`, `ci_low`, `ci_high` | Todos los seis pares incluyen cero | Bootstrap pareado de 5.000 remuestras de familias, condicionado a ocho familias positivas; no intervalo poblacional. |
| Máximo común de 30 candidatos devueltos, 320 palabras y cinco bloques | `results/rag*/protocol.json`; campos `candidate_slots`, `context_words`, `context_ids` en `runs.json` | Ninguna corrida supera límites; hasta dos reformulaciones | Tope de salida del retriever, no de documentos puntuados ni de tiempo. Actual spend y anchura por llamada difieren. |
| Descomposición posterior 2×2: reformulación y vecinos, con ancho fijo diez | `results/rag-ablation/protocol.json`, `results/rag-e5-ablation/protocol.json` y `runs.json` | 384 corridas por encoder; mismas 32 preguntas, qrels y umbral | Diagnóstico posterior a observar el test; no validación independiente ni reparación confirmatoria. |
| Con reformulación, retirar vecinos aumenta recall de contexto MiniLM denso 0,479→0,583 | `results/rag-ablation/runs.json`: test positivo, `dense`, `rewrite_1_neighbors_1` frente a `rewrite_1_neighbors_0`, media `context_support_recall` | 0,479167→0,583333 | Ranking y abstenciones iguales; efecto sobre selección de bloques en este presupuesto y gold provisional. |
| Con reformulación, contexto E5 denso 0,542→0,573 e híbrido 0,542→0,615 al retirar vecinos | `results/rag-e5-ablation/runs.json`: mismos filtros y pares | Denso 0,541667→0,572917; híbrido 0,541667→0,614583 | No implica que contexto vecino sea inútil en otras preguntas ni que las referencias capturen todo el contexto necesario para leer. |
| Las corridas RAG originales y su ablation son offline | `results/rag*/protocol.json`, `runs.json` | Cero API calls de inferencia y $0 de tarifa API dentro de esas corridas | No describe las fases posteriores alojadas; adquisición de letras y tiempo local son costes distintos. |
| OpenAI revisó 24 contextos test ya congelados | `results/llm-review/summary.json`: `review[provider==openai]` | 24 válidas; 16 positivas y ocho negativas; coste USD 0,020694 | Fase posterior al test y basada en anotaciones del asistente. No comparación de proveedores ni validación humana independiente. |
| El revisor se abstuvo en las ocho negativas | `results/llm-review/reviews.json`: `assistant_answerable==false`, `abstained` | 8/8, cuatro familias; incluye cuatro preguntas de dos familias difíciles | Las parejas EN/ES no son ocho casos independientes. No garantiza seguridad general ni mide factualidad de una respuesta extensa. |
| Las ocho abstenciones positivas no son ocho errores equivalentes | `results/llm-review/summary.json`: `positive_complete_contexts`, `complete_context_abstentions`, `positive_incomplete_contexts`, `incomplete_context_abstentions` | Todos los soportes designados: 1 abstención entre 8; contextos incompletos: 7 entre 8 | “Completo” significa todos los soportes provisionales del asistente; pueden ser redundantes o no exhaustivos. No equivalencia automática con entailment humano. |
| IDs citados pertenecen al contexto | `results/llm-review/summary.json`: `invalid_citation_reviews` | Cero de 24 | Comprueba pertenencia de IDs, no que cada afirmación esté implicada por su contenido. Las afirmaciones modelo permanecen privadas. |
| Anthropic ejecutó veinte pruebas originales sintéticas | `results/llm-review/anthropic-synthetic.json`, `summary.json`: estado y coste | 20/20 ejecutadas; USD 0,037317 | No recibió letras, no produjo embeddings y no se adjudicó exactitud global de estas etiquetas. |
| El juez detectó negación/roles/orden pero también sobreinterpretó contradicción | `anthropic-synthetic.json`: negation, role_reversal, order_en y translation_en_es/translation_es_en | AB contradicción en dos negaciones, AB inversión de roles en dos casos y AC inversión de orden; AC contradicción en dos contrastes de actividades que pueden coexistir | La taxonomía del juez debe auditarse: estar temeroso y disfrutar música no son hechos necesariamente contradictorios. |
| Etiquetas semánticas se proponen después del clustering | `results/topic_labels.json`, `topic_labels_protocol.json` | 30 clusters y 30 propuestas publicadas; formato 2–6 palabras tras revisión editorial de una etiqueta originalmente de dos palabras; USD 0,02992725 | Prompt original 3–6; un intento de reparación histórico conservado y ninguna llamada por la relajación. Tres representantes como máximo, no todo el cluster; c-TF-IDF sigue separado. No validación humana, probabilidades calibradas ni subtopics descubiertos. |
| Fase LLM y etiquetas bajo techo autorizado | `results/llm-review/summary.json`: `budget` | USD 0,08793825; 75 requests, cero facturación incierta | Coste derivado de tokens comunicados y tarifa estándar, no recibo bancario. El ledger es por fase; OpenAI embeddings se suma por separado al total autorizado USD 1. |
| Bayes cuenta presencia por canción, con cuatro canciones por celda | `results/bayes/cell_counts.csv`, `song_presence.csv`, `protocol.json`: `observation`, `trials`, `estimand` | Una Bernoulli por canción/topic; diez celdas con n = 4; ceros conservados | La fracción finita c/n es conocida exactamente. El posterior corresponde a θ bajo un modelo hipotético de intercambiabilidad; selección deliberada no establece prevalencia externa. |
| Posterior plug-in, jerárquico y Jeffreys difieren en T12/EN/2000 | `results/bayes/posterior_rates.csv`: MiniLM, topic 12, en, 2000, `prior_context==pooled_languages` o `fixed` | 2/4; EB 0,141666 [0,077991;0,220384], jerárquico 0,279335 [0,061997;0,667738], Jeffreys 0,5 [0,122754;0,877246] | Intervalos 95 % condicionados al clustering; la diferencia jerárquica añade hiperpriors y no es sólo corrección de incertidumbre en el mismo máximo de likelihood. |
| Integración por cuadratura utiliza hiperpriors y límites explícitos | `results/bayes/protocol.json`: `full_hierarchical`, `grid`, `credible_intervals` | μ Beta(1,1), log κ Normal(log4,1,5), κ [0,05;200]; grid 64×81, control 96×121 | Mayor precisión numérica de una integral no valida intercambiabilidad, familia de prior, selección de temas o independencia de artistas. |
| La estimación EB es frágil con pocas celdas | `results/bayes/summary.json`, `optimization_diagnostics.json`, `leave_one_cell_out.csv` | 22/34 fits conjuntos en límite; dos fits por idioma no convergen; mayor cambio leave-cell-out 0,252925 | Retirar celdas mantiene topic discovery con todas las canciones: sensibilidad condicional, no predicción de topics fuera de muestra. Valores con banderas siguen visibles. |

## Defectos de comparabilidad que deben seguir visibles

### El ranking no es el contexto final

`recall_at_5` se calcula sobre los primeros cinco IDs de `all_ranked` después de
recuperar, antes de que el gate permita emitir un paquete. Una abstención puede
tener recall alto. `context_support_recall` usa los bloques realmente suministrados
después del gate y del límite de contexto; una abstención tiene contexto vacío.
La validación de citas solo comprueba pertenencia de IDs, no implicación semántica.

En cada consulta híbrida, RRF combina listas BM25/dense con constante 60.
La pasada única combina listas de hasta 30; una llamada contextual, de hasta diez.
Después de reformular, el ranking nuevo precede a los IDs únicos de intentos
anteriores. No hay fusión RRF entre intentos, suma de hits ni boost por repetición.
El brazo cambia consulta, anchura, gate y vecinos a la vez. No permite adjudicar
la diferencia de recall a un único mecanismo ni a “más inteligencia”.

Los índices puntúan todos los bloques en cada búsqueda; el límite de 30 cuenta
candidatos devueltos. Las dos versiones tienen el mismo máximo de contexto, pero
no exactamente el mismo consumo realizado, número de búsquedas o coste CPU.
Las latencias se miden offline con todas las consultas permitidas previamente
vectorizadas; no comparan servicio API ni inferencia de planificación/generación.

La descomposición posterior separa vecinos y reformulación mediante un diseño
2×2 con ancho fijo diez por llamada. Dentro de cada pareja con la misma política
de reformulación, los vecinos no intervienen en ranking ni gate. Su diferencia
de contexto puede atribuirse a esa selección dentro del presupuesto. Se ejecutan
768 corridas adicionales en directorios separados, sin sustituir las 384 corridas
primarias. Elegir una variante a partir de esos resultados es desarrollo posterior;
haría falta un nuevo benchmark para confirmar su mejora.

No se implementó una regla específica para arreglar las cuatro negativas ya
observadas. Un futuro gate debe distinguir una pregunta sobre palabras de una
letra de una sobre hechos verificados, exigir apoyo del mismo sujeto y conservar
negación e incertidumbre. Bloquear por la sola presencia de “diagnóstico”, “nombre”
o “fecha” produciría falsas abstenciones. Antes de evaluarlo, conviene especificar
controles nuevos con datos ficticios y fuentes apropiadas, fuera de estas familias.

### La partición protege preguntas, no canciones

Las versiones EN/ES de una familia comparten partición. El desarrollo contiene
tres familias positivas y una negativa; el test, ocho positivas y cuatro negativas.
Esto evita colocar la traducción directa de la misma pregunta en ambas partes,
pero no crea validación de canciones o artistas inéditos. Por ejemplo, vigilancia
y el intruso de la ducha comparten una canción en familias de splits distintos.

La selección del corpus, las preguntas, el glosario de expansión y los IDs de
soporte fueron propuestos por asistentes de investigación. La anotación primaria
de las preguntas carece de adjudicación externa y de referencias exhaustivas.
Los votos posteriores de modelos son una medición adicional, distinta de esa
anotación. Las fichas de diez canciones se escribieron después del ajuste
exploratorio y tampoco funcionan como validación externa de los clusters.
El bootstrap conserva juntas las versiones bilingües, pero no corrige selección
de preguntas, dependencia por canción, confirmación del diseñador ni falta de
gold humano. Hace visible sensibilidad condicional; no certifica generalización.

### La unidad histórica sigue siendo pequeña

Los bloques son dependientes dentro de canción. Remuestrear canciones por idioma
es más apropiado que remuestrear versos, pero cuatro canciones por celda siguen
siendo cuatro casos deliberados. El control de concentración por artista y su
retirada describe composición con los topics fijados; no reentrena BERTopic sin
ese artista. La sensibilidad entre encoders y el diagnóstico de idioma son
motivos para no presentar una taxonomía histórica confirmada.

## Bibliografía técnica: alcance y fechas

Se comprobaron fuentes primarias publicadas antes del corte. Las fechas de
consulta de documentación mutable no se presentan como fechas de publicación.

| Fuente | Fecha verificable | Uso defendible |
|---|---|---|
| [Reagan et al., seis formas afectivas](https://link.springer.com/article/10.1140/epjds/s13688-016-0093-1) | 4 noviembre 2016 | Arcos emocionales en libros; diferencia explícita entre arco y trama. No valida transferencia a letras cortas. |
| [BERTopic](https://arxiv.org/abs/2203.05794) | 11 marzo 2022 | Pipeline de embeddings, agrupación y representación c-TF-IDF. |
| [Julia Silge, Singing the Bayesian Beginner Blues](https://juliasilge.com/blog/bayesian-blues/) y [código fuente](https://github.com/juliasilge/old_bloggy_blog/blob/master/_R/2016-09-28-Bayesian-Blues.Rmd) | 28 septiembre 2016 | Motiva intervalos y partial pooling; usa menciones por población estatal y prior beta por momentos. Aquí se cambia a canción como Bernoulli y likelihood beta-binomial, no se replica una tasa de población. |
| [EmotionArcs](https://aclanthology.org/2024.latechclfl-1.7/) | Marzo 2024 | Perfiles emocionales de novelas con evaluación humana; no canciones. |
| [ALIGN-SIM](https://aclanthology.org/2024.findings-emnlp.436/) | Noviembre 2024 | Motiva controles semánticos finos; veinte tripletas originales no replican su benchmark. |
| [Reliability of Topic Modeling](https://aclanthology.org/2025.naacl-long.134/) | Abril 2025 | Medición de fiabilidad; tres semillas son un diagnóstico limitado, no aplicación completa de sus métricas. |
| [Sentence Smith](https://aclanthology.org/2025.emnlp-main.1343/) | Noviembre 2025 | Transformaciones controladas y negativos difíciles; no se implementó su pipeline AMR. |
| [Social Story Frames](https://aclanthology.org/2026.acl-long.1934/) | Julio 2026 | Inferencia contextual de intención/recepción con validación humana en historias sociales. No medida de arcos líricos. |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | 17 octubre 2023 | Diseño con tokens de reflexión entrenados; el grafo de reglas no es una réplica. |
| [CRAG](https://arxiv.org/abs/2401.15884) | 29 enero 2024; revisión 7 octubre 2024 | Evaluación/corrección de retrieval; no se implementó su evaluador aprendido ni fallback web. |
| [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | Marzo 2024 | Ruteo por complejidad con clasificador; no se reprodujo entrenamiento. |
| [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) | 19 septiembre 2024 | Contextualiza antes de indexar, distinto de ampliar vecinos después de recuperar. |
| [LangGraph: workflows y agentes](https://docs.langchain.com/oss/python/langgraph/workflows-agents) | Documentación mutable, consultada 11/10/2026 | Fundamenta la distinción entre rutas programadas y decisiones autónomas. |
| [OpenAI: gpt-5.4-mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini) y [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) | Documentación mutable, consultada 11/10/2026 | Nombre/modelo, tarifas y JSON schema. El recibo de ejecución respalda llamadas concretas; documentación no valida criterio literario del juez. |
| [Anthropic: Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview) | Documentación mutable, consultada 11/10/2026 | Identidad del modelo y tarifas. Pruebas originales de relaciones, sin corpus lírico enviado. |

## Afirmaciones que este estudio no hace

- Que el miedo aumentó, se interiorizó o cambió de manera causal en la música popular entre 1980 y 2026.
- Que los 19 clusters sean 19 emociones, o que las cinco capas de lectura fueran descubiertas y validadas por BERTopic.
- Que NMI con idioma o estabilidad entre semillas validen pertinencia de topics para narratología.
- Que cosine sea intensidad de miedo, diagnóstico clínico, intención del artista o emoción del oyente.
- Que pendientes de similitud confirmen seis formas universales de arcos musicales, o que un resultado no significativo pruebe ausencia de arcos.
- Que las fichas y referencias del asistente sean anotaciones humanas independientes o gold literario.
- Que las preguntas bilingües sean réplicas independientes, o que el split por familia demuestre generalización a nuevos artistas.
- Que un ID de cita válido pruebe apoyo de una afirmación, o que aceptar un paquete candidato sea una respuesta interpretativa correcta.
- Que los números de retrieval demuestren superioridad universal de MiniLM, E5, híbridos o agentes.
- Que una llamada real de embeddings a OpenAI y una revisión de suficiencia de contexto permitan comparar proveedores en igualdad de condiciones. Gemini no produjo vectores; Anthropic recibió sólo originales sintéticos.
- Que el estudio distribuya las letras o una licencia para redistribuirlas.

## Correcciones editoriales efectuadas

Se sustituyó la atribución causal de mejora al “contexto” por una descripción del
workflow completo. Se hizo explícito que preguntas, referencias y fichas provienen
del asistente, sin validación humana. Se separó ranking antes del gate del contexto
final, se explicaron el tope de candidatos y RRF/reintentos, y se incorporaron los
intervalos condicionales por familia. Las cifras principales del artículo coinciden
con las salidas finales consultadas; no se alteraron figuras ni resultados.

La revisión del builder detectó además que su opción de recálculo debía apuntar
al caché efectivo, reexportar los inputs del corpus y ejecutar ambos modelos RAG.
Se informó al autor del builder para su corrección coordinada. La ejecución final
del notebook y su vista previa se verifican por separado en el flujo editorial.

## Extensión comprobada: cuatro encoders y corpus ampliado

Esta revisión posterior conserva las corridas del piloto. El contraste de encoders
usa las mismas 40 canciones; el contraste de tamaño usa dos encoders locales con
40 y 120 casos. Ninguna llamada de esta extensión del panel envió las 120 letras.
Los modelos alojados y sus consultas se abren desde vectores privados cacheados;
el benchmark RAG posterior también se ejecutó offline.

| Afirmación actual | Archivo y medida | Resultado comprobado | Límite |
|---|---|---|---|
| Ampliación: 120 canciones, 108 identidades de artista, 1.137 bloques únicos | `expanded/results/corpus_diagnostics.json` | 1.333 bloques totales, 196 repeticiones; diez celdas de 12 canciones | Dos cohortes deliberadas, no muestreo probabilístico. Año final parcial; dos fechas exactas de 2026 están marcadas como no verificadas. |
| Comparación emparejada de cuatro encoders | `results/<modelo>/projections.json`, `receipt.input_documents`, `input_songs`, `corpus_sha256`, `document_order_sha256` | 386 bloques, 40 canciones e identidad/orden comunes; 384 dimensiones locales y 1.024 alojadas | Igual dimensión entre los dos OpenAI, sin equivalencia automática entre geometrías ni taxonomías. |
| OpenAI large: 11 tópicos, 14 bloques únicos sin asignar; small: cinco y siete | `results/openai--text-embedding-3-{large,small}/section_assignments.csv`, filas `document_id==canonical_id`, semilla17 | 11/14 y 5/7 | Los IDs pertenecen a cada ajuste; ruido o cantidad de clusters no miden emociones ni calidad. |
| Large es más estable entre estas semillas que small | `seed_stability.csv`: `ari_including_noise` | Large 0,942085–0,946182; small 0,104916–0,791207 | Tres semillas, mismo corpus; estabilidad no demuestra pertinencia de las categorías. |
| NMI tópico–idioma: 0,113/0,863/0,508/0,676 | `results/topic_language_association.csv` | Valores redondeados para MiniLM, E5, large y small | Medida posterior a descubrir grupos. No relación causal ni benchmark multilingüe independiente. |
| E5 cambia de dos a 28 tópicos al ampliar el corpus | `expanded/results/intfloat--multilingual-e5-small/section_assignments.csv`, únicos | 28 tópicos, 123 bloques únicos sin asignar; MiniLM 51/350 sobre 1.137 únicos | Parámetros fijos, pero corpus y densidades distintas. Los topics nuevos no son los antiguos con más observaciones. |
| PCA conserva 15,16/19,15/17,19/15,86% en tres componentes | `results/<modelo>/projections.json`: `receipt.explained_variance_total` | 0,151645 / 0,191483 / 0,171935 / 0,158558 | PCA se ajusta sobre bloques únicos con pesos iguales. Los centroides por canción/idioma son una agregación posterior; no reponderan el ajuste de los ejes. |
| Large y small alcanzan distintos recall densos e híbridos | `pilot-comparison/results/rag-openai-{large,small}/runs.json`, test positivo | Denso large 0,875→0,9375, small 0,916667→0,947917; híbrido contextual 0,864583 y 0,78125 | Ranking previo al gate, ocho familias positivas, anotaciones provisionales. El brazo compuesto cambia consulta, anchura y contexto; no es una ablation causal de una sola operación. |
| Ningún encoder del piloto tiene pendiente por canción con q<0,05 | `results/<modelo>/narrative_song_order_tests.csv`: `permutation_q_bh_within_variant` | Cero filas con q<0,05 en los cuatro encoders | Prueba de un contraste específico de similitud y orden, no prueba de ausencia de trama ni validación de seis formas narrativas. |
| Ninguna dirección de presencia del piloto resiste los cuatro priors | `results/bayes/summary.json`: `model_sensitivity[].endpoint_directions_consistent_across_declared_priors` | Listas vacías en cuatro modelos | Contrastes 1980–2020, condicionados a topics y muestra; no equivalencia histórica entre décadas. |
| MiniLM T0/EN: 1/4→3/4, diferencia jerárquica 19,6 puntos | `results/bayes/temporal_endpoint_contrasts.csv`, método `hierarchical_grid`, `pooled_languages` | Diferencia0,195555; intervalo[−0,193685;0,667274]; P(aumento)0,8159 | Hiperpriors propios, incertidumbre de hiperparámetros integrada; interpretación cultural del topic no entra en la posterior. |
| MiniLM T15 ampliado: 4/12 recientes, 13,0/20,7/34,6% posteriores | `expanded/results/bayes/posterior_rates.csv`: es,2020,T15, EB/jerárquico pooled y Jeffreys | 0,129583 / 0,206559 / 0,346154; observado0,333333 | Un tópico reestimado, no continuación de T15 del piloto. Sus cuatro canciones no convierten el grupo en categoría de miedo. |
| Algunas direcciones ampliadas persisten bajo los priors | `expanded/results/bayes/summary.json`, direcciones declaradas | MiniLM T7/EN aumenta; T13/ES y T18/ES disminuyen; E5 T7/EN disminuye | Diagnóstico exploratorio entre muchos contrastes, sin corrección de comparaciones múltiples ni garantía de coherencia temática. |

## Extensión comprobada: lectores, votos y tiempo argumental

La ontología y los prompts se congelaron antes de las llamadas. Los lectores reciben
el texto anónimo en orden, sin título, artista, fecha ni hipótesis de selección. Eso
suprime esas variables explícitas del prompt; no elimina la posible familiaridad
del modelo con canciones conocidas. El tercer lector es otra llamada a mini con
instrucción crítica, ciega a las propuestas previas: sus votos están correlacionados
con el primer mini y no constituyen una tercera arquitectura independiente.

El análisis primario exige campos, enumeraciones, referencias y pasos cronológicos
válidos simultáneamente. La extensión posterior separa validez categórica por campo
de validez del arco. Nunca rehabilita anclas cruzadas ni sustituye los ficheros
primarios. Mayoría de subtemas significa dos votos por etiqueta; no mayoría de un
conjunto completo idéntico. La sensibilidad conjunta requiere además que esos
mismos votos vinculen la etiqueta a miedo explícito o implícito.

| Afirmación actual | Archivo y medida | Resultado comprobado | Límite |
|---|---|---|---|
| Dos lectores y hasta20 terceros ciegos sobre40 canciones | `pilot-comparison/results/arc-panel/protocol.json`, `panel_votes.json` | 80 lecturas primarias +20 críticas; 100 llamadas reales | Los40 casos tenían desacuerdo de núcleo o esquema; el tope20 se aplicó por hash predefinido. El tercero no se seleccionó después por éxito de campos categóricos. |
| 64 propuestas pasan el esquema íntegro | `arc-panel/summary.json`; `fieldwise-posthoc/summary.json`: `invalid_primary_reasons` | 64 válidas,36 inválidas, todas por orden de pasos | No son36 canciones sin narrativa. Son36 lecturas cuyo esquema de anclas incumple el validador; no se ocultaron ni repararon con más llamadas. |
| Acuerdo inicial por campo | `fieldwise-posthoc/field_agreement.csv`: `initial_agree_songs`, denominador40 | Voz30; postura19; relevancia20; polaridad19; conjunto exacto de subtemas10; arco7 | Todos los80 votos primarios tienen enums categóricos válidos; solamente48 tienen arco íntegro válido. Comparar acuerdos sin esta distinción confunde tareas. |
| Mayoría posterior por campo | Mismo CSV: `majority_resolved_songs`, denominador40 | Voz37, postura28, relevancia33, polaridad27, subtemas35, arco14 | Uso posterior de los20 votos críticos ya ejecutados. Tipo/enum válido no verifica por sí solo interpretación del contenido. No desaparecen los no resueltos. |
| 23 canciones con referencia de miedo, diez ausentes y siete sin resolución | `fieldwise-posthoc/song_annotations.json`: `fear_reference_consensus` | 12 explicit +11 implied;10 absent;7 no resueltos | Nombrar, negar o hipotetizar miedo es referencia; no intensidad, experiencia afirmada ni emoción del artista. Los33 resueltos incluyen las10 ausencias. |
| 14 arcos apoyados,12 repetitivos | `fieldwise-posthoc/anchored_cases.json`, votos íntegros del mismo tipo | 12 repetition,1turn,1mixed;26/40 sin mayoría estricta de arco | Lista de TODOS los14 casos con≥2 lecturas y anclas válidas, no muestra representativa del siglo. La categoría compartida puede coexistir con diferentes pasos/lecturas. |
| Cinco casos de la tabla tienen apoyo cronológico | `anchored_cases.json`: Rockwell1983, Rayando1990, A Dios2002, Woods2014, Aperture2026 | 3/2/2/2/2 votos válidos; cuatro repetition y un turn | El año de Rockwell es el de primera edición registrado, no la fecha del álbum o de chart1984. Aperture conserva postura y polaridad sin consenso. |
| Incertidumbre/cambio aparece2,3,4,3,3 veces | `fieldwise-posthoc/concern_presence_by_period.csv`, balanced, `fear_paired_joint_votes`, `uncertainty_change` | Counts2/3/4/3/3, denominador8 en cada período | Misma secuencia bajo `fear_consensus_cross`; no evolución monotónica ni prevalencia externa. Dos idiomas tienen cuatro casos cada uno. |
| Fragilidad/identidad tiene3/8 en cada período desde2000 con miedo | Mismo CSV, `fragility_identity`, vistas condicionadas | 3/3/3; todas las preocupaciones recientes5/8 | Los cinco motivos no son cinco referencias al miedo. El denominador conserva todos los controles y no resueltos; etiquetas múltiples pueden superar100%. |
| Dos lectores revisan los mismos24 contextos RAG | `pilot-comparison/results/llm-comparison-context/{protocol,runs,summary}.json` | 48 filas válidas;24 mini cacheadas,24 solicitudes nano nuevas | Misma pregunta original y bloques; segunda etapa real de LangGraph con rutas programadas, sin planificación autónoma ni nuevo retrieval. |
| Ambos abstienen8/8 negativas y8/16 positivas | `llm-comparison-context/summary.json`: `models` | Ambos1/8 abstenciones con todos los soportes designados y7/8 con soporte incompleto; citas ajenas0 | Referencias iniciales provisionales e incompletas. Ausencia de un ID designado no equivale automáticamente a insuficiencia semántica. |
| Los agregados iguales ocultan desacuerdos | `llm-comparison-context/summary.json`: `pairwise` | Decisión20/24; abstención22/24 | Difieren en aceptación de `shower_intruder_es` y `lunar_bargain_es`. Acuerdo no equivale a corrección; parejas bilingües no son réplicas independientes. |
| Cuatro modelos revisan20 tripletas sintéticas comunes | `results/llm-comparison-synthetic/{protocol,runs,summary}.json` | 80 filas válidas; mini/nano/Haiku9/9 restricciones, Sonnet7/9 | Las nueve restricciones se especificaron para diagnóstico tras el Sonnet piloto. No exactitud general de los40 pares ni test nuevo independiente. |
| Sonnet confunde coexistencia con contradicción en dos casos | Misma fase: `coexistence_false_contradictions` y runs de traducción | Dos etiquetas AC de contradiction ante miedo y actividad agradable compatibles | Patrón observado en dos originales, no superioridad universal de proveedor. Anthropic recibió únicamente originales sintéticos. |
| Doce controles nuevos verifican alcance solicitado | `results/llm-comparison-detail-controls-v2/{protocol,runs,summary}.json` | mini/nano/Haiku12/12;Sonnet11/12; todos4/4 abstienen cuando falta fuente | Apéndice diagnóstico: IDs de evidencia revelan la condición y ofrecen un atajo. Cuatro familias de tres variantes EN; no headline ni validación independiente. V1 tiene un fallo de formato del adaptador preservado, no fracaso semántico de Claude. |
| Coste LLM acumulado después de contextos | `llm-comparison-context/summary.json`: `budget_after` | USD0,38318290;355 requests; cero resultados de facturación desconocida | Incluye panel, formatos fallidos/repetidos y fases previas;24 mini reutilizadas no se cobran otra vez. Embeddings se contabilizan aparte. No es un recibo bancario. |

### Decisión editorial del agente metodológico

**ACCEPT** para la sección de arcos y la conclusión temporal de la versión leída.
Los números y cinco casos corresponden a los outputs; el cierre defiende
recurrencias y variación de respuestas en la muestra, sin fabricar una tendencia
monotónica. Esta decisión es una revisión automatizada de correspondencia
editorial; los votos de aprobación entre agentes no son gold de interpretación
literaria ni nuevas observaciones independientes de canciones.

Las medidas categóricas por campo y el filtro de relevancia son extensiones
posteriores declaradas. No se presenta el análisis40 del panel como análisis120.
No se atribuyen propiedades de miedo a todos los clusters ni se promueve nano
para producción: la documentación consultada lo marca deprecated, aunque sus
llamadas experimentales concretas funcionaron.

## Revisión anual uniforme del 12 de octubre de 2026

Esta revisión sustituye la capa interpretativa principal del artículo, conserva el panel40 como fase anterior y elimina del post las referencias al cuaderno. El cierre termina exclusivamente con conclusiones.

| Afirmación | Evidencia nueva | Consecuencia editorial |
|---|---|---|
|120 canciones,240lecturas;53referencias,37ausencias,30inciertas|`results/annual-120/summary.json`; validación de esquema/IDs/cronología0defectos|No atribuir motivos a todas las canciones recuperadas. Referencia incluye negación y amenaza implícita; no es experiencia interior automáticamente.|
|Motivos anuales,47años; seis sin observaciones|`annual_presence.csv`, `visualization-receipt.json`|Recuento/tasa con n por año; ausencia de observación distinta de cero casos en un año observado. No suavizar huecos.|
|Fragilidad15casos,11EN; coerción12,6EN6ES|`annotations.json`, tablas de embeddings|El idioma y la selección condicionan el recorrido; no fechar nacimiento histórico de valores.|
|Arcos53:14repetición,8confrontación,4contraste,4giro,3mixto,20inciertos|`summary.json`, `votes.json`|Conservar dos segmentaciones cuando la categoría coincide; no convertir coincidencia categórica en identidad de pasos.|
|Aperture queda incierta;26/40presencias coinciden con el panel anterior|`previous_panel_sensitivity.csv`|Retirar el antiguo ejemplo de giro de la conclusión; lectura y protocolo alteran resultados.|
|Large91/116vecinos con miedo frente a70,4% uniforme; small84/117frente a70,6%|`embeddings/neighbor_checks.csv`|Control por idioma,26consultas comunes; máscaras posteriores dan denominadores distintos y destinos repetidos son dependientes. No ranking general.|
|120locals: vecinos con miedo +0,9/+0,6pp sobre baseline; motivos +14,5/+2,9pp|Misma tabla y asociaciones cluster×motivo|Separar concordancia binaria, consultas positivas e intersección de motivos. Un cluster amplio no equivale a tema universal.|
|28intervalos estrictos y28sensibles entre extremos incluyen0|`bayes/annual-bayes-endpoint-posteriors.csv`|Ninguna dirección resuelta por idioma/motivo bajo estos dos priors. No afirmar equivalencia histórica ni confundir sensibilidad descriptiva con incertidumbre posterior.|
|Varianza temporal versus posterior exacta|`bayes/annual-bayes-variance.csv`|Cuantidades distintas de la varianza PCA; conteos se mantienen mientras el prior cambia la posterior.|

La nueva ontología está fijada por el protocolo interpretativo y no se presenta como etiquetas descubiertas por BERTopic. Los tres lectores son procesos del mismo asistente configurado; no tres proveedores ni gold externo. La comparación es una lectura de casos y sus relaciones, no una estimación causal o poblacional. Ninguna llamada de las APIs del proyecto ni envío adicional se realizó en esta revisión.
