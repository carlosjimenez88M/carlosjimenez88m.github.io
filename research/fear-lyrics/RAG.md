# Recuperación y grafo contextual: qué se ejecutó

`rag.py` compara BM25, recuperación densa multilingüe y combinación híbrida
mediante Reciprocal Rank Fusion. Cada método se ejecuta como una pasada y como
un grafo LangGraph con rutas condicionales, hasta dos reformulaciones léxicas
bilingües y expansión a bloques vecinos de la misma canción.

El controlador usa reglas transparentes de coincidencia léxica; no es un agente
autónomo basado en LLM ni una reproducción de Self-RAG, CRAG o GraphRAG. Su salida
es una plantilla que indica cuántos bloques candidatos se recuperaron y ofrece
sus identificadores. **No se evaluó generación de interpretaciones literarias.**

## Ejecutar sin APIs pagadas

```sh
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py --self-test
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py
```

El modelo debe estar disponible en el caché privado antes de ejecutar. Para
ejecutar solamente el baseline léxico, usar `--bm25-only`. Para un segundo modelo,
usar `--model intfloat/multilingual-e5-small --output research/fear-lyrics/results/rag-e5`.
El módulo no descarga modelos, obtiene letras ni llama proveedores de inferencia.

La descomposición posterior conserva las corridas principales y utiliza cuatro
variantes de ancho fijo: reformulación y vecinos, activados o desactivados. Para
ejecutarla, es obligatorio un directorio diferente:

```sh
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py --ablation --output research/fear-lyrics/results/rag-ablation
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py --model intfloat/multilingual-e5-small --ablation --output research/fear-lyrics/results/rag-e5-ablation
```

Las cuatro variantes usan diez candidatos por búsqueda y el mismo máximo total
de treinta; las versiones sin reformulación gastan una búsqueda. El contexto
conserva su máximo de 320 palabras y cinco bloques. Esta descomposición fue
diseñada después de observar los resultados originales: no ofrece confirmación
independiente ni transforma el test original en un nuevo conjunto de validación.

## Comparación y límites

La indexación utiliza exclusivamente bloques únicos de letras: no añade títulos,
artistas, años, hipótesis de selección, etiquetas de topics ni respuestas gold.
Las reformulaciones utilizan un glosario genérico, independiente de los golds.
Las versiones EN/ES de una pregunta pertenecen a la misma familia y partición.
El split es por familias de preguntas, **no por canciones o artistas inéditos**.

Todas las variantes tienen un máximo de 320 palabras de contexto y cinco bloques.
El máximo de candidatos consultados es 30, contados incluso cuando se repiten
identificadores. La pasada única consulta treinta candidatos de una vez; el grafo
puede repartir ese mismo máximo en hasta tres búsquedas de diez. La expansión de
vecinos consume el mismo presupuesto de contexto. La cantidad realmente usada,
las llamadas y los reintentos se guardan por ejecución; un máximo común no implica
que cada variante incurra en el mismo coste computacional.

Los soportes de referencia son identificadores elegidos por el asistente tras leer los
bloques privados. No son anotaciones exhaustivas de toda la relevancia del corpus
ni tienen adjudicación independiente. Recall, MRR y nDCG describen este benchmark
diagnóstico de 32 preguntas originales. Los negativos incluyen ausencias sencillas
de otros dominios y cuatro consultas adversarias: mezclan motivos presentes con
identidades verificadas o datos clínicos ausentes. Su abstención no demuestra
fiabilidad general ante preguntas musicales ambiguas. El solapamiento léxico del
controlador tampoco prueba implicación.

El JSON de anotaciones se guarda en el caché privado; el código permite reconstruir
la asociación entre preguntas e identificadores gold para reproducir la evaluación.
Los resultados públicos incluyen preguntas, identificadores, métricas, trazas y
configuración; nunca letras, respuestas interpretativas gold ni embeddings.
La validación de citas comprueba que un ID esté en el contexto, **no** que apoye
una afirmación semántica. Las latencias corresponden al grafo offline con todas
las variantes permitidas de preguntas ya vectorizadas; la preparación del índice
y de preguntas se informa por separado. No comparan latencias de APIs.

## Fuentes primarias

- [LangGraph: infraestructura de orquestación con estado](https://docs.langchain.com/oss/python/langgraph/overview).
- [LangGraph: recuperación, evaluación y reformulación condicional](https://docs.langchain.com/oss/python/langgraph/agentic-rag).
- [Anthropic: Contextual Retrieval, 19 septiembre 2024](https://www.anthropic.com/engineering/contextual-retrieval). Nuestro vecino textual es una variante local explícita, no la contextualización mediante LLM de ese experimento.
- [FRANQ, ACL 2026: distinción entre factualidad y fidelidad a evidencia](https://aclanthology.org/2026.findings-acl.338/).

## Revisión posterior con proveedores reales

`llm_review.py` separa los resultados offline originales de una revisión posterior
de suficiencia por OpenAI. Recibe las 24 consultas test originales y los contextos
ya congelados de MiniLM híbrido con reformulación y sin vecinos. No utiliza los
soportes anotados para decidir ni completar el contexto; las métricas anotadas se
añaden al registro público únicamente después de ejecutar. Anthropic clasifica
20 tripletas originales sintéticas; no recibe letras ni material derivado de ellas.
Gemini no se utiliza para inferencia después de que el usuario limitó la continuación
a OpenAI y Anthropic.

```sh
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/llm_review.py --self-test
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/llm_review.py
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/llm_review.py --run --providers openai,anthropic
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/llm_review.py --run --label-topics --providers openai
```

Sin `--run`, sólo informa el plan. El ledger privado reserva el coste antes de cada
solicitud, conserva la reserva ante facturación incierta y no reintenta automáticamente.
La fase tiene techo de USD 0.80 y las etiquetas de topics USD 0.04 dentro de ese
techo. Se calcula gasto a partir de tokens informados y tarifas pagadas estándar;
no se presuponen créditos ni descuentos. `temperature=0` no garantiza idéntica
salida en futuras ejecuciones; los recibos y las respuestas originales quedan
en el caché privado para reproducir esta ejecución sin llamadas nuevas.

Los resultados públicos en `results/llm-review` incluyen decisiones, identificadores,
conteos y uso, pero excluyen afirmaciones y texto de incertidumbre generados. La
estratificación entre contextos con todos los soportes designados e incompletos
evita confundir un contexto insuficiente con un error del revisor. Los soportes
siguen siendo propuestas del asistente, no adjudicación humana independiente.

`results/topic_labels.json` es otra capa: propuestas españolas de 2–6 palabras y
paráfrasis originales cortas para clusters ya ajustados. Se leen hasta tres bloques
cercanos al centroide de canciones distintas, con máximo cien palabras por bloque.
La publicación valida IDs y descarta coincidencias de cinco palabras consecutivas
con letras. No mide exhaustividad del cluster ni descubre subtopics nuevos. Las
palabras c-TF-IDF originales siguen disponibles por separado.
El prompt original pedía 3–6 palabras. Una etiqueta de dos palabras pasó los
controles de contenido pero no el formato; hubo un único intento de reparación.
La revisión editorial permitió 2–6 sin nuevas llamadas y conserva ambas solicitudes
en el coste y los metadatos. Esta modificación de formato no valida la interpretación.

- [OpenAI: modelo gpt-5.4-mini y tarifas](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
- [OpenAI: Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
- [Anthropic: Claude Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview).

## Embeddings alojados y comparación de lectores

`RemoteCachedEncoder` verifica IDs, hashes, orden y dimensiones de corpus y consultas;
normaliza los arrays y sólo lee el caché. Una consulta no guardada produce un error,
sin fallback de red. Los vectores de las 32 preguntas y sus dos reformulaciones
se preparan por separado: son 96 textos únicos en esta versión del benchmark.

```sh
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py --export-queries .research-cache/fear-lyrics/rag-query-input.json
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/rag.py --model openai/text-embedding-3-small --remote-npz .research-cache/fear-lyrics/openai-small.npz --remote-metadata .research-cache/fear-lyrics/openai-small.json --output research/fear-lyrics/results/rag-openai-small
```

El NPZ contiene `documents` y `queries`; puede incluir además probes/prototypes.
El sidecar contiene `model_id`, `document_ids`, `document_hashes` y `query_hashes`,
en el mismo orden que los arrays. Los hashes de texto son SHA256 de UTF-8. El coste
de adquirir esos vectores pertenece a la fase API; ejecutar RAG sobre ellos no
realiza solicitudes y sus latencias siguen siendo locales.

`llm_comparison.py` compara GPT-5.4 mini/nano y Claude Sonnet 4.6/Haiku 4.5 sobre
las mismas veinte tripletas originales. Su resumen separa acuerdo de corrección.
Las nueve restricciones lógicas seleccionadas estaban informadas por el piloto:
no son un benchmark independiente ni una exactitud global de los cuarenta pares.
Nano se incluye como modelo experimental accesible con estado deprecado; Haiku
4.5 figura como legacy. La disponibilidad observada no es una recomendación de
modelos para producción.

`detail_control_benchmark.py --freeze` guarda doce escenarios nuevos y expectativas
antes de llamar APIs: nombre, fecha, código diagnóstico inventado y dosis de un
compuesto ficticio, cada uno con tres alcances de procedencia. La versión v1 omitió
los nombres de campos JSON en el prompt de Anthropic y no es comparable para
calidad semántica. V2 enumera los mismos campos para todos los modelos y conserva
casos, expectativas, fallos y costes anteriores. Las expectativas son del asistente;
la autoridad documental existe sólo dentro del escenario ficticio. Doce casos en
inglés no establecen seguridad clínica, exactitud factual real ni generalización
bilingüe.

`context_review.py` ejecuta un LangGraph real de lectores OpenAI mini/nano sobre
contextos congelados idénticos: adjuntar contexto, revisar apoyo por API, validar
IDs y emitir o abstenerse. No ofrece planificación de herramientas ni permite
modificar retrieval por modelo. Comprueba el hash del corpus antes de enviar;
las anotaciones de soporte se usan sólo para evaluar después. Al ampliar el corpus,
los soportes originales dejan abierta la posibilidad de nueva evidencia válida:
sus IDs no constituyen relevancia exhaustiva.

```sh
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/llm_comparison.py --self-test
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/detail_control_benchmark.py --freeze
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/detail_control_benchmark.py --run
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/context_review.py --self-test
.research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/context_review.py --retrieval-dir research/fear-lyrics/results/rag-ablation
```

Las llamadas posteriores siguen usando el ledger acumulado de USD 0.80, dentro
del techo total autorizado de USD 1. Los modelos alojados se registran por
requested/returned ID; caché y recibos fijan la ejecución observada aunque el
alias del modelo pueda cambiar. El piloto de cuarenta canciones permanece en
`pilot-40/`; los nuevos resultados no lo reemplazan.
