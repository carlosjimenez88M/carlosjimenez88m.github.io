# Miedo, canciones y representaciones: protocolo previo al análisis

Fecha del protocolo: 11 de octubre de 2026. Corte histórico: 11 de octubre de 2026.

## Corrección de procedencia antes de ejecutar resultados: versión 2

La revisión de fuentes detectó marcas de tiempo LRC dentro del campo `plainLyrics` de Aperture. Esas marcas inflaban el desacuerdo entre proveedores y podían introducir números en la segmentación y los embeddings. Se eliminan únicamente etiquetas numéricas de tiempo al principio de una línea; se mantienen los encabezados de estrofa, las palabras entre corchetes y los pasajes hablados originales, incluida la narración de Thriller.

La entrada de LRCLIB elegida para Zombie estaba asociada a una recopilación MTV Unplugged y una duración de 82 segundos. Se conserva esa respuesta privada para la comparación de proveedores, pero el corpus analítico utiliza la página de Genius de la grabación original mediante una preferencia explícita en el manifiesto. Las ediciones `unplugged`, `acoustic` y `demo` se excluyen de futuras selecciones automáticas. No se excluyen indiscriminadamente los radio edits: pueden formar parte de una edición original y requieren revisión específica.

En esta fase histórica del piloto, los 120 registros de proveedores corresponden a 40 canciones consultadas en tres servicios; no son 120 canciones. Se reutilizan sin consultas de red y permanecen sin modificaciones. Se reconstruyen el corpus, los hashes y las tablas de cobertura/acuerdo con la limpieza corregida. El registro público de auditoría guarda los hashes anteriores y posteriores y los motivos, sin letras. Esta revisión ocurre después de adquirir los datos y antes de interpretar los resultados; no se presenta como una decisión predefinida. La copia de esta fase permanece congelada en `pilot-40/`.

La adquisición ampliada posterior reúne 120 canciones y conserva 281 respuestas privadas, incluidas consultas repetidas y respuestas históricas. La tabla pública vigente tiene 240 filas: una consulta seleccionada por canción y cada uno de los dos proveedores consultados en la ampliación. Archivos de respuesta, filas de cobertura y canciones son denominadores distintos; las respuestas antiguas de lyrics.ovh se conservan pero no constituyen nuevas consultas de las 120 canciones.

La preferencia automática de LRCLIB por la letra más larga entre coincidencias de artista/título es una heurística cuando falta el álbum. No demuestra que una edición sea correcta. Las recopilaciones, duraciones inconsistentes y canciones disponibles en un solo proveedor permanecen como límites explícitos. El corpus contiene casos conocidos con evidencia comercial heterogénea; no todos cumplen un único umbral de éxito.

## Pregunta

¿Qué relaciones textuales detectan distintas representaciones al buscar miedo en canciones, qué subtemas aparecen en una muestra bilingüe de éxitos y qué evidencia recupera un RAG cuando una interpretación depende del contexto o del orden de los fragmentos?

No asumimos que una distancia vectorial mida miedo, que un tópico sea una emoción, que una curva de similitud sea un arco narrativo validado o que una selección de éxitos represente toda la música de una época.

## Diseño del corpus

- Bloques de primera edición: 1980–1989, 1990–1999, 2000–2009, 2010–2019 y 2020–11/10/2026. La fecha de entrada en listas se registra por separado.
- Inglés y español, equilibrados dentro de cada bloque. Artistas y canciones se seleccionan antes de observar clusters.
- Selección intencional de candidatos asociados a miedo, amenazas, incertidumbre o negación del miedo, junto con controles y casos ambiguos. Las etiquetas de selección son hipótesis de lectura del analista; no constituyen anotaciones humanas independientes.
- La evidencia de éxito comercial requiere listas o certificaciones verificables. Una fuente de lanzamiento acredita publicación, no éxito. El corpus ejecutado reúne casos de difusión heterogénea, con evidencia comercial débil o pendiente registrada en el manifiesto; no aplica un umbral común de éxito a las 120 canciones. Mezclar mercados exige describirlos; las posiciones de listas diferentes no son una escala común de popularidad.
- Un año parcial no tiene el mismo tiempo de exposición que una década completa. Los resultados temporales describirán esta muestra, con denominadores visibles.
- Los fallos de cobertura de letras se registran por proveedor, canción, idioma y período. No se reemplaza una letra ausente por una paráfrasis inventada.

## Unidades y procedencia

La canción es la unidad de selección y remuestreo. Los bloques de letra son unidades computacionales dependientes dentro de ella. Se conserva el orden y se identifica si los límites proceden de estrofas, marcadores o una partición heurística.

Las letras completas y las respuestas originales de proveedores quedan en la caché privada ignorada por Git. Los materiales públicos contienen IDs, metadatos, hashes, palabras de tópicos, estadísticas y referencias, sin reproducir letras completas. El notebook no mostrará filas de texto completo ni respuestas que reconstruyan una canción.

## Experimentos definidos antes de inspeccionar resultados

1. **Proveedores de letras.** Cobertura y acuerdo textual normalizado entre servicios. El acuerdo no demuestra independencia de las fuentes ni exactitud editorial. Una discrepancia activa revisión de versión, artista, idioma y limpieza.
2. **Representaciones.** Referencias locales multilingües y APIs disponibles, con nombres, dimensiones, tareas, fechas y costes registrados. Los vectores de modelos distintos no se comparan directamente entre sí: se comparan resultados de las mismas tareas.
3. **Pruebas semánticas sintéticas.** Pares bilingües originales que cambian negación, sujeto/objeto, repetición, amenaza y afrontamiento. Las expectativas se fijan antes del ranking; son tests diseñados, no una evaluación exhaustiva del significado musical.
4. **BERTopic compartido.** Un espacio global sobre los bloques, con UMAP/HDBSCAN y c-TF-IDF. Semillas y parámetros se registran; los outliers permanecen visibles. No se impone un número de tópicos para obtener las categorías deseadas.
5. **Subtemas.** Jerarquía por proximidad entre representaciones de tópicos y lectura de ejemplos mediante IDs. Un corte jerárquico es una decisión analítica, no una taxonomía natural demostrada.
6. **Temporalidad.** Composición de tópicos normalizada por canción e idioma, con remuestreo de canciones y sensibilidad a estribillos repetidos. Una diferencia entre bloques no identifica una causa histórica; artista, mercado, género, traducción y selección pueden explicarla.
7. **Orden dentro de la canción.** Trayectorias de similitud a prototipos bilingües sintéticos de amenaza, evitación y afrontamiento. Se comparan con permutaciones del orden y con el efecto del estribillo. Se llaman trayectorias de representación, no emociones medidas en oyentes.
8. **Recuperación.** Baselines léxico, denso e híbrido sobre el mismo corpus y presupuesto. Consultas sin nombres/títulos y controles difíciles; separación entre consultas usadas para configurar y consultas de evaluación. Se incluyen preguntas que el corpus no puede responder.
9. **RAG contextual con LangGraph.** Ruteo, recuperación, expansión de vecinos de la misma canción, revisión de evidencia, reintento acotado y abstención. Comparación con recuperación simple con el mismo presupuesto total de evidencia, no solo el contexto final. El grafo se ejecuta y sus trazas se guardan.
10. **Generación y verificación.** Si se autorizan APIs pagadas, la generación usa los mismos paquetes de evidencia para cada proveedor. La fidelidad de citas, el soporte de afirmaciones y la abstención se examinan aparte del juicio de otro modelo. Un juez LLM no sustituye validación humana.

## Lectura crítica y salida

Se informan cobertura, tamaños efectivos, incertidumbre, fallos y resultados negativos. Las curvas y mapas 2D son ayudas de inspección; no prueban por sí solos estructura narrativa ni validez semántica. Se documentará qué decisiones se modificaron después de ver los datos y por qué.

La notebook en español contendrá el diseño, el código y las ejecuciones reales. El artículo final se escribirá después de revisar resultados y supuestos. Se mantendrá como borrador hasta completar la comprobación editorial y la revisión de las afirmaciones.

## Fases ejecutadas y alcance final

El registro acumulado se encuentra en `phase-manifest.json`, con recibos y hashes por modelo. `results/analysis_protocol.json` y `results/run_status.json` conservan la última invocación del CLI, dedicada a OpenAI small; no enumeran retrospectivamente todas las fases.

| Fase | Canciones y bloques únicos | Representaciones y lectura | Salidas |
|---|---|---|---|
| Piloto congelado | 40 / 386 | Dos encoders locales y OpenAI large, incorporado después | `pilot-40/`; registro histórico inmutable |
| Comparación emparejada final | 40 / 386 | MiniLM, E5, OpenAI large y small sobre los mismos IDs; semillas 17, 42 y 73 | `results/`: BERTopic, UMAP, PCA global de 20 componentes con proyección 3D, Bayes y 192 trazas RAG por encoder |
| Ampliación local anidada | 120 / 1.137 | MiniLM y E5; las 40 canciones anteriores forman parte de esta muestra | `expanded/results/`: las mismas familias de diagnósticos; los tópicos se vuelven a ajustar y sus IDs no son continuaciones del piloto |
| Panel narrativo | 40 | Dos lectores OpenAI, 80 votos iniciales y 20 lecturas críticas; análisis por campo posterior sin nuevas llamadas | `pilot-comparison/results/arc-panel/`; desacuerdos y validación cronológica conservados |

La comparación de Anthropic y OpenAI utiliza cuatro lectores sobre controles sintéticos originales. La lectura de contextos de canciones compara dos lectores OpenAI sobre 24 paquetes idénticos. El artículo final preparado y el notebook ejecutado reúnen estas fases; enviar la ampliación completa a encoders o lectores alojados es una extensión pendiente, no un resultado de cuatro modelos sobre 120 canciones.
