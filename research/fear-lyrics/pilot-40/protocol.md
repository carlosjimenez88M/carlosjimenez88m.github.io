# Miedo, canciones y representaciones: protocolo previo al análisis

Fecha del protocolo: 11 de octubre de 2026. Corte histórico: 11 de octubre de 2026.

## Corrección de procedencia antes de ejecutar resultados: versión 2

La revisión de fuentes detectó marcas de tiempo LRC dentro del campo `plainLyrics` de Aperture. Esas marcas inflaban el desacuerdo entre proveedores y podían introducir números en la segmentación y los embeddings. Se eliminan únicamente etiquetas numéricas de tiempo al principio de una línea; se mantienen los encabezados de estrofa, las palabras entre corchetes y los pasajes hablados originales, incluida la narración de Thriller.

La entrada de LRCLIB elegida para Zombie estaba asociada a una recopilación MTV Unplugged y una duración de 82 segundos. Se conserva esa respuesta privada para la comparación de proveedores, pero el corpus analítico utiliza la página de Genius de la grabación original mediante una preferencia explícita en el manifiesto. Las ediciones `unplugged`, `acoustic` y `demo` se excluyen de futuras selecciones automáticas. No se excluyen indiscriminadamente los radio edits: pueden formar parte de una edición original y requieren revisión específica.

Los 120 registros de proveedores se reutilizan sin consultas de red y permanecen sin modificaciones. Se reconstruyen el corpus, los hashes y las tablas de cobertura/acuerdo con la limpieza corregida. El registro público de auditoría guarda los hashes anteriores y posteriores y los motivos, sin letras. Esta revisión ocurre después de adquirir los datos y antes de interpretar los resultados; no se presenta como una decisión predefinida.

La preferencia automática de LRCLIB por la letra más larga entre coincidencias de artista/título es una heurística cuando falta el álbum. No demuestra que una edición sea correcta. Las recopilaciones, duraciones inconsistentes y canciones disponibles en un solo proveedor permanecen como límites explícitos. El corpus contiene casos conocidos con evidencia comercial heterogénea; no todos cumplen un único umbral de éxito.

## Pregunta

¿Qué relaciones textuales detectan distintas representaciones al buscar miedo en canciones, qué subtemas aparecen en una muestra bilingüe de éxitos y qué evidencia recupera un RAG cuando una interpretación depende del contexto o del orden de los fragmentos?

No asumimos que una distancia vectorial mida miedo, que un tópico sea una emoción, que una curva de similitud sea un arco narrativo validado o que una selección de éxitos represente toda la música de una época.

## Diseño del corpus

- Bloques de primera edición: 1980–1989, 1990–1999, 2000–2009, 2010–2019 y 2020–11/10/2026. La fecha de entrada en listas se registra por separado.
- Inglés y español, equilibrados dentro de cada bloque. Artistas y canciones se seleccionan antes de observar clusters.
- Selección intencional de candidatos asociados a miedo, amenazas, incertidumbre o negación del miedo, junto con controles y casos ambiguos. Las etiquetas de selección son hipótesis de lectura del analista; no constituyen anotaciones humanas independientes.
- El éxito comercial exige una fuente verificable de listas, certificación o lanzamiento. Mezclar mercados exige describirlos; las posiciones de listas diferentes no son una escala común de popularidad.
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
