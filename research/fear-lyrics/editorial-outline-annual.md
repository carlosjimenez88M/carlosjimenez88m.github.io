# Propuesta editorial: el miedo, año por año

Propuesta inicial de estructura; no modifica el artículo ni las lecturas. Revisión del 12 de octubre de 2026, preparada antes del cierre de las 240 lecturas. Este documento no agrega conclusiones numéricas. La síntesis ejecutada posterior está en `results/annual-120/`, con su protocolo y recibos separados.

La pregunta rectora debería aparecer en título, apertura, figuras y cierre: **entre 1980 y 2026, ¿qué amenazas y preocupaciones reaparecen en las canciones seleccionadas, cómo cambia la respuesta de sus voces y qué consideran valioso conservar o transformar?** El año calendario ordena la comparación. Los embeddings proponen relaciones entre textos; la lectura explica qué significan y cómo se desarrollan.

**Por qué se diluye hoy el argumento.** El artículo tiene 12 secciones principales y 18 figuras. Su apertura presenta tres veces la curiosidad por permanencia, cambio y valor; después, el itinerario sigue UMAP, comparación de encoders, PCA, Bayes, RAG y LangGraph. Las canciones llegan como ejemplos de comprobaciones técnicas. El lector encuentra más respuestas sobre los instrumentos que sobre los años. Además, el mapa ofrece las 120 fechas, pero sus interpretaciones proceden del piloto de 40: una cobertura visual amplia no equivale a una lectura anual uniforme. Las curvas por períodos tampoco responden plenamente a la promesa del título. El valor subjetivo reaparece en formulaciones generales, sin que cada comparación concrete qué se protege y mediante qué respuesta.

Leí completos los posts propios [Aquamosh](../../content/post/2026-05-20-aquamosh-quadrilingual-anatomy.md) y [Attention Windows](../../content/post/2026-02-10-attention-windows-beatles-floyd.md), además del [artículo actual](../../content/post/2026-10-11-miedo-embeddings-bertopic-rag.md). Ambos abren con una tensión musical comprensible, delimitan una pregunta y convierten la discrepancia entre medida e interpretación en parte del argumento. Conviene conservar esa voz inquisitiva y concreta. No conviene trasladar al nuevo texto toda la arquitectura de correcciones y límites: cada precisión debe aparecer junto a la afirmación que delimita. La voz propia puede resultar del recorrido caso → problema → instrumento → comprobación → consecuencia, sin imitar giros ni ritmos de otros ensayistas.

**Jerarquía visual antes de reescribir.**

| Prioridad | Figura y función narrativa | Qué debe conservar visible |
| --- | --- | --- |
| 1 | Mapa anual 1980–2026: año × subtema leído; cada marca corresponde a una canción | Identidad del caso, referencia y polaridad, arco resuelto o incierto, número de canciones y composición por idioma; años sin observaciones vacíos |
| 2 | Cobertura anual y resolución: distingue lo que se leyó, lo ausente y lo no resuelto | Denominadores por año y por idioma; ausencia de muestra distinta de ausencia de miedo |
| 3 | Comparación de dos o tres casos históricos con pasos internos y valor en juego | Fecha de cada caso separada del orden de sus bloques; pasos y enlaces válidos; paráfrasis, nunca versos |
| 4 | Un diagnóstico compacto de embeddings: relaciones entre casos y sensibilidad al idioma/modelo | Vecinos calculados en el espacio original; categorías leídas separadas de tópicos automáticos |
| Apoyo | PCA, UMAP completos, priors, varianzas y trazas del sistema | Acceso desde los atlas o materiales del repositorio; no interrumpir el recorrido anual con una figura por técnica |

La portada y el recurso principal de `readerGuide` deben anunciar la comparación anual. Una curva de varianza PCA sirve al diagnóstico del instrumento, pero no comunica el problema central. Reservaría cuatro o cinco figuras para el cuerpo. Las tasas de años con uno o dos casos no deben parecer tendencias: mostrar casos y recuentos, evitar líneas entre años sin observaciones y no suavizar huecos. Los períodos pueden ayudar a resumir después de presentar los años; no sustituyen ese eje ni convierten la selección en una muestra poblacional.

## 1. Una misma palabra puede sostener historias distintas

**Función:** abrir con la experiencia de lectura y llegar inmediatamente a la pregunta anual. Dos letras pueden nombrar miedo y hacer cosas diferentes: reconocerlo, negarlo, atribuirlo a otro o responder a una amenaza. Un tercer caso puede hablar de pérdida sin establecer temor. No publicar como ejemplo positivo ninguna clasificación heredada que la nueva pareja de lecturas deje incierta.

**Apertura propuesta:** “Cuando una canción nombra el miedo, todavía falta saber quién teme y qué está en juego. Puede defender su intimidad, pedir que alguien permanezca o negar que una aventura lo asuste. Me interesa seguir esas diferencias en canciones publicadas entre 1980 y 2026: ver qué preocupación regresa, qué respuesta cambia y qué vuelve importante para la voz que habla”.

En un párrafo adicional, distinguir los dos tiempos: los años comparan canciones; los bloques ordenados permiten leer el desarrollo dentro de cada una. Reducir las cuatro entradas actuales a estas dos escenas conceptuales.

**Puente:** “Para seguir ese cambio necesitamos saber qué entendemos por una trayectoria. Las investigaciones sobre letras ya ofrecen dos maneras de plantear el problema”.

## 2. Las letras tienen trayectorias; leer un argumento exige otro paso

**Función:** colocar antecedentes directamente pertinentes antes del método propio, con una diferencia operativa comprensible.

[Song y Beck (2023)](https://aclanthology.org/2023.tacl-1.10/) modelan la dinámica emocional de letras como series mediante un predictor por oración y un modelo de espacio de estados. El antecedente ayuda a preguntar cómo varía una señal a lo largo del texto; no establece por sí mismo quién teme, qué amenaza enfrenta ni qué relación conecta dos posiciones.

[Alberhasky y Durkee (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11098490/) analizan 12.280 letras y otro conjunto de 2.823 tras sus exclusiones. Dividen los textos en cinco segmentos y usan categorías léxicas de LIWC para estudiar establecimiento de escena, progresión y tensión cognitiva. Sus patrones agregados constituyen un antecedente narrativo, pero esos indicadores no equivalen a nuestros enlaces entre amenaza, cuidado, evitación y agencia. Tampoco corresponde trasladar sus resultados en inglés al corpus bilingüe sin comprobación.

Definir a continuación, sin catálogo de teorías, tres preguntas propias: qué preocupa a la voz; qué hace frente a ello; qué objeto o posibilidad vuelve valioso mediante esa respuesta. El valor es una interpretación textual con apoyo, no una utilidad numérica ni una declaración biográfica. Reagan y EmotionArcs pueden quedar como contexto secundario con una frase; las letras y sus repeticiones merecen conducir este apartado.

**Puente:** “Estas distinciones fijan lo que pediremos a cada canción. Ahora podemos comparar años sin cambiar la pregunta a mitad del recorrido”.

## 3. Una lectura común para canciones de años distintos

**Función:** explicar la selección y el procedimiento una sola vez, antes del primer resultado. Conservar las 120 canciones, versiones y fechas documentadas; aclarar que 2026 tiene corte parcial. La selección equilibra idiomas por período, no ofrece idéntica cobertura anual ni representa todas las canciones de cada año.

Presentar la nueva revisión uniforme: dos lecturas por canción, bloques anónimos ordenados, referencia/polaridad/voz/postura/subtema, pasos cronológicos y valor con anclajes. Explicar que son procesos del mismo asistente configurado, sin atribuirles validación humana ni independencia entre proveedores. Los desacuerdos se conservan; no introducir una tercera lectura para resolverlos. Identificar el antiguo panel de 40 como piloto y evitar que sus resultados suplanten la nueva capa anual.

Explicar embeddings y BERTopic en un párrafo: fragmentos representados como vectores, candidatos cercanos y grupos descriptivos que requieren lectura. Los cuatro encoders comparados sobre el piloto y los dos locales de la ampliación responden a alcances distintos. Promediar fragmentos ayuda a buscar vecinos, pero descarta su orden. La ficha de E5, dimensiones y parámetros puede quedar en una nota breve o material enlazado.

**Puente:** “Con la misma regla de lectura para todos los casos, el primer resultado que necesitamos ver es su distribución por año”.

## 4. El mapa anual: recurrencias, diferencias y huecos

**Función:** convertir el mapa anual en centro del artículo, antes de discutir el rendimiento de cada instrumento. Aquí deben entrar los resultados de las 240 lecturas cuando estén completos, con denominadores y estado de resolución comprobados. No anticipar una dirección temporal.

Primero recorrer el gráfico con tres preguntas: dónde reaparece un subtema; qué voces o respuestas se distinguen entre casos de años diferentes; qué años contienen tan pocos casos que sólo permiten lectura individual. Después incorporar un vecindario de embeddings que cruce años y comprobar qué sostiene su cercanía: preocupación, idioma, recurso formal o estribillo. Un grupo puede relacionar textos sin constituir un subtema del miedo.

Mantener abierta la discrepancia entre geometría y lectura. Si dos canciones cercanas difieren en polaridad o voz, esa diferencia explica el sentido de la comparación. Si dos casos del mismo año difieren, preservarlos en el mapa: el año no impone una experiencia única. Las observaciones por período sólo resumen los casos seleccionados y deben acompañarse de sensibilidad a idioma, desacuerdos y retiro de artistas.

**Puente:** “El mapa señala dónde comparar. Para explicar por qué una preocupación cambia de forma tenemos que regresar al orden de las letras”.

## 5. Lo que hace una voz con su miedo y con aquello que aprecia

**Función:** dedicar la lectura más extensa a dos o tres comparaciones seleccionadas después de cerrar la revisión. Preferir una pareja con preocupación compartida y respuestas distintas, y otra con forma argumental semejante pero objetos diferentes. Elegir casos por la relación que permiten explicar, no para llenar una canción por década ni fabricar una marcha de progreso.

En cada comparación, desarrollar situación → voz y polaridad → respuesta → valor → alternativa de lectura. Distinguir la insistencia que mantiene una posición del giro que la modifica. Un final repetido puede sostener una petición de cuidado o una resistencia; no significa automáticamente inmovilidad o fracaso. Cuando no exista un arco compartido, mostrar qué parte del vínculo permanece sin resolver.

Tratar el estilo mediante evidencia concreta: primera persona o voz colectiva, pregunta o mandato, metáfora o escena, negación, repetición y cambio de registro. Esos recursos importan porque sitúan al hablante y hacen legible su apuesta, no porque podamos adjudicar una estética completa a un año. Las paráfrasis deben conservar el conflicto sin reconstruir versos.

**Puente:** “Estas diferencias muestran qué gana la lectura al usar candidatos semánticos. También permiten preguntar qué cambiaría si buscáramos los casos con otro instrumento”.

## 6. Cuánto depende la historia del instrumento

**Función:** reunir las comprobaciones técnicas en un apartado que responda a consecuencias para la interpretación. Un solo diagnóstico visual puede mostrar sensibilidad a idioma, encoder o selección de artistas. Bayes debe aparecer únicamente cuando una tasa y su incertidumbre ayuden a decidir cuánto peso dar a una diferencia observada. Suavizar recuentos no valida el significado de una categoría ni autoriza inferencia poblacional.

Reducir RAG y LangGraph a un ejemplo de suficiencia: encontrar un pasaje relevante todavía exige decidir si sostiene la relación solicitada. Eliminar la competición extensa de benchmarks del cuerpo anual y conservar enlaces a resultados verificables. No mencionar Notebook en el post ni cerrar con instrucciones de reproducción. Las referencias deben acompañar la afirmación correspondiente.

Explicitar aquí la consecuencia que resulte de la revisión completa: qué comparaciones permanecen al cambiar supuestos y cuáles deben conservarse como preguntas. Ninguna coincidencia entre lectores transforma una categoría amplia en verdad definitiva; por eso la alternativa de lectura debe acompañar los ejemplos centrales.

**Puente:** “Después de estas comprobaciones, podemos volver a la pregunta inicial y distinguir una recurrencia observada de una transformación que los casos permiten explicar”.

## 7. Conclusiones

El cierre deberá responder, en ese orden, qué preocupaciones reaparecen entre los años observados, cómo sus voces cambian la forma de responder y qué valores vuelven legibles esas respuestas. Cada afirmación debe remitir a los casos y figuras ya discutidos. Si los datos no sostienen un cambio uniforme, el resultado debe describir esa coexistencia y su cobertura; no atribuirla a toda una generación ni a una causa política o cultural no estudiada.

El segundo párrafo deberá concretar el aporte de los embeddings: qué relaciones permitieron encontrar, dónde fueron insuficientes y cómo el orden textual cambió la interpretación. El tercero deberá cerrar sobre la apuesta de valor que las comparaciones sostengan: qué se intenta proteger, qué se decide arriesgar y qué posibilidad se desea transformar. Hablar de una formulación nueva en estos casos cuando corresponda, sin anunciar el nacimiento histórico de un valor.

La última frase tiene que culminar la lectura de las canciones. No necesita un inventario de herramientas, una llamada a ejecutar el proyecto ni una nueva agenda de trabajo. Su fuerza debe proceder de unir una diferencia contada por año con una diferencia explicada entre voces, respuestas y aquello que consideran valioso.
