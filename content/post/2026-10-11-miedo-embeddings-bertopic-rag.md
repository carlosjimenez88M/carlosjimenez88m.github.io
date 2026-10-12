---
author: Carlos Daniel Jiménez
date: 2026-10-11T18:00:00-06:00
title: "La evolución de la canción que trata sobre el miedo: experiencia desde los embeddings"
description: "Cuarenta años de canciones que hablan de miedo, leídas por un modelo de lenguaje y medidas con embeddings: la amenaza se mudó de la calle a la propia mente, el miedo pasó a confesarse en primera persona y las canciones aprendieron a atravesarlo."
categories: ["Music Analysis", "Engineering"]
tags: ["embeddings", "music", "lyrics", "narrative", "semantic-axes", "UMAP", "BERTopic", "evaluation"]
draft: true
images: ["/research/fear-lyrics/story/fear-locus-periods.png"]
socialImageAlt: "Composición de las canciones con miedo por período: la amenaza colectiva domina los ochenta y la propia mente crece hasta 2020–26"
readerGuide:
  summary: "Cuando una canción habla de miedo, ¿de qué tiene miedo, quién lo siente, cómo responde y qué protege? Leí 120 canciones de 1980 a 2026 con dos métodos que se ejecutan por separado pero comparten autor y diseño: una lectura anonimizada por pares hecha por un modelo de lenguaje y la geometría de un modelo de embeddings. Ambos apuntan en la misma dirección: el miedo se mudó de la calle a la cabeza."
  scope: "120 canciones, 12 en inglés y 12 en español por período; 56 con miedo explícito o implícito. Selección intencional, solo letras (sin audio), corte el 11 de octubre de 2026. Los porcentajes describen esta muestra, no toda la música popular."
  resources:
    - label: "Código y resultados públicos"
      url: "/examples/fear-lyrics-study.zip"
    - label: "Explorador: las 56 canciones, sus lecturas y sus arcos"
      url: "/research/fear-lyrics/story/explorador.html"
---

Lejos de criticar o alabar una época en particular, me interesa entender el significado de las canciones, explorar sus relaciones desde los embeddings y seguir cómo se desarrolla la historia que cuentan a través de su arco argumental. Poder visualizar esa evolución me lleva a preguntarme: **¿qué sentimientos perduran al pasar de una década a otra? ¿Qué valores o apreciaciones aparecen, cambian de forma o adquieren otro peso?**

La evolución social de lo que nos preocupa tiene un cambio de frente que nace en la percepción de lo urbano. El miedo, como en las películas, anduvo primero de puertas hacia afuera, en la calle, y terminó hablando del mantra del miedo mental. Eso es lo que pretendo observar a nivel de arco argumental. Esta vez tomo como hilo las canciones que hablan del miedo, desde los años ochenta hasta 2026. Quiero observar qué permanece en sus preocupaciones y qué cambia en la forma de narrarlas: quién teme, a qué le teme, qué aprecia y cómo responde. Escuchando discos, esa convivencia entre permanencia y cambio es parte de lo que me atrae de seguir la evolución musical de una banda. Aquí intento convertir esa curiosidad en una lectura que podamos contrastar y visualizar.

La respuesta corta, en esta muestra de 120 canciones en inglés y español:

1. **La amenaza se mudó hacia adentro.** En los ochenta, siete de cada diez canciones con miedo (9,5 de 13,5) la ponen en la sociedad: la calle, el Estado, la guerra. En los 2000 la amenaza más frecuente es alguien cercano, casi siempre la pareja. En 2020–26, el lugar más frecuente es la propia mente.
2. **El miedo dejó de narrarse y empezó a confesarse.** En los ochenta suele ser de otro («tú», «ellos»); desde 2000, cerca de ocho de cada diez canciones lo dicen en primera persona, singular o plural («yo» o «nosotros»). El «nosotros» aparece en los 2010.
3. **El desenlace parece cambiar.** Bajan las canciones donde el miedo crece hasta el final y suben las que lo atraviesan, aunque ambos cambios son sugerentes y no firmes (p ≈ 0,06–0,08). El bucle obsesivo, el estribillo que no se resuelve, sigue ahí en todas las décadas.
4. **Cambió lo que se protege.** De estar a salvo, a la dignidad y, en la última década, a sanar. El amor es lo que menos cambia: en cada período, entre un tercio y más de la mitad de estas canciones lo protege.

Llegué a esas conclusiones con dos mediciones que no ven los resultados de la otra: una **lectura anonimizada** de cada letra y la **geometría de un modelo de embeddings**. Son independientes en su ejecución, no en su diseño: yo elegí el corpus y escribí las frases ancla conociendo la hipótesis, y los ejes se validan contra las propias lecturas. Que coincidan es lo que me permite afirmar la dirección del cambio; donde no coinciden está, como veremos, la lección más útil para quien trabaja con embeddings. Si vienes por la ingeniería, la sección [Lo que ven los embeddings, y lo que no](#lo-que-ven-los-embeddings-y-lo-que-no) empieza con un resumen técnico.

## Cómo se lee el miedo en una canción

El corpus tiene **120 canciones**: 12 en inglés y 12 en español en cada período (1980–89, 1990–99, 2000–09, 2010–19 y 2020–26, este último parcial). Las elegí antes de ver resultados: candidatas asociadas al miedo, casos ambiguos y controles sin un motivo de miedo evidente. Es una selección intencional de éxitos con difusión heterogénea, no una muestra de toda la música popular. Por eso no comparo **cuántas** canciones con miedo hay por década, que depende de quién armó la lista, sino **cómo es el miedo** dentro de las que lo tienen.

**Primera medición: lectura anonimizada por pares.** Seis procesos lectores, subagentes de Claude Opus 5.5 en Claude Code con la configuración de muestreo por defecto (no fijé la temperatura), leyeron cada letra sin artista, título ni año, con los bloques en orden y los estribillos repetidos marcados. Cada canción recibió dos lecturas separadas, que registran:

- si hay miedo, explícito o implícito, y si se afirma o se niega;
- **dónde** está la amenaza: la sociedad o el Estado, un monstruo o presagio, otra persona, la pérdida de un vínculo, el tiempo, o la propia mente;
- **quién** teme: la voz, un «nosotros», un «tú» o un personaje;
- **cómo responde** la voz: huir, suplicar, confesar, cuidar, enfrentar;
- el **estado de cada bloque** de la letra, de la amenaza al control, y la forma del arco;
- **qué protege**: el valor en juego.

El [protocolo](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/research/fear-lyrics/story-reading-protocol.md) está publicado, también [en inglés](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/research/fear-lyrics/story-reading-protocol.en.md). Las dos lecturas coinciden en si hay miedo en el 95 % de las canciones (κ = 0,90), en dónde está la amenaza en el 94 % (κ = 0,92) y en la forma del arco en el 82 % (κ = 0,76). Hay un matiz importante: los seis lectores son el mismo asistente con el mismo protocolo, no seis personas, así que ese acuerdo mide consistencia del procedimiento, no independencia de criterio. **56 de las 120 canciones** tienen miedo según al menos una lectura; en 50, según ambas.

**Anonimizar no es cegar.** Quitar artista, título y año no impide que un modelo de lenguaje reconozca *Thriller* o *Anti-Hero*. Hice una prueba de reconocimiento: con los mismos bloques anonimizados, el mismo modelo nombró el título de 110 de las 120 canciones (92 %), incluidas todas las canciones con miedo de 1980 a 2019. Las lecturas pueden traer la interpretación cultural que el modelo aprendió de cada canción, y ese es un riesgo justo para la variable que más importa: un modelo que «sabe» que una canción de 2019 trata de salud mental puede encontrar salud mental en ella. Las cuatro canciones con miedo que no reconoció son de 2020–26 (*Ansiedad*, *Tengo Un Pensamiento*, *Algo tú*, *Die With a Smile*), así que *Ansiedad*, la más «adentro» de ese período, se leyó solo por su texto; pero cuatro canciones son muy pocas para repetir el análisis solo con ellas. Por eso importa la segunda medición: un embedding solo codifica el texto de cada bloque, así que está menos expuesto a lo que se ha escrito sobre la canción.

**Segunda medición: la geometría.** Cada bloque de letra se convierte en un vector con `multilingual-e5-large-instruct`, un modelo local, así que ninguna letra sale de la máquina. Se le da una instrucción: *representa qué teme quien habla, de dónde viene la amenaza y cómo responde*. Después le hago preguntas al espacio con frases ancla que yo mismo escribí. Esta medición nunca ve las lecturas: solo se compara con ellas al final.

## De la calle a la cabeza

{{< figure class="wide" link="/research/fear-lyrics/story/fear-locus-timeline.svg" src="/research/fear-lyrics/story/fear-locus-timeline.svg" alt="Línea de tiempo de 1980 a 2026 con las 56 canciones con miedo ubicadas en seis filas según el lugar de la amenaza, de la sociedad a la propia mente" caption="Cada punto es una canción con miedo, en el año de su primera edición. La fila indica dónde sitúan la amenaza las dos lecturas anonimizadas; un punto hueco indica que solo una lectura ve miedo o que difieren en el lugar." >}}

La figura se lee de arriba abajo y de izquierda a derecha. La fila superior, la de la amenaza colectiva, se concentra en los ochenta y noventa. La inferior, la de la propia mente, está vacía hasta 1994 y se puebla a partir de 2000.

{{< figure class="wide" link="/research/fear-lyrics/story/fear-locus-periods.svg" src="/research/fear-lyrics/story/fear-locus-periods.svg" alt="Barras apiladas por período: dónde está la amenaza y cómo responde la voz en las canciones con miedo" caption="Composición de las canciones con miedo en cada período. Igual peso para inglés y español; una canción con dos lecturas distintas aporta mitad a cada categoría." >}}

Los porcentajes dan igual peso al inglés y al español; entre paréntesis va el número de canciones que los sostiene (aparecen mitades cuando las dos lecturas difieren). Con 6,5 a 13,5 canciones con miedo por período, cada porcentaje descansa en un puñado de canciones.

**Los ochenta: el miedo tiene domicilio afuera.** El 70 % de las canciones con miedo (9,5 de 13,5) sitúa la amenaza en la sociedad, el Estado o la calle, y otro 15 % (2) en un monstruo. *Somebody's Watching Me* convierte la vigilancia en paranoia doméstica. *Sunday Bloody Sunday* canta la violencia sectaria en Irlanda del Norte. *Livin' on a Prayer*, *Fast Car* y, bajo su alegría obligada, *Don't Worry, Be Happy* tienen de fondo deudas, desalojos y trabajo perdido. En español, la noche urbana de *Escuela de calor*, la Buenos Aires de la que se huye en *En la ciudad de la furia* y la mujer a la que declaran loca en *Dr. Psiquiatra*. La voz huye, se resigna o cuida a otro; enfrenta en menos de un tercio de los casos.

**Los noventa: la calle sigue ahí, pero asoma el yo.** La violencia colectiva persiste en *Zombie*, *Matador* y *Gangsta's Paradise*, y aparece una pérdida que asusta (37 %; 2,5 de 8,5 canciones): *Don't Speak*, *Losing My Religion*. Con *Losing My Religion* y *Hurt* aparece también, por primera vez en el corpus, la amenaza dentro de uno mismo, aunque en ambas solo una de las dos lecturas la ve así.

**Los 2000: el enemigo duerme en casa.** La amenaza colectiva cae al 11 % (2 de 12,5). Domina otra persona (37 %; 5 canciones), casi siempre alguien cercano: la pareja en *Toxic*, *Time Is Running Out* y, en español, *Malo*, de Bebe, publicada el mismo año en que España aprobó su ley integral contra la violencia de género (2004); un fan obsesivo en *Stan*; alguien que controla y exige en *Numb*. Al mismo tiempo, el nu-metal y el rock gótico ponen al monstruo dentro de la cabeza: *Crawling*, *Bring Me to Life*, *Disturbia*. En América Latina, las crisis se responden con esperanza o plegaria: *Color Esperanza*, cuyo disco salió en noviembre de 2001, semanas antes del estallido argentino, y *A Dios Le Pido*.

**Los 2010: la calle vuelve, en plural.** La historia no es una línea recta. La amenaza colectiva rebota al 25 % (3,5 de 12) con *Alright*, *This Is America*, *Latinoamérica* y *Stressed Out*. Pero en las tres primeras la voz ya no huye: enfrenta, y lo hace en plural, como «nosotros». En paralelo, la salud mental entra al centro del pop con *Demons*, *Chandelier* y *bury a friend*, que Billie Eilish describió como escrita [desde el monstruo debajo de su cama, que es ella misma](https://www.rollingstone.com/music/music-news/billie-eilish-bury-a-friend-video-new-lp-786599/).

**2020–26: el miedo como estado de ánimo.** La propia mente es el lugar más frecuente (35 %; 2 de 6,5 canciones): *Ansiedad*, *Anti-Hero*. El cuerpo y el poder ajeno aparecen en *Your Power* y *Ay Mamá*. Y crece un miedo de paso, un instante de vértigo dentro de una canción de amor: *Tengo Un Pensamiento*, *Algo tú*, *Die With a Smile*. Es la década con menos casos (6,5 canciones con miedo), así que sus porcentajes son los más frágiles.

En la muestra completa, la amenaza colectiva cae **12,9 puntos porcentuales por década** (p = 0,01) y la amenaza interior sube **8,2 puntos por década** (p = 0,02). Son pendientes ponderadas con efectos de idioma, y el valor p se obtiene permutando los años dentro de cada idioma. Son dos de las 18 pendientes que probé; con corrección por comparaciones múltiples conservan q = 0,065 y 0,084 (ver «Qué resiste y qué no»).

### Dos idiomas, dos caminos

En inglés, la mente aparece en los noventa y ocupa un tercio de las canciones con miedo desde 2000: el grunge, el nu-metal y el pop confesional son, entre otras cosas, la historia de esa mudanza. En español no aparece hasta *Ansiedad*, en 2020. Antes, el miedo íntimo pasa por **la casa y la pareja** (*Malo*, *La Puerta Violeta*) y por **la pérdida** (*Miedo*, *A Dios Le Pido*). Además, se responde con más frecuencia enfrentando: entre el 57 % y el 71 % de las canciones con miedo desde 2000 (de 2 a 2,5 canciones de 3,5 a 4 por década), frente al 40–44 % en inglés, y la diferencia aparece en todas las décadas. Con dos a cuatro canciones por celda, esto es una observación de casos y no una estimación.

## Del «tú» al «yo», y al «nosotros»

¿De quién es el miedo? En los ochenta, solo el 33 % de las lecturas (9 de 27) atribuye el miedo a la voz que canta; el resto pertenece a un «tú» al que se le habla (30 %; 8), como en *Thriller*, o a personajes de una historia (26 %; 7), como en *Cruz de Navajas* o *Hijo de la Luna*. En los 2000, la voz confiesa su propio miedo en el 80 % de las lecturas (20 de 25). En los 2010, el 42 % de las lecturas (10 de 24) encuentra un «nosotros»: el miedo vuelve a ser colectivo, pero ahora lo canta quien lo sufre.

Sumando «yo» y «nosotros», el miedo es de quien canta en el 40 % de las canciones en los ochenta (5,5 de 13,5) y en el 100 % en 2010–19 (12 de 12), una subida de **12,7 puntos por década** (p = 0,005; q = 0,065 con corrección). Es la tendencia más fuerte del estudio, y no es una rareza de esta muestra. [DeWall y colaboradores](https://uknow.uky.edu/research/hit-songs-offer-window-societys-psyche) encontraron más pronombres de primera persona en los éxitos de Billboard entre 1980 y 2007, y [Parada-Cabaleiro y colaboradores](https://www.nature.com/articles/s41598-024-55742-x), en 353.320 letras en inglés, que las canciones se volvieron más personales, más repetitivas y más negativas. Esos estudios cuentan palabras. Aquí se ve qué hace la primera persona con el miedo: lo confiesa.

## Los subtemas: de la guerra y la calle a la salud mental

{{< figure class="wide" link="/research/fear-lyrics/story/fear-subtopics-context.svg" src="/research/fear-lyrics/story/fear-subtopics-context.svg" alt="Mapa de calor de once subtemas del miedo por período y barras con el AUC de los embeddings para reconocer cada subtema" caption="Izquierda: porcentaje de canciones con miedo cuyo contexto incluye cada subtema, según los lectores (una canción puede tener varios). Derecha: si una pregunta al espacio de embeddings distingue las canciones con ese subtema." >}}

Guerra, policía, calle, pobreza y terror ocupan los ochenta y noventa y desaparecen en 2020–26. La salud mental está en todas las décadas, pero cambia de forma. En los ochenta (22 %; 3 de 13,5 canciones) es algo que el mundo le hace a uno: la alienación de *Mad World*, la paranoia de *Somebody's Watching Me*, el encierro psiquiátrico de *Dr. Psiquiatra*. Desde 2000 (31–44 %) es la propia mente la que ataca. El amor está en todas las décadas, entre el 35 % y el 43 %: es el contexto más estable del miedo en la canción popular.

La columna de la derecha anticipa la parte técnica. Si le pregunto al espacio de embeddings por un **escenario** concreto, lo reconoce bien: guerra (AUC 0,99, aunque solo con 4 canciones), control en la pareja (0,82), calle y crimen (0,81), monstruos (0,80). Si le pregunto por una **condición** abstracta, falla: pobreza (0,50), fama (0,53), familia (0,54). El modelo reconoce mejor una escena que una estructura social.

## Lo que ven los embeddings, y lo que no

> **La versión corta, para ingenieros.**
> 1. Los embeddings multilingües ordenan las letras primero por idioma: hasta retirar la media de cada idioma, el 100 % de los vecinos más próximos de una canción comparte su idioma.
> 2. La estructura libre no recupera el concepto. UMAP, el agrupamiento y los vecinos más próximos reúnen pasajes por escena, vocabulario y estribillo; un clasificador de 10 vecinos acierta menos que la línea base (59 % frente a 65 %).
> 3. Una pregunta explícita sí lo recupera. Un eje construido con frases ancla bilingües separa las canciones «adentro» de las «afuera» con AUC 0,89 [0,79–0,96] y reproduce la tendencia de los lectores sin ver sus etiquetas.
> 4. El prefijo de instrucción no ayuda de forma medible: su ganancia sobre el mismo modelo cae dentro del ruido del bootstrap.
> 5. Los embeddings ubican una letra mejor de lo que siguen su movimiento: dentro de una canción, la correlación con los estados de los lectores es 0,15.
> 6. Un lector basado en un modelo de lenguaje no es ciego: reconoció el 92 % de las letras anonimizadas.

Un primer intento natural es dejar que los subtemas del miedo salgan solos, con BERTopic sobre los embeddings. Cuando lo probé, los grupos resultaron ser sobre todo idioma y vocabulario. Esto explica por qué, y qué funciona en su lugar.

### 1. Lo primero que ve un embedding es el idioma

{{< figure class="wide" link="/research/fear-lyrics/story/fear-language-geometry.svg" src="/research/fear-lyrics/story/fear-language-geometry.svg" alt="Tres proyecciones UMAP de los mismos bloques: con E5 small y E5 large aparecen dos islas por idioma; al retirar la media de cada idioma se mezclan" caption="Sin corrección, los cinco vecinos más cercanos de cada canción están en su mismo idioma. Retirar la media de cada idioma devuelve esa proporción al azar." >}}

Con E5 small y E5 large, el **100 %** de los cinco vecinos más próximos de cada canción está en su mismo idioma. El mapa separa inglés de español antes que cualquier cosa. La instrucción de miedo no lo resuelve sola. Retirar la media de cada idioma, como proponen [Libovický, Rosa y Fraser](https://aclanthology.org/2020.findings-emnlp.150/), devuelve esa proporción al 51 %, es decir, al azar. Cualquier análisis de letras bilingües que se salte este paso descubrirá, con mucha confianza, que el inglés y el español son distintos.

### 2. El mapa libre no responde la pregunta

{{< figure class="wide" link="/research/fear-lyrics/story/fear-map-vs-question.svg" src="/research/fear-lyrics/story/fear-map-vs-question.svg" alt="Izquierda: UMAP de 320 pasajes de miedo coloreados por el lugar de la amenaza, mezclados. Derecha: las canciones sobre el eje afuera-adentro, ordenadas por zona" caption="Los mismos pasajes de miedo, ya sin la señal de idioma. El mapa libre muestra tendencias sin fronteras; el eje separa las zonas que marcaron los lectores." >}}

Sin el idioma, el agrupamiento libre de canciones se vuelve inestable (ARI bootstrap ≈ 0,4) y apenas se parece a las categorías de los lectores (NMI ≈ 0,3). Los vecinos de un pasaje de miedo comparten escena, vocabulario o estribillo, pero no el lugar de la amenaza. Un clasificador de 10 vecinos acierta la zona en el 59 % de los pasajes, menos que el 65 % que se obtiene diciendo siempre «afuera». UMAP y BERTopic siguen siendo buenos para explorar, pero no miden un concepto por el que no se les pregunta.

### 3. Preguntar con ejes

Lo que sí funciona es hacer una pregunta explícita. Para cada dimensión escribí dos polos de 16 frases ancla: 8 en inglés y 8 en español, traducciones paralelas, todas en primera persona y con miedo explícito, para que solo cambie lo que quiero medir. Algunos ejemplos: *tengo miedo del extraño que espía mi casa* frente a *tengo miedo de mis propios pensamientos*, o *me escondo hasta que pase* frente a *tengo miedo, pero lo enfrento*. El eje es la diferencia entre los dos polos, una idea que viene de la [geometría de la cultura de Kozlowski, Taddy y Evans](https://arxiv.org/abs/1803.09288) y de [SemAxis](https://arxiv.org/abs/1806.05521). Cada letra se proyecta sobre él.

| AUC frente a las lecturas [intervalo del 90 %] | Afuera → adentro | Amenaza colectiva | Huir → enfrentar |
| --- | ---: | ---: | ---: |
| MiniLM multilingüe | 0,77 [0,61–0,91] | 0,77 [0,65–0,88] | 0,54 [0,38–0,70] |
| E5 small | 0,79 [0,64–0,92] | 0,73 [0,61–0,84] | 0,69 [0,55–0,83] |
| E5 large instruct, sin instrucción | 0,85 [0,74–0,94] | 0,80 [0,69–0,89] | 0,69 [0,54–0,84] |
| **E5 large instruct, con instrucción de miedo** | **0,89 [0,79–0,96]** | **0,85 [0,74–0,93]** | **0,72 [0,57–0,85]** |
| Canciones comparadas | 10 adentro frente a 34 afuera | 22 colectivas frente a 34 | 22 enfrentan frente a 17 huyen |

Con 56 canciones, los intervalos son anchos. En comparaciones bootstrap pareadas, que remuestrean las mismas canciones para todos los encoders, el modelo grande con instrucción supera a MiniLM en los tres ejes y a E5 small en dos. Pero su ganancia de 0,02 a 0,05 sobre el mismo modelo sin instrucción cae dentro del ruido: esos intervalos incluyen cero (+0,035 [−0,012, +0,085] en el eje afuera→adentro). No puedo afirmar que la instrucción ayude. Con las anclas, el eje afuera→adentro clasifica correctamente el 100 % de las frases que dejo fuera al construirlo.

{{< figure class="wide" link="/research/fear-lyrics/story/fear-compass.svg" src="/research/fear-lyrics/story/fear-compass.svg" alt="Dispersión de las 56 canciones con miedo en dos ejes semánticos, coloreadas por el lugar de la amenaza, y medias por período con intervalos" caption="La geometría no ve las etiquetas de los lectores; el color sí. Las canciones que los lectores leen como «adentro» quedan a la derecha. A la derecha de la figura, las medias por período: el desplazamiento afuera→adentro, con el regreso de la calle en los 2010." >}}

Sin mirar ninguna etiqueta, el eje afuera→adentro se desplaza **+0,23 desviaciones estándar por década** (p = 0,01; q = 0,065). La dirección se repite con los cuatro encoders, entre +0,20 y +0,27 (p ≤ 0,024), y reproduce el zigzag de los lectores: avanza en los 2000, retrocede en los 2010 y vuelve a avanzar. El eje de agencia apenas se mueve (p = 0,21). La geometría capta mejor **dónde** está la amenaza que **cómo** responde la voz.

## El arco: cómo se mueve el miedo dentro de una canción

Un tópico describe de qué habla una canción; un arco describe qué le pasa a la voz mientras canta. Que las letras tienen arco ya se sabe: [Alberhasky y Durkee](https://doi.org/10.1371/journal.pone.0303188), con 12.280 letras, encontraron que las canciones montan la escena, avanzan la trama y concentran la tensión a la mitad, como los relatos que estudiaron [Boyd, Blackburn y Pennebaker](https://www.science.org/doi/10.1126/sciadv.aba2196). Mi pregunta es más estrecha: qué le pasa al **miedo** de la voz a lo largo de ese arco. Para verlo, cada lector asignó un estado a cada bloque de la letra, de la amenaza (rojo) al control (azul), en el orden en que suena.

{{< figure class="wide" link="/research/fear-lyrics/story/fear-story-strips.svg" src="/research/fear-lyrics/story/fear-story-strips.svg" alt="Tiras de colores para doce canciones de 1982 a 2022 que muestran el estado de la voz bloque a bloque según dos lecturas" caption="Cada fila son dos tiras, una por lectura anonimizada, en el orden de la letra con los estribillos repetidos incluidos. Rojo: amenaza o miedo; azul: desafío o agencia." >}}

Las tiras hacen visible lo que un promedio esconde. *Thriller* y *Somebody's Watching Me* empiezan en rojo y terminan en rojo: el miedo crece. *Crawling*, *bury a friend* y *Anti-Hero* alternan rojo y rosa sin salir: son bucles, el estribillo como obsesión. *Lose Yourself*, *Malo*, *La Puerta Violeta* y *Ansiedad* empiezan en rojo y terminan en azul: la voz atraviesa el miedo.

{{< figure class="wide" link="/research/fear-lyrics/story/fear-arcs.svg" src="/research/fear-lyrics/story/fear-arcs.svg" alt="Trayectorias medianas de seis formas de arco y su composición por período" caption="Arriba: cómo se mueve cada forma de arco a lo largo de la letra. Abajo: qué proporción de las canciones con miedo tiene cada forma en cada período." >}}

En los ochenta y noventa, un tercio de las canciones con miedo **escala** (4,5 de 13,5 y 2 de 8,5 canciones). Desde 2010 lo hace solo entre el 6 % y el 10 % (1 de 12 y 0,5 de 6,5). El **giro a la agencia** pasa de casi no existir (0,5 de 13,5 canciones, y luego ninguna) a ser la forma más frecuente en 2000–09 y 2010–19 (37 % y 42 %; 3,5 canciones en cada período). Ambos cambios son sugerentes, no firmes: −7,7 y +6,5 puntos por década, con p = 0,06 y 0,08 (q = 0,155 con corrección). El **bucle** casi no cambia: después del 48 % de los ochenta (6,5 de 13,5), ronda el 30 % en cada década. Y en 2020–26 crece el miedo **episódico** (38 %; 3 de 6,5 canciones; +6,9 puntos por década, p = 0,01, q = 0,065), el instante de vértigo dentro de una canción de amor.

¿Pueden los embeddings dibujar estos arcos solos? Solo a medias. La correlación entre el estado que asignan los lectores y la posición de cada bloque en el eje de agencia es de 0,32 en el conjunto, pero de 0,15 en promedio dentro de una misma canción. El vector sabe en qué zona está una letra, pero sigue mal su movimiento. El arco todavía necesita lectura.

## Qué protege la voz

Me interesaba una pregunta más: qué vuelve valioso algo para quien canta. El miedo es una buena puerta, porque siempre teme perder algo.

{{< figure class="wide" link="/research/fear-lyrics/story/fear-values.svg" src="/research/fear-lyrics/story/fear-values.svg" alt="Seis paneles de barras con el valor en juego por período: estar a salvo, la vida de otros, el vínculo amoroso, el propio yo, dignidad, sanar" caption="Valor en juego según las paráfrasis de los lectores, codificado con reglas explícitas de palabras clave. Una canción puede proteger varias cosas." >}}

**Estar a salvo** (la privacidad, el refugio, sobrevivir la noche) domina los ochenta (43 %; 6 de 13,5 canciones) y desaparece en 2020–26. **El propio yo** (el autocontrol, la identidad, sentirse vivo) crece en los 2000 (22 %; 4 canciones, todas en inglés), con *Crawling* y *Bring Me to Life*. **Dignidad, libertad y voz** llega al 71 % en 2010–19 (7 de 12 canciones: todas las canciones con miedo en español de la década y 3,5 de 8,5 en inglés; el porcentaje da igual peso a cada idioma): *Alright*, *This Is America*, *La Puerta Violeta*, *Take Me to Church*. **Sanar y ser aceptado** es casi exclusivo de la última década (32 %; 2 de 6,5 canciones): poder hablar y respirar en *Ansiedad*, ser querida a pesar de los defectos en *Anti-Hero*. Y **el vínculo amoroso** atraviesa todo el período, entre el 35 % y el 57 %: es lo que más se protege en cuatro de las cinco décadas.

Leído en secuencia, cuenta otra versión de la misma mudanza: primero se protege el cuerpo, después la identidad y la dignidad, y al final la salud. Esta codificación describe las paráfrasis de los lectores; no es una medida independiente, y la última década tiene pocos casos.

## Qué resiste y qué no

- **Resiste.** Con solo las 50 canciones en las que ambos lectores ven miedo, todas las tendencias se mantienen o se refuerzan: amenaza interior +9,2 puntos por década (p = 0,01), colectiva −13,5 (p = 0,008). Los cuatro encoders reproducen el eje afuera→adentro.
- **Se debilita.** 24 de las 56 canciones con miedo fueron elegidas por eso. Sin ellas quedan 32, y la dirección se conserva: interior +3,5 puntos por década, colectiva −10,9, eje +0,20 DE por década. Pero la evidencia baja (p entre 0,07 y 0,37). Parte del contraste lo puso la selección. Lo que sostiene la **dirección**, no la magnitud, es su coincidencia con estudios de miles de letras.
- **Comparaciones múltiples.** Probé 18 pendientes. Con corrección de Benjamini–Hochberg, los cinco resultados más fuertes (el miedo en primera persona, la amenaza colectiva, el eje geométrico afuera→adentro, el miedo episódico y la amenaza interior) tienen q entre 0,065 y 0,084: sobreviven a una tasa de falsos descubrimientos del 10 %, no del 5 %. Todo lo que tiene p > 0,05 (la escalada, el giro a la agencia, enfrentar como respuesta) debe leerse como sugerente.
- **Es frágil.** 2020–26 tiene 6,5 canciones con miedo, y los cortes por idioma tienen entre dos y cuatro por celda.
- **No es ciego.** Los lectores son el mismo modelo, y ese modelo reconoce el 92 % de las canciones anonimizadas: su acuerdo no equivale al de personas distintas y sus lecturas pueden traer lo que el modelo ya sabía de cada canción. El eje geométrico, que solo lee texto, es la medición menos expuesta a ese sesgo, y apunta en la misma dirección.
- **No se mide.** Leo letras: no audio, interpretación vocal ni recepción. Un año de publicación no es un año de escucha.

## Qué perdura y qué cambia

Perdura **el amor** como lo que el miedo protege, y perdura **el bucle**: en cada década, al menos tres de cada diez de estas canciones dan vueltas alrededor de su miedo sin resolverlo, porque un estribillo también es una forma de no poder salir.

Cambia **el lugar** de la amenaza, que se mudó de la calle a la casa y de la casa a la cabeza. Cambia **la persona** que teme, del «tú» narrado al «yo» que confiesa. **El desenlace** parece cambiar: las canciones parecen aprender a atravesar el miedo además de padecerlo, aunque ese cambio es sugerente y no firme. Y cambia **lo que se protege**: de sobrevivir a ser digno y, ahora, a sanar.

En los ochenta el miedo era algo que pasaba afuera y que la canción contaba. En los 2020 es algo que pasa adentro y que la canción confiesa, a veces para salir de él.

Para quien trabaja con embeddings, la lección cabe en dos pasos: **primero, quitar el idioma; después, preguntar.** Un mapa libre ofrece vecinos, y los vecinos de una letra comparten escena, vocabulario o estribillo. Una dirección anclada, validada contra una lectura independiente, contesta la pregunta que uno trae. Y cuando lo que se quiere medir es un movimiento dentro de la canción, como un arco, hay que volver a leer. Si quien lee es un modelo de lenguaje, primero hay que probar lo que ya sabe: anonimizar no es cegar.

<details>
<summary><strong>Método, datos y reproducción</strong></summary>

- **Corpus.** 120 letras con hashes de integridad, procedentes de LRCLIB y Genius. Las letras permanecen en una caché privada; los materiales públicos contienen IDs, categorías, puntuaciones y paráfrasis breves sin citas. Año: primera edición comercial de la grabación incluida.
- **Lectura.** [Protocolo](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/research/fear-lyrics/story-reading-protocol.md) ([versión en inglés](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/research/fear-lyrics/story-reading-protocol.en.md)). Seis procesos lectores (subagentes de Claude Opus 5.5 en Claude Code, muestreo por defecto, sin fijar la temperatura), 40 casos cada uno y asignación aleatoria (semilla 20261012), de modo que cada canción tiene dos lecturas de lectores distintos. Prueba de reconocimiento: con los mismos bloques anonimizados, el mismo modelo nombró 110 de los 120 títulos. Un validador comprueba el esquema, los IDs y el orden de los pasos del arco. Agregación por voto fraccional.
- **Bloques.** Hasta ocho líneas y al menos cuatro palabras, en orden: 1.333 bloques con repeticiones, 1.137 únicos.
- **Embeddings.** `intfloat/multilingual-e5-large-instruct` (1.024 dimensiones, local), con instrucción de miedo y sin ella. MiniLM multilingüe y E5 small como comparación. Idioma retirado por centrado (para vecindarios y agrupamiento) o por proyección de una dirección estimada con anclas paralelas (para ejes y conceptos).
- **Ejes y conceptos.** `semantic-anchors.json`, escrito antes de proyectar las letras. Validación interna (anclas fuera de muestra) y externa (AUC frente a las lecturas, con intervalos bootstrap del 90 % y diferencias pareadas entre encoders).
- **Tendencias.** Medias por período con igual peso por idioma. Pendiente por década por mínimos cuadrados ponderados, con efectos de idioma. Valor p por 10.000 permutaciones de años dentro de cada idioma, con q de Benjamini–Hochberg sobre las 18 pendientes. Intervalos bootstrap del 90 % remuestreando canciones.
- **Reproducir.** `embed_instruct.py` → `fear_story.py` → `visualize_story.py`. Los resultados públicos quedan en `results/story-v3`.

</details>
