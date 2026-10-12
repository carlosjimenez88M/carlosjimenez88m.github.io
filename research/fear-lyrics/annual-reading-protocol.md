# Lectura uniforme para el mapa anual

Revisión iniciada el 12 de octubre de 2026. Corte del corpus: 11 de octubre de 2026. Se mantienen las 120 letras, sus versiones y el panel anterior sin modificaciones.

La lectura nueva examina las 120 canciones con el mismo protocolo. Tres procesos de revisión del asistente reciben lotes aleatorizados de 80 casos; cada canción recibe exactamente dos lecturas. Son procesos del mismo asistente configurado, no tres proveedores independientes. No se generan llamadas con las claves API del proyecto. Esta fase es posterior a los análisis anteriores y sus etiquetas son interpretaciones revisables.

Los casos omiten artista, título, fecha, etiquetas de selección y resultados de embeddings. Las letras conocidas pueden reconocerse. Se conservan bloques ordenados de hasta ocho líneas y al menos cuatro palabras, incluidas repeticiones. Letras y lotes quedan privados; las salidas públicas admiten categorías, IDs y paráfrasis breves, nunca citas.

## Reglas de lectura

- Leer el texto completo del caso, no suponer la emoción por vocabulario o memoria del artista. Distinguir voz lírica de persona real.
- `explicit`: referencia léxica clara al miedo o temor, incluso negada o hipotética. `implied`: situación de amenaza o temor apoyada en el texto sin esa referencia explícita. Tristeza, desamor, incertidumbre o una petición no bastan por sí solas para inferir miedo. `absent`: no se encuentra referencia apoyada. `uncertain`: lectura de miedo plausible pero competida o insuficiente.
- La polaridad distingue `affirmed`, `denied`, `hypothetical`, `mixed`, `absent`, `unclear`. Referir al miedo no equivale a experimentarlo.
- Voz: `first_singular`, `first_collective`, `third_person_story`, `dialogue_or_multiple`, `impersonal`, `unclear`.
- Postura: `avoidance`, `resistance`, `care`, `ambivalence`, `resignation`, `other`, `unclear`.
- Subtemas: como máximo dos de `surveillance_pursuit`, `collective_violence`, `loss_abandonment`, `fragility_identity`, `uncertainty_change`, `relational_coercion`, `supernatural`, `other`, `none`, `uncertain`. Cada motivo requiere apoyo propio; no imponer esta ontología como resultado de BERTopic.
- Arco: `repetition`, `contrast`, `confrontation`, `turn`, `mixed`, `no_supported_arc`, `unresolved`. Repetición requiere al menos dos posiciones ordenadas que reiteren una situación relevante. Giro requiere cambio apoyado de posición o respuesta; cambiar de verso no basta. Confrontación vincula amenaza con respuesta; contraste puede conservar posiciones opuestas sin resolución.
- Para un arco apoyado, registrar dos o tres pasos con IDs estrictamente crecientes: el máximo ID del paso anterior debe ser menor al mínimo del siguiente. Cada flecha enlaza pasos consecutivos. No usar spans que crucen, no reparar el orden a posteriori y no fabricar un arco para toda letra.
- El valor en juego es una interpretación breve de lo que la voz aprecia, protege o desea transformar, con IDs propios. No se adjudican utilidad numérica, novedad histórica ni intención biográfica.
- Usar `uncertain` o `unresolved` cuando correspondan. Anotar una alternativa de lectura breve; ningún caso debe forzarse a hablar de miedo.

## Registro privado de cada lector

Un JSON por caso con estas claves:

```json
{
  "case_id": "N001",
  "fear_reference": "uncertain",
  "fear_polarity": "unclear",
  "reference_support_ids": [],
  "voice": "unclear",
  "stance": "unclear",
  "subtopics": ["uncertain"],
  "subtopic_support_ids": {},
  "arc_type": "unresolved",
  "arc_steps": [],
  "arc_links": [],
  "value_reading": "Lectura sin resolver",
  "value_support_ids": [],
  "counterreading": "Alternativa breve sin reproducir versos"
}
```

Pasos: `{"ids":["b001"],"state":"fear","reading":"Paráfrasis de hasta diez palabras"}`. Estados: `threat`, `fear`, `avoidance`, `loss`, `care`, `agency`, `reflection`, `negation`, `celebration`, `exposure`, `unclear`. Vínculos: `{"from":0,"to":1,"relation":"repetition"}`.

Las paráfrasis deben ser originales y breves, sin citas, versos reconstruidos ni datos externos. Todos los IDs deben existir en el caso. Una ausencia puede tener IDs que justifiquen la lectura alternativa, pero no inventar evidencia positiva.

## Agregación y temporalidad

Se conserva cada voto. Presencia de referencia al miedo requiere que ambas lecturas sean explícitas o implícitas; los desacuerdos de presencia permanecen sin resolver. Cada subtema requiere dos votos que lo vinculen con referencia al miedo. Un arco resuelto requiere dos categorías iguales y dos estructuras cronológicas válidas. No se elige una tercera lectura para cerrar desacuerdos.

El gráfico anual muestra cada año observado con su número de canciones, los motivos presentes y los casos identificables. Años sin observaciones aparecen vacíos, nunca como frecuencia cero. Las tasas de uno o dos casos no se presentan como tendencias poblacionales. Las síntesis por período se ponderan por canción e idioma y se acompañan de sensibilidad a casos sin resolución y a retirar artistas; las lecturas de valores permanecen cualitativas.
