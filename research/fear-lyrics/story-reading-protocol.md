# Lectura argumental del miedo: protocolo v3

Fecha: 12 de octubre de 2026. Corte del corpus: 11 de octubre de 2026. Se aplica a las 120 canciones con el mismo procedimiento.

Esta lectura sustituye, para la pregunta histórica, a la revisión anual que quedó incompleta (104 de 240 lecturas). Responde a tres preguntas por canción: **dónde está la amenaza**, **cómo responde la voz** y **cómo se mueve la letra entre esos estados**. Las lecturas anteriores (panel de 40 y revisión anual parcial) se conservan sin cambios.

## Diseño

- Seis lectores (L1–L6), procesos independientes del asistente con el mismo protocolo. Cada lector recibe 40 casos y cada canción recibe exactamente **dos lecturas** de lectores distintos. Asignación aleatoria con semilla 20261012. No son anotadores humanos ni proveedores independientes.
- Los casos son ciegos: sin artista, título, año, idioma declarado, etiqueta de selección ni resultados de embeddings. Se entregan los bloques en orden (`b01`, `b02`, …); un bloque repetido aparece como `repeat_of`, sin texto.
- Se lee solo el texto. Si se reconoce la canción, no se usan biografía, videoclip ni recepción.
- Las salidas públicas contienen categorías, IDs y paráfrasis breves en español; nunca citas ni versos reconstruidos.

## Campos

| Campo | Valores |
| --- | --- |
| `fear_reference` | `explicit` (miedo/temor nombrado, aunque se niegue), `implied` (situación de amenaza o temor apoyada sin nombrarlo), `absent`, `uncertain`. Tristeza, desamor o deseo no bastan. |
| `fear_polarity` | `affirmed`, `denied`, `hypothetical`, `mixed`, `absent`, `unclear` |
| `who_fears` | `speaker`, `speaker_collective`, `addressee`, `third_party`, `nobody`, `unclear` |
| `threat_locus` | Dónde sitúa la letra la fuente de la amenaza: `external_person` (alguien concreto: acosador, pareja, perseguidor), `external_collective` (sociedad, Estado, guerra, ciudad, economía, sistema), `supernatural_fiction` (monstruos, terror narrado, fantasía), `relational_loss` (perder a alguien o un vínculo), `internal_self` (la propia mente, el propio cuerpo, el propio yo como amenaza), `existential_time` (futuro, muerte, paso del tiempo, incertidumbre sin agente), `none`, `uncertain` |
| `threat_locus_secondary` | Mismo conjunto o `null` |
| `response` | Respuesta dominante de la voz: `flight_avoidance`, `paralysis`, `plea_prayer`, `defiance_confrontation`, `confession_naming`, `care_protection`, `resignation`, `transformation_agency`, `none`, `unclear` |
| `block_states` | Un estado por cada bloque, repeticiones incluidas: `threat`, `fear`, `avoidance`, `plea`, `defiance`, `agency`, `care`, `confession`, `loss`, `desire`, `celebration`, `reflection`, `filler` |
| `arc_shape` | `loop` (la situación de miedo se reitera sin cambio), `escalation` (se intensifica hacia el final), `release` (cede por alivio externo, consuelo o tiempo), `turn_to_agency` (la voz pasa del miedo a enfrentar o decidir por sí misma), `collapse` (de la esperanza o el control hacia el miedo, la pérdida o la resignación), `episodic` (el miedo aparece en un solo lugar, sin desarrollo), `no_fear_arc` |
| `arc_steps` | 2–4 pasos `{"ids":[...],"reading":"≤12 palabras"}` con IDs estrictamente crecientes entre pasos; vacío si `no_fear_arc` |
| `threat_object` | ≤8 palabras: a qué se teme |
| `value_at_stake` | ≤8 palabras: qué aprecia, protege o quiere transformar la voz |
| `context_markers` | Hasta 3: `war_conflict`, `state_police_surveillance`, `urban_violence_crime`, `economy_class`, `migration_borders`, `gender_violence`, `mental_health`, `media_technology_fame`, `religion_faith`, `nature_disaster`, `romance_sex`, `family_home`, `horror_fiction`, `none` |
| `confidence` | `high`, `medium`, `low` |
| `counterreading` | ≤20 palabras: la lectura alternativa más fuerte |

Reglas: `threat_locus`, `response` y `arc_shape` se registran también cuando el miedo es `uncertain`; con `absent` se usan `none` / `no_fear_arc` salvo que haya amenaza sin temor. `internal_self` exige que el texto presente la propia mente, conducta o identidad como peligro (no basta una voz en primera persona). `external_collective` exige un agente o condición social, no un individuo.

## Agregación

Se conserva cada lectura. Una categoría se considera **acordada** cuando ambas lecturas coinciden; los desacuerdos se reportan y, para las figuras, la canción contribuye con peso 0,5 a cada categoría propuesta (voto fraccional). Las series temporales se muestran con recuentos de canciones, separadas por idioma, y con intervalos bootstrap por canción. El acuerdo entre lectores (κ de Cohen) se publica por campo.

`block_states` alimenta el arco: cada estado se proyecta en dos ejes ordinales (amenaza ↔ control) y se interpola a diez posiciones del tiempo de la letra. Los embeddings se comparan con esas lecturas; no se usan para crearlas.
