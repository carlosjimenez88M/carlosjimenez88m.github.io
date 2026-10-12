"""Construye y ejecuta el cuaderno narrativo del estudio (miedo-embeddings-bertopic-rag.ipynb).

Lee solo resultados públicos (results/story-v3) y figuras (static/research/fear-lyrics/story); no necesita letras ni APIs.
Uso: python build_story_notebook.py [--no-execute]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / 'miedo-embeddings-bertopic-rag.ipynb'

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS = [
    md("""# ¿Dónde vive el miedo? 120 canciones, 1980–2026

**Cuaderno en español · lectura anonimizada por pares + embeddings locales · corte: 11 de octubre de 2026**

La pregunta: cuando una canción popular habla de miedo, **¿de qué tiene miedo, quién lo siente, cómo responde y qué protege?** ¿Y cómo cambia eso entre 1980 y 2026?

El cuaderno combina dos mediciones independientes sobre las mismas 120 letras (12 en inglés y 12 en español por período):

1. **Lectura anonimizada por pares.** Seis procesos lectores (subagentes de Claude Opus 5.5 en Claude Code, temperatura por defecto, el mismo protocolo) leen cada letra sin artista, título ni año. Anonimizar no ciega: el mismo modelo identifica el 92 % de las letras (sección 2). Cada canción recibe dos lecturas: si hay miedo, dónde está la amenaza, quién teme, cómo responde la voz, el estado de cada bloque y qué está en juego.
2. **Geometría de embeddings.** `multilingual-e5-large-instruct`, local, con una instrucción orientada al miedo, sin la señal de idioma, proyectado sobre ejes construidos con frases ancla bilingües. Los ejes no ven las lecturas.

### Hallazgos

| | Qué cambia | Evidencia |
|---|---|---|
| 1 | **Dónde.** La amenaza se muda de la calle, el Estado y el monstruo a la pareja y, después, a la propia mente. | Lectores: amenaza colectiva −12,9 pts/década (p = 0,01); interior +8,2 (p = 0,02). Embeddings, sin etiquetas: eje afuera→adentro +0,23 DE/década (p = 0,01), estable en cuatro encoders. |
| 2 | **Quién.** El miedo pasa de narrarse (tú, ellos) a confesarse (yo) y, en los 2010, a compartirse (nosotros). | Teme quien canta: 40 % → 100 % (2010–19); +12,7 pts/década (p = 0,005). |
| 3 | **Cómo se mueve la canción** (sugerente). Menos miedo que crece, más miedo que se atraviesa. El bucle persiste. | Escalada 33 % → 6–10 % (p = 0,06); giro a la agencia 3 % → 37–42 % (p = 0,08); bucle ~30 % desde los noventa. |
| 4 | **Qué protege.** De estar a salvo a la dignidad y, al final, a sanar. El vínculo amoroso no se mueve. | «Estar a salvo» 43 % → 0; «dignidad, libertad, voz» 71 % en 2010–19; «sanar» aparece en 2020–26 (32 %). |
| 5 | **Lección técnica.** Un embedding ve primero el idioma; el clustering libre no encuentra el lugar de la amenaza; una pregunta explícita sí. | 100 % de vecinos del mismo idioma sin corregir; kNN por debajo de la línea base; eje anclado AUC 0,89. |

Con corrección de Benjamini-Hochberg sobre las 18 pendientes, las cinco principales tienen q = 0,065–0,084 (sobreviven a una tasa de falsos descubrimientos del 10 %, no del 5 %). Límites que acompañan cada número: lecturas no ciegas (el modelo reconoce las canciones), muestra intencional (excluir las canciones elegidas *por* miedo debilita las tendencias), 6,5–13,5 canciones con miedo por período, lectores del mismo modelo y solo texto (sin audio)."""),
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

pd.set_option('future.infer_string', False)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', 160)
candidates = [Path.cwd(), *Path.cwd().parents]
STUDY = next(d for p in candidates for d in [p / 'research/fear-lyrics', p] if (d / 'fear_story.py').exists())
ROOT = STUDY.parents[1] if (STUDY.parents[1] / 'static').exists() else STUDY
RES = STUDY / 'results' / 'story-v3'
FIG = ROOT / 'static' / 'research' / 'fear-lyrics' / 'story'
PLAB = {1980: '1980–89', 1990: '1990–99', 2000: '2000–09', 2010: '2010–19', 2020: '2020–26'}

def fig(name, width=1100):
    display(Image(filename=str(FIG / f'{name}.png'), width=width))

def pct(df, cols):
    return df.style.format({c: '{:.0%}' for c in cols})

table = pd.read_csv(RES / 'song_table.csv', index_col=0)
readings = pd.read_csv(RES / 'readings_public.csv')
trends = pd.read_csv(RES / 'trends.csv')
means = pd.read_csv(RES / 'period_means.csv')
fear = table[table.fear_weight > 0]
print(f'{len(table)} canciones · {len(readings)} lecturas · {len(fear)} con miedo explícito o implícito')"""),
    md("""## 1. El corpus y la selección

Cinco períodos por año de primera edición de la grabación; 12 canciones en inglés y 12 en español en cada uno; 2020–26 es parcial. La selección es **intencional**: el analista eligió candidatas asociadas al miedo, casos ambiguos y controles antes de ver resultados. Por eso el cuaderno compara **la composición dentro de las canciones con miedo**, no cuántas canciones con miedo hay por década: esa cifra depende de quién eligió la lista."""),
    code("""display(Markdown('**Etiqueta de selección del analista frente a la lectura anonimizada** (`agreed`: ambas lecturas ven miedo; `split`: solo una)'))
display(pd.crosstab(table.selection_label, table.fear_status, margins=True))
display(Markdown('**Peso de canciones con miedo por período e idioma** (1 = ambas lecturas; 0,5 = una)'))
display(fear.groupby(['period', 'language']).fear_weight.sum().unstack().rename(index=PLAB))"""),
    md("""**Lectura.** De las 25 canciones elegidas como «miedo», 24 lo tienen según los lectores; 30 de 66 ambiguas y 2 de 29 controles también. Así, 32 de las 56 canciones con miedo no fueron elegidas por eso: la sección 9 repite los análisis solo con ellas."""),
    md("""## 2. La lectura anonimizada, su acuerdo y la prueba de reconocimiento

Protocolo: `story-reading-protocol.md`. Los bloques llegan en orden, con repeticiones marcadas; las salidas públicas solo contienen categorías, IDs y paráfrasis breves en español. El acuerdo es alto, pero ambos lectores son el mismo modelo: un κ alto mide consistencia del procedimiento, no independencia humana."""),
    code("""agreement = pd.read_csv(RES / 'reader_agreement.csv')
display(agreement.style.format({'agreement': '{:.0%}', 'kappa': '{:.2f}'}))
display(Markdown('**Prueba de reconocimiento**: el mismo modelo intenta identificar cada letra anonimizada (proporción de títulos correctos)'))
display(pd.read_csv(RES / 'recognition_summary.csv').style.format({c: '{:.0%}' for c in ['title_correct', 'claimed', 'en', 'es', 'p1980', 'p1990', 'p2000', 'p2010', 'p2020']}))
rec = pd.read_csv(RES / 'recognition_by_song.csv', index_col=0)
display(rec[(rec.fear_weight > 0) & ~rec.title_correct][['title', 'year', 'language', 'claimed']])"""),
    md("""**Lectura.** Anonimizar no ciega a un modelo de lenguaje: identifica 110 de las 120 letras, y todas las canciones con miedo de 1980 a 2019. Las lecturas pueden traer la interpretación cultural aprendida de cada canción. Las cuatro canciones con miedo no reconocidas son de 2020–26 (*Ansiedad*, *Tengo Un Pensamiento*, *Algo tú*, *Die With a Smile*): demasiado pocas para repetir el análisis solo con ellas. La geometría de embeddings, que solo ve texto, es la medición menos expuesta a ese sesgo."""),
    md("""## 3. Dónde está la amenaza

Seis lugares, del afuera al adentro: sociedad/Estado/calle, monstruo o presagio, otra persona, perder un vínculo, tiempo/futuro, la propia mente."""),
    code("""fig('fear-locus-timeline')
fig('fear-locus-periods')"""),
    code("""cols = ['threat_locus:external_collective', 'threat_locus:external_person', 'threat_locus:relational_loss', 'threat_locus:internal_self']
view = means[(means.column.isin(cols)) & (means.language == 'both')].pivot(index='column', columns='period', values='mean')
view.columns = [PLAB[c] for c in view.columns]
display(Markdown('**Proporción por período** (igual peso por idioma)'))
display(view.style.format('{:.0%}'))
display(Markdown('**Tendencia por década** (WLS con efectos de idioma; p por permutación de años dentro de idioma)'))
display(trends[trends.measure.str.startswith('lectura: amenaza') | trends.measure.str.startswith('lectura: pérdida')]
        [['measure', 'slope_per_decade', 'p_two_sided', 'songs']].style.format({'slope_per_decade': '{:+.1%}', 'p_two_sided': '{:.3f}'}))"""),
    md("""**Lectura.** En los ochenta, siete de cada diez canciones con miedo ponen la amenaza afuera: la vigilancia (*Somebody's Watching Me*), la ciudad (*En la ciudad de la furia*), la violencia sectaria (*Sunday Bloody Sunday*), la pobreza (*Livin' on a Prayer*, *Fast Car*) o el monstruo (*Thriller*). En los 2000 domina otra persona, casi siempre la pareja (*Toxic*, *Time Is Running Out*, *Malo*), y aparece la mente como amenaza (*Crawling*, *Bring Me to Life*, *Disturbia*). En 2020–26 la propia mente es el lugar más frecuente (*Ansiedad*, *Anti-Hero*).

La línea no es recta: en 2010–19 la calle vuelve (*Alright*, *This Is America*, *Latinoamérica*), ahora con una voz que enfrenta. Y los idiomas no recorren el mismo camino:"""),
    code("""for col, name in [('threat_locus:internal_self', 'La propia mente'), ('threat_locus:external_person', 'Otra persona'),
                  ('threat_locus:relational_loss', 'Perder un vínculo')]:
    v = means[(means.column == col) & (means.language.isin(['en', 'es']))].pivot(index='language', columns='period', values='mean')
    n = means[(means.column == col) & (means.language.isin(['en', 'es']))].pivot(index='language', columns='period', values='songs')
    v.columns = [PLAB[c] for c in v.columns]
    display(Markdown(f'**{name}**, por idioma (canciones con miedo por celda: {", ".join(f"{x:g}" for x in n.values.ravel())})'))
    display(v.style.format('{:.0%}'))"""),
    md("""**Lectura.** En inglés, la mente aparece en los noventa y ocupa un tercio desde 2000 (grunge, nu-metal, pop confesional). En español no aparece hasta 2020 (*Ansiedad*); antes, el miedo íntimo pasa por la casa y la pareja (*Malo*, *La Puerta Violeta*) y por la pérdida (*A Dios Le Pido*, *Miedo*). Con 2–4 canciones por celda en español, esto es una observación de casos, no una estimación."""),
    md("""## 4. Quién tiene miedo"""),
    code("""who = means[(means.column == 'first_person') & (means.language == 'both')][['period', 'mean', 'lo90', 'hi90', 'songs']]
who['period'] = who.period.map(PLAB)
display(Markdown('**Teme quien canta (yo o nosotros)**, con intervalo bootstrap del 90 %'))
display(who.style.format({'mean': '{:.0%}', 'lo90': '{:.0%}', 'hi90': '{:.0%}'}))
r = readings[readings.fear_reference.isin(['explicit', 'implied'])].assign(period=lambda d: d.year.map(lambda y: PLAB[min(y // 10 * 10, 2020)]))
display(pd.crosstab(r.period, r.who_fears, normalize='index').style.format('{:.0%}'))
display(trends[trends.measure.str.startswith('voz')][['measure', 'slope_per_decade', 'p_two_sided']].style.format({'slope_per_decade': '{:+.1%}', 'p_two_sided': '{:.3f}'}))"""),
    md("""**Lectura.** En los ochenta, el miedo suele pertenecer a otro: el «tú» al que se dirige *Thriller*, los personajes de *Cruz de Navajas* o *Hijo de la Luna*. Desde 2000, ocho de cada diez canciones lo confiesan en primera persona; en 2010–19 aparece el «nosotros» (42 % de las lecturas). Es la tendencia más fuerte del estudio y coincide con trabajos a gran escala que encontraron más pronombres de primera persona en las letras de éxito (DeWall et al., 2011; Parada-Cabaleiro et al., 2024)."""),
    md("""## 5. Los subtemas: de qué hablan cuando hablan de miedo"""),
    code("""fig('fear-subtopics-context', 1000)
cv = pd.read_csv(RES / 'concept_validation.csv')
display(Markdown('**¿Reconoce una pregunta al espacio cada subtema?** AUC de la similitud con un concepto-ancla frente a la etiqueta de los lectores'))
display(cv.pivot(index='concept', columns='encoder', values='auc').join(cv.groupby('concept').songs_with_marker.first())
        .sort_values('e5_large_fear', ascending=False).style.format('{:.2f}', subset=['e5_large_fear', 'e5_large_plain', 'e5_small', 'minilm']))"""),
    md("""**Lectura.** Guerra, policía, calle, pobreza y terror ocupan los ochenta y noventa y desaparecen en 2020–26. La salud mental sube a 31–44 % desde 2000. El amor atraviesa todas las décadas (35–43 %): es el contexto más estable del miedo.

Las preguntas al espacio funcionan con escenas concretas (guerra 0,99 con 4 canciones; control en la pareja 0,82; calle 0,81; monstruos 0,80) y fallan con categorías abstractas (pobreza 0,50, fama 0,53, familia 0,54). Un embedding reconoce mejor un escenario que una condición social."""),
    md("""## 6. Embeddings I: el idioma llega primero"""),
    code("""fig('fear-language-geometry')
display(pd.read_csv(RES / 'language_neighbors.csv').style.format({'same_language_neighbors': '{:.0%}', 'fear_status_neighbor_agreement': '{:.0%}'}))"""),
    md("""**Lectura.** Sin corrección, los cinco vecinos de cada canción comparten idioma en E5 small y E5 large: el mapa separa inglés de español antes que cualquier miedo. Retirar la media de cada idioma (Libovický et al., 2020) devuelve la proporción al azar (51 %). La instrucción de miedo, sola, no lo resuelve, pero combinada con una dirección de idioma estimada con anclas paralelas reduce la señal a 70 % sin borrar diferencias reales de contenido entre corpus."""),
    md("""## 7. Embeddings II: el mapa libre no responde la pregunta"""),
    code("""fig('fear-map-vs-question')
clus = pd.read_csv(RES / 'clustering_vs_readers.csv')
display(Markdown('**Agrupamiento libre de canciones (Ward) frente a las lecturas, el idioma y su estabilidad bootstrap**'))
display(clus.style.format({c: '{:.2f}' for c in ['nmi_reader_locus', 'ari_reader_locus', 'nmi_language', 'bootstrap_ari']}))
display(pd.read_csv(RES / 'knn_zone_check.csv').style.format({'knn10_zone_accuracy': '{:.0%}', 'majority_baseline': '{:.0%}'}))"""),
    md("""**Lectura.** Esta es la razón por la que la versión anterior del estudio se sentía vaga. Sin corrección, el agrupamiento parece estable (ARI bootstrap ≈ 0,6) porque agrupa por idioma (NMI con idioma ≈ 0,6). Sin idioma, la estabilidad cae a ≈ 0,4 y el acuerdo con el lugar de la amenaza queda en NMI ≈ 0,3. Los vecinos más próximos de un pasaje comparten escena, vocabulario o estribillo, no el lugar de la amenaza: kNN acierta menos que asignar siempre «afuera».

BERTopic y UMAP siguen siendo útiles para explorar, pero no para medir un concepto que no les preguntamos."""),
    md("""## 8. Embeddings III: preguntar con ejes

Cada eje es la diferencia entre dos polos de 16 frases ancla (8 EN, 8 ES, traducciones paralelas, todas en primera persona y con miedo explícito, para aislar la dimensión). Archivo: `semantic-anchors.json`. Método: Kozlowski, Taddy y Evans (2019); An, Kwak y Ahn (2018)."""),
    code("""loo = pd.read_csv(RES / 'axis_anchor_loo.csv').pivot(index='encoder', columns='axis', values='loo_accuracy')
display(Markdown('**Validación interna**: cada par de anclas proyectado sobre un eje construido sin él'))
display(loo.style.format('{:.0%}'))
av = pd.read_csv(RES / 'axis_validation.csv')
display(Markdown('**Validación externa**: AUC de cada eje frente a las categorías de los lectores'))
display(av.pivot(index='encoder', columns='axis', values='auc').style.format('{:.2f}'))
display(Markdown('**Intervalos bootstrap del 90 % y diferencias pareadas** (mismas réplicas de canciones para todos los encoders)'))
display(pd.read_csv(RES / 'axis_validation_ci.csv').style.format({'auc': '{:.3f}', 'lo90': '{:.3f}', 'hi90': '{:.3f}', 'p_diff_positive': '{:.2f}'}))"""),
    code("""fig('fear-compass', 1150)
display(trends[trends.measure.str.startswith('eje')][['measure', 'slope_per_decade', 'p_two_sided']].style.format({'slope_per_decade': '{:+.2f}', 'p_two_sided': '{:.3f}'}))"""),
    md("""**Lectura.** El encoder instruido tiene el AUC más alto en los tres ejes (afuera/adentro 0,89 [0,79–0,96]; colectivo 0,85 [0,74–0,93]; huir/enfrentar 0,72 [0,57–0,85]). Supera a MiniLM y a E5 small en la mayoría de las comparaciones pareadas, pero la ganancia de la instrucción sobre el mismo modelo (0,02–0,05) cae dentro del ruido: sus intervalos incluyen cero. Sin mirar etiquetas, el eje afuera→adentro se desplaza +0,23 desviaciones estándar por década. El eje de agencia apenas se mueve (p = 0,21): la geometría capta mejor **dónde** está la amenaza que **cómo** responde la voz."""),
    md("""## 9. Arcos argumentales"""),
    code("""fig('fear-story-strips')
fig('fear-arcs')
display(trends[trends.measure.str.startswith('arco') | trends.measure.str.startswith('lectura: respuesta')]
        [['measure', 'slope_per_decade', 'p_two_sided', 'q_bh']].style.format({'slope_per_decade': '{:+.1%}', 'p_two_sided': '{:.3f}', 'q_bh': '{:.3f}'}))
display(Markdown('**¿Sigue la geometría el movimiento dentro de una canción?** Correlación de Spearman entre el estado de los lectores y el eje de agencia, bloque a bloque'))
display(pd.Series(json.loads((RES / 'arc_embedding_validation.json').read_text())))"""),
    md("""**Lectura.** En los ochenta y noventa, un tercio de las canciones con miedo escala: la amenaza crece hasta el final (*Thriller*, *Somebody's Watching Me*, *Matador*). Desde 2000, entre un tercio y dos quintos dan un **giro a la agencia**: la voz termina enfrentando o decidiendo (*Lose Yourself*, *Malo*, *La Puerta Violeta*, *Ansiedad*). El **bucle** persiste en todas las décadas (~30 %): el estribillo como obsesión que no se resuelve (*Mad World*, *Crawling*, *bury a friend*, *Anti-Hero*). En 2020–26 crece el miedo **episódico**: un instante de vértigo dentro de una canción de amor.

Los embeddings siguen ese movimiento débilmente (ρ medio 0,15 dentro de la canción): el arco necesita lectura, no solo vectores."""),
    md("""## 10. Qué protege la voz"""),
    code("""fig('fear-values')
display(pd.read_csv(RES / 'values_by_period.csv', index_col=0).rename(columns=lambda c: PLAB[int(c)]).style.format('{:.0%}'))"""),
    md("""**Lectura.** «Estar a salvo» (privacidad, refugio, sobrevivir) domina los ochenta y desaparece en 2020–26. «Dignidad, libertad, voz» alcanza 71 % en 2010–19. «Sanar, ser aceptado» solo aparece en la última década (*Ansiedad*, *Anti-Hero*). El vínculo amoroso es lo que más se protege en casi todos los períodos. La codificación es una regla de palabras clave sobre paráfrasis (`VALUES` en `fear_story.py`): describe las lecturas, no las mide de forma independiente."""),
    md("""## 11. Sensibilidad: ¿qué resiste?"""),
    code("""display(Markdown('**Todas las pendientes con su q de Benjamini-Hochberg**'))
display(trends[['measure', 'slope_per_decade', 'p_two_sided', 'q_bh']].sort_values('p_two_sided').style.format({'slope_per_decade': '{:+.3f}', 'p_two_sided': '{:.3f}', 'q_bh': '{:.3f}'}))
sens = pd.read_csv(RES / 'trend_sensitivity.csv')
display(sens.style.format({'slope_per_decade': '{:+.3f}', 'p_two_sided': '{:.3f}'}))"""),
    md("""**Lectura.**

- **Resiste:** con solo las 50 canciones de miedo acordado, todas las tendencias se mantienen o se refuerzan. Los cuatro encoders reproducen el eje afuera→adentro (+0,20 a +0,27 DE/década, p ≤ 0,024).
- **Se debilita:** sin las 24 canciones elegidas como «miedo» quedan 32 canciones. La dirección se conserva (interior +3,5 pts/década; colectivo −10,9; eje +0,20 DE/década), pero la evidencia baja (p = 0,07–0,37). Parte del contraste lo aportó la selección; la triangulación con estudios de miles de letras (más primera persona, más negatividad) sostiene la dirección, no la magnitud.
- **Es frágil:** 2020–26 tiene 6,5 canciones con miedo, y los cortes por idioma tienen 2–4 por celda."""),
    md("""## 12. Conclusiones

1. **El miedo se mudó hacia adentro.** De la calle, el Estado y el monstruo a la pareja y, después, a la propia mente. Lo dicen los lectores y lo dice, por separado, la geometría.
2. **Se dejó de narrar y empezó a confesarse.** El miedo de otro («tú», «ellos») cedió ante el «yo»; en los 2010 el «nosotros» devolvió la calle a la canción, ahora con una voz que enfrenta.
3. **Cambió el desenlace, no el estribillo.** Bajan los miedos que escalan y suben los que se atraviesan; el bucle obsesivo sigue ahí.
4. **Cambió lo que se protege.** De estar a salvo a la dignidad y a sanar; el amor permanece como lo más frágil.
5. **Para quien trabaja con embeddings:** primero hay que quitar el idioma; después, preguntar. Un mapa libre ofrece vecinos; una dirección anclada y validada contesta una pregunta.

## Reproducir

```bash
# con la caché privada (letras, vectores y lecturas)
python embed_instruct.py --device mps   # vectores locales (≈2 min en Apple Silicon)
python fear_story.py                    # agregados públicos en results/story-v3
python visualize_story.py               # figuras en static/research/fear-lyrics/story
python build_story_notebook.py          # este cuaderno
```

Las lecturas se generaron con `story-reading-protocol.md` y `story-review/validate_readings.py`. La auditoría metodológica previa (cuatro encoders sobre 40 canciones, BERTopic, Bayes, RAG y LangGraph) se conserva en `anexo-auditoria-metodologica.ipynb`."""),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-execute', action='store_true')
    args = parser.parse_args()
    nb = nbf.v4.new_notebook(cells=CELLS, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3 (study environment)', 'language': 'python'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'study': {'cutoff': '2026-10-11', 'language': 'es', 'privacy': 'metadata_and_paraphrases_only_no_lyrics',
                  'results': 'results/story-v3', 'execution': 'saved_results_no_remote_calls'}})
    nbf.validate(nb)
    if not args.no_execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
    nbf.write(nb, OUT)
    errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error']
    print(f'{OUT.name}: {len(nb.cells)} celdas, {sum(c.cell_type == "code" for c in nb.cells)} de código, errores: {len(errors)}')


if __name__ == '__main__':
    main()
