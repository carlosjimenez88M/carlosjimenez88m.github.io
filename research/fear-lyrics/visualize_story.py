"""Figuras de la lectura argumental del miedo (results/story-v3 → static/research/fear-lyrics/story).

Solo lee agregados públicos: IDs, categorías, puntuaciones y paráfrasis breves. No necesita letras.
Uso: python visualize_story.py
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

pd.set_option('future.infer_string', False)

STUDY = Path(__file__).resolve().parent
RES = STUDY / 'results' / 'story-v3'
FIG = STUDY.parents[1] / 'static' / 'research' / 'fear-lyrics' / 'story'

# Paleta de referencia validada (dataviz): superficie, tintas y slots categóricos en orden fijo.
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
LANG = {'en': ('Inglés', SLOTS[0]), 'es': ('Español', SLOTS[1])}
PERIODS = [1980, 1990, 2000, 2010, 2020]
PLAB = {1980: '1980–89', 1990: '1990–99', 2000: '2000–09', 2010: '2010–19', 2020: '2020–26'}
# rampa ordinal azul (pasos 250 → 700) para el tiempo histórico
PCOL = {1980: '#86b6ef', 1990: '#5598e7', 2000: '#2a78d6', 2010: '#1c5cab', 2020: '#0d366b'}
# del afuera al adentro: orden fijo de la categoría y su slot (validado en orden adyacente)
LOCI = [('external_collective', 'Sociedad, Estado, calle', SLOTS[1]),
        ('supernatural_fiction', 'Monstruo o presagio', SLOTS[6]),
        ('external_person', 'Otra persona', SLOTS[3]),
        ('relational_loss', 'Perder un vínculo', SLOTS[4]),
        ('existential_time', 'Tiempo, futuro, fracaso', SLOTS[2]),
        ('internal_self', 'La propia mente', SLOTS[0])]
RESP_GROUPS = [('Huir, paralizarse, resignarse', ['flight_avoidance', 'paralysis', 'resignation'], '#e66767'),
               ('Suplicar', ['plea_prayer'], '#f2a7a2'),
               ('Confesar, nombrar', ['confession_naming'], '#c3c2b7'),
               ('Cuidar, proteger', ['care_protection'], '#9ec5f4'),
               ('Enfrentar, transformar', ['defiance_confrontation', 'transformation_agency'], '#2a78d6')]
ARCS = [('escalation', 'Escalada', '#e66767'), ('collapse', 'Derrumbe', '#b3261e'), ('loop', 'Bucle', '#c3c2b7'),
        ('episodic', 'Episodio', '#ebeae4'), ('release', 'Alivio', '#9ec5f4'), ('turn_to_agency', 'Giro a la agencia', '#2a78d6')]
# estados → escala amenaza (−2) … control (+2); color divergente rojo ↔ gris ↔ azul
STATE_SCORE = {'threat': -2, 'fear': -2, 'avoidance': -1.5, 'plea': -1, 'loss': -1, 'confession': -0.5,
               'reflection': 0, 'desire': 0.5, 'care': 1, 'celebration': 1, 'defiance': 1.5, 'agency': 2}
DIVERGING = ['#b3261e', '#e66767', '#f2a7a2', '#f0efec', '#b7d3f6', '#5598e7', '#184f95']


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'grid.color': GRID,
        'grid.linewidth': .8, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlepad': 8, 'legend.frameon': False, 'svg.fonttype': 'none'})


OUT_LANG = 'es'
# Traducción de las figuras al inglés: coincidencia exacta (ignorando saltos de línea) y patrones para textos con cifras.
EN = {
    'Del afuera al adentro: dónde sitúa cada canción la amenaza': 'From outside to inside: where each song places the threat',
    'Año de primera edición de la grabación. Corpus: 12 EN + 12 ES por período, selección intencional; 2020–26 parcial. Las filas no miden intensidad.':
        "Year of the recording's first release. Corpus: 12 EN + 12 ES per period, purposive selection; 2020–26 partial. Rows do not measure intensity.",
    'Sociedad, Estado, calle': 'Society, state, street', 'Monstruo o presagio': 'Monster or omen', 'Otra persona': 'Another person',
    'Perder un vínculo': 'Losing a bond', 'Tiempo, futuro, fracaso': 'Time, future, failure', 'La propia mente': "One's own mind",
    'afuera': 'outside', 'adentro': 'inside', 'Ambas lecturas coinciden': 'Both readings agree',
    'Una sola lectura ve miedo, o difiere el lugar': 'Only one reading sees fear, or they disagree on where',
    'La amenaza se acerca: de la calle a la pareja, y de la pareja a la propia mente':
        "The threat moves closer: from the street to the partner, and from the partner to one's own mind",
    'Composición de las canciones con miedo en cada período. Igual peso por idioma; votos fraccionales de dos lecturas.':
        'Composition of the fear songs in each period. Equal weight per language; fractional votes from two readings.',
    'Una canción con dos lecturas distintas aporta mitad y mitad. «Canciones»: suma de presencia de miedo (1 = ambas lecturas lo ven; 0,5 = una).':
        'A song with two different readings counts half and half. "Songs": sum of fear presence (1 = both readings see it; 0.5 = one).',
    '¿Dónde está la amenaza?': 'Where is the threat?', '¿Cómo responde la voz?': 'How does the voice respond?',
    'Huir, paralizarse, resignarse': 'Flee, freeze, resign', 'Suplicar': 'Plead', 'Confesar, nombrar': 'Confess, name it',
    'Cuidar, proteger': 'Care, protect', 'Enfrentar, transformar': 'Confront, transform',
    'La brújula del miedo, medida por los embeddings': 'The compass of fear, measured by embeddings',
    'Cada letra proyectada sobre dos ejes construidos con frases ancla bilingües. Los ejes no ven las etiquetas de los lectores; el color sí.':
        "Each lyric projected onto two axes built from bilingual anchor sentences. The axes never see the readers' labels; the colors do.",
    'E5 large instruct con instrucción de miedo; dirección de idioma retirada. Horizontal: pasajes de miedo; vertical: toda la letra. Unidades: desviaciones estándar sobre los 1.137 bloques.':
        'E5 large instruct with a fear instruction; language direction removed. Horizontal: fear passages; vertical: whole lyric. Units: standard deviations over the 1,137 blocks.',
    'la amenaza está afuera   ·   ·   ·   la amenaza está adentro': 'the threat is outside   ·   ·   ·   the threat is inside',
    'huir, suplicar, rendirse   ·   ·   ·   enfrentar, decidir': 'flee, plead, give up   ·   ·   ·   confront, decide',
    'Cada canción con miedo; color = lugar de la amenaza según los lectores': 'Each fear song; color = where readers place the threat',
    'Media por período: ¿afuera o adentro?': 'Mean by period: outside or inside?',
    'Media por período: ¿huir o enfrentar?': 'Mean by period: flee or confront?',
    '−1 afuera': '−1\noutside', '+1 adentro': '+1\ninside', '−1 huir': '−1\nflee', '+1 enfrentar': '+1\nconfront',
    'Media, IC 90 %': 'Mean, 90% CI', 'Inglés': 'English', 'Español': 'Spanish',
    'Lo primero que un embedding ve en una letra es el idioma': 'The first thing an embedding sees in a lyric is its language',
    'UMAP de los 1.137 bloques, coloreados por idioma. Retirar la media de cada idioma libera la geometría para el contenido.':
        "UMAP of the 1,137 blocks, colored by language. Removing each language's mean frees the geometry for content.",
    'UMAP solo para inspección; las medidas se calculan en el espacio original. Vecinos: centroides de canción, coseno, k = 5.':
        'UMAP for inspection only; measurements use the original space. Neighbors: song centroids, cosine, k = 5.',
    'E5 small, sin corregir': 'E5 small, uncorrected', 'E5 large + instrucción': 'E5 large + instruction',
    'E5 large + instrucción, sin la media de cada idioma': 'E5 large + instruction,\nlanguage means removed',
    '− dirección de idioma': '− language direction', '− media de cada idioma': '− language means',
    'Vecinos del mismo idioma': 'Same-language neighbors', 'azar': 'chance',
    'El mapa libre no responde la pregunta; un eje sí': 'The free map does not answer the question; an axis does',
    'Los mismos pasajes de miedo, sin la señal de idioma. Izquierda: UMAP, vecindarios sin pregunta. Derecha: proyección sobre el eje afuera–adentro.':
        'The same fear passages, without the language signal. Left: UMAP, neighborhoods with no question. Right: projection onto the outside–inside axis.',
    'Cada canción sobre el eje «¿dónde está la amenaza?»': 'Each song on the "where is the threat?" axis',
    'posición en el eje (desviaciones estándar)   afuera  ·  ·  ·  adentro': 'position on the axis (standard deviations)   outside  ·  ·  ·  inside',
    'Afuera: sociedad, monstruo, otra persona': 'Outside: society, monster, another person',
    'Entre: un vínculo, el tiempo': 'In between: a bond, time', 'Adentro: la propia mente': "Inside: one's own mind",
    'Afuera': 'Outside', 'Entre': 'In between', 'Adentro': 'Inside',
    'El arco argumental: del miedo que crece al miedo que se atraviesa': 'The narrative arc: from fear that grows to fear that is worked through',
    'Arriba: cómo se mueve cada forma, con los estados por bloque de los lectores (amenaza −2 … control +2) en diez puntos de la letra.':
        "Top: how each shape moves, using the readers' block-by-block states (threat −2 … control +2) at ten points of the lyric.",
    'Banda: rango intercuartílico de las lecturas; línea: mediana. Abajo: igual peso por idioma, votos fraccionales.':
        'Band: interquartile range of readings; line: median. Bottom: equal weight per language, fractional votes.',
    'Escalada': 'Escalation', 'Derrumbe': 'Collapse', 'Bucle': 'Loop', 'Episodio': 'Episode', 'Alivio': 'Relief',
    'Giro a la agencia': 'Turn to agency', 'inicio': 'start', 'final': 'end', 'amenaza': 'threat', 'neutro': 'neutral',
    'control': 'control', 'Forma del arco entre las canciones con miedo, por período': 'Arc shape among fear songs, by period',
    'Cómo se mueve una canción: el estado de la voz, bloque a bloque': 'How a song moves: the state of the voice, block by block',
    'Cada fila son dos tiras, una por lectura anonimizada, en el orden de la letra (estribillos repetidos incluidos). Derecha: forma del arco.':
        'Each row holds two strips, one per anonymized reading, in lyric order (repeated choruses included). Right: arc shape.',
    'Estados de los lectores sin textos de la letra. El ancho de cada bloque es proporcional a su posición, no a su duración en el audio.':
        "Readers' states, without lyric text. Each block's width reflects its position, not its duration in the audio.",
    'amenaza, miedo': 'threat, fear', 'evitación': 'avoidance', 'súplica, pérdida, confesión': 'plea, loss, confession',
    'reflexión': 'reflection', 'deseo, cuidado': 'desire, care', 'desafío': 'defiance', 'agencia': 'agency',
    'inicio de la letra': 'start of lyric', 'mitad': 'middle',
    'Los subtemas del miedo: de la guerra y la calle a la salud mental': "Fear's subtopics: from war and the street to mental health",
    'Porcentaje de canciones con miedo cuyo contexto incluye el subtema (lectores; multietiqueta). Filas agrupadas: afuera, cerca, adentro.':
        'Share of fear songs whose context includes the subtopic (readers; multi-label). Rows grouped: outside, near, inside.',
    'Igual peso por idioma; votos fraccionales. AUC: similitud de los pasajes de miedo con un concepto-ancla, frente a la etiqueta de los lectores; en azul, AUC ≥ 0,7.':
        "Equal weight per language; fractional votes. AUC: similarity of fear passages to an anchor concept, against the readers' label; blue = AUC ≥ 0.7.",
    'Guerra y violencia política': 'War and political violence', 'Policía y vigilancia': 'Police and surveillance',
    'Calle y crimen': 'Street and crime', 'Pobreza y clase': 'Poverty and class', 'Terror y monstruos': 'Horror and monsters',
    'Fe y plegaria': 'Faith and prayer', 'Familia y hogar': 'Family and home', 'Amor y deseo': 'Love and desire',
    'Violencia de género': 'Gender violence', 'Fama y mirada ajena': "Fame and others' gaze", 'Salud mental': 'Mental health',
    '¿Lo reconoce una pregunta al espacio de embeddings?': 'Does a question to the\nembedding space recognize it?',
    '0,5 azar': '0.5 chance',
    'Qué protege la voz cuando tiene miedo': 'What the voice protects when it is afraid',
    'Valor en juego según las paráfrasis de los lectores, codificado con reglas explícitas; una canción puede proteger varias cosas.':
        "Value at stake in the readers' paraphrases, coded with explicit rules; a song can protect several things.",
    'Proporción de canciones con miedo por período, igual peso por idioma. 2020–26 reúne 6,5 canciones: sus barras son frágiles.':
        'Share of fear songs per period, equal weight per language. 2020–26 holds 6.5 songs: its bars are fragile.',
    'Estar a salvo, sobrevivir': 'Staying safe, surviving', 'La vida y la paz de otros': "Others' lives and peace",
    'El vínculo amoroso': 'The love bond', 'El propio yo': 'The self', 'Dignidad, libertad, voz': 'Dignity, freedom, voice',
    'Sanar, ser aceptado': 'Healing, being accepted',
}
EN_PATTERNS = [
    (r'^(\d+) de 120 canciones tienen miedo explícito o implícito\. Fila: lugar de la amenaza según dos lecturas anonimizadas\.$',
     r'\1 of 120 songs carry explicit or implied fear. Row: where the threat sits, according to two anonymized readings.'),
    (r'(\d+(?:,\d+)?) canciones', r'\1 songs'),
    (r'^(\d+%) de los vecinos\ncomparten idioma$', r'\1 of neighbors\nshare the language'),
    (r'^kNN \(k = 10, validación agrupada por canción\) acierta la zona en (\d+%) de los pasajes, por debajo de asignar siempre '
     r'«afuera» \((\d+%)\)\. El eje separa canciones «adentro» de «afuera» con AUC (\d,\d+)\.$',
     r'kNN (k = 10, song-grouped validation) gets the zone right for \1 of passages, below always guessing "outside" (\2). '
     r'The axis separates "inside" from "outside" songs with AUC \3.'),
    (r'^UMAP de (\d+) pasajes de miedo \(color: lectores\)$', r'UMAP of \1 fear passages (color: readers)'),
    (r'^mediana ', 'median '),
    (r'^(\d+) %$', r'\1%'),
    (r'\n(\d+) lecturas$', r'\n\1 readings'),
    (r'^(.*) \((\d+)\)$', None),  # etiqueta de leyenda con recuento: se traduce la parte de texto
]


def tr(text: str) -> str:
    """Traduce un texto de figura al inglés; conserva lo que no reconoce (títulos de canciones, años, cifras)."""
    if OUT_LANG != 'en' or not text:
        return text
    flat = re.sub(r'\s*\n\s*', ' ', text)
    if flat in EN:
        out = EN[flat]
        if '\n' in text and '\n' not in out and len(out) > 18 and len(text.split('\n')[0]) <= 20:
            out = '\n'.join(textwrap.wrap(out, 18))
        return out
    if ' / ' in text and all(p in EN for p in text.split(' / ')):
        return ' / '.join(EN[p] for p in text.split(' / '))
    out = text
    for pattern, repl in EN_PATTERNS:
        m = re.match(pattern, out, flags=re.S)
        if repl is None and m and m.group(1) in EN:
            return f'{EN[m.group(1)]} ({m.group(2)})'
        if repl is not None:
            out = re.sub(pattern, repl, out, flags=re.S)
    out = re.sub(r'(\d),(\d)', r'\1.\2', out)
    return out


def translate_figure(fig):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            axis.set_major_formatter(FuncFormatter(lambda x, pos, f=fmt: tr(f(x, pos))))
    for text in fig.findobj(Text):
        text.set_text(tr(text.get_text()))


def save(fig, name: str):
    out = FIG / 'en' if OUT_LANG == 'en' else FIG
    out.mkdir(parents=True, exist_ok=True)
    if OUT_LANG == 'en':
        translate_figure(fig)
    fig.savefig(out / f'{name}.svg', bbox_inches='tight', pad_inches=.15)
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def frame(fig, title: str, subtitle: str, note: str | None = None, top: float = 1.05, bottom: float = .55,
          left: float | None = None, right: float | None = None):
    """Título, subtítulo y nota en pulgadas fijas; los ejes quedan entre `top` y `bottom` (pulgadas)."""
    w, h = fig.get_size_inches()
    fig.text(.12 / w, 1 - .18 / h, title, fontsize=15, weight='bold', color=INK, ha='left', va='top')
    fig.text(.12 / w, 1 - .5 / h, subtitle, fontsize=10, color=INK2, ha='left', va='top')
    if note:
        fig.text(.12 / w, .1 / h, note, fontsize=8.6, color=MUTED, ha='left', va='bottom')
    kw = dict(top=1 - top / h, bottom=bottom / h)
    if left is not None:
        kw['left'] = left / w
    if right is not None:
        kw['right'] = 1 - right / w
    fig.subplots_adjust(**kw)


def place_labels(ax, xs, ys, texts, fontsize=8.8, color=INK2, offsets=None):
    """Etiquetado voraz: prueba posiciones alrededor del punto y descarta las que chocan con otras."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offsets = offsets or [(5, 3), (5, -9), (-5, 3), (-5, -9), (0, 7), (0, -12), (9, 9), (-9, -14), (12, -3), (-12, -3)]
    taken = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    boxes = [matplotlib.transforms.Bbox.from_bounds(px - 4, py - 4, 8, 8) for px, py in taken]
    for x, y, t in sorted(zip(xs, ys, texts), key=lambda z: -z[1]):
        for dx, dy in offsets:
            ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
            ann = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=fontsize,
                              color=color, ha=ha, va='bottom', zorder=6)
            bb = ann.get_window_extent(renderer).expanded(1.04, 1.1)
            axbb = ax.get_window_extent(renderer)
            inside = axbb.x0 - 40 < bb.x0 and bb.x1 < axbb.x1 + 40
            if inside and not any(bb.overlaps(b) for b in boxes):
                boxes.append(bb)
                break
            ann.remove()


def load():
    table = pd.read_csv(RES / 'song_table.csv', index_col=0)
    readings = pd.read_csv(RES / 'readings_public.csv')
    return table, readings


def balanced_shares(df: pd.DataFrame, cols: list[str], normalize: bool = True) -> pd.DataFrame:
    """Composición por período con igual peso por idioma; dentro de idioma, peso = presencia de miedo."""
    rows = []
    for p in PERIODS:
        parts = []
        for _, s in df[df.period == p].groupby('language'):
            w = s.fear_weight.values
            if w.sum() == 0:
                continue
            m = s[cols].div(s.fear_weight, axis=0).fillna(0)
            parts.append((m.values * w[:, None]).sum(0) / w.sum())
        v = np.mean(parts, axis=0)
        v = v / v.sum() if normalize else v
        rows.append(dict(period=p, **{c: x for c, x in zip(cols, v)}, songs=float(df[df.period == p].fear_weight.sum())))
    return pd.DataFrame(rows)


def stacked_rows(ax, data: pd.DataFrame, spec, min_label=.09, fontsize=8.5):
    for i, row in data.reset_index(drop=True).iterrows():
        left = 0
        for col, _, color in spec:
            v = row[col]
            if v <= 0:
                continue
            ax.barh(i, v - .004, left=left + .002, height=.62, color=color, edgecolor='none')
            if v >= min_label:
                r, g, b = matplotlib.colors.to_rgb(color)
                ink = INK if (.299 * r + .587 * g + .114 * b) > .55 else 'white'
                ax.text(left + v / 2, i, f'{v:.0%}', ha='center', va='center', fontsize=fontsize, color=ink)
            left += v
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '25 %', '50 %', '75 %', '100 %'], fontsize=8.5)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)


# ---------------------------------------------------------------- 1. línea de tiempo del lugar de la amenaza

def fig_locus_timeline(table: pd.DataFrame):
    fear = table[table.fear_weight > 0].copy()
    loc_cols = [f'threat_locus:{k}' for k, _, _ in LOCI]
    fear['main_locus'] = fear[loc_cols].idxmax(axis=1).str.split(':').str[1]
    fear = fear[fear[loc_cols].max(axis=1) > 0]
    step, char_years = .28, .245  # separación entre carriles; ancho aproximado de un carácter en años
    order = [0, -1, 1, -2, 2, -3, 3, -4, 4]
    placed, centers, y0 = [], {}, 0.0
    for k, label, color in LOCI:
        sub = fear[fear.main_locus == k].sort_values('year')
        lane_end, rows = {}, []
        for _, r in sub.iterrows():
            title = r.title if len(r.title) <= 24 else r.title[:22] + '…'
            lane = next(l for l in order if r.year - .4 > lane_end.get(l, -1e9))
            lane_end[lane] = r.year + .9 + len(title) * char_years
            rows.append((r, title, lane))
        lo, hi = min([l for *_, l in rows] + [0]), max([l for *_, l in rows] + [0])
        center = y0 - lo * step
        centers[k] = center
        placed += [(r, t, center + l * step, k, color) for r, t, l in rows]
        y0 = center + hi * step + .62
    height = y0
    fig, ax = plt.subplots(figsize=(13, 2.3 + height * 1.25))
    frame(fig, 'Del afuera al adentro: dónde sitúa cada canción la amenaza',
          f'{len(fear)} de 120 canciones tienen miedo explícito o implícito. Fila: lugar de la amenaza según dos lecturas anonimizadas.',
          'Año de primera edición de la grabación. Corpus: 12 EN + 12 ES por período, selección intencional; 2020–26 parcial. '
          'Las filas no miden intensidad.', top=1.0, bottom=.95, left=2.0, right=.2)
    for p in PERIODS[1:]:
        ax.axvline(p - .5, color=GRID, lw=1, zorder=0)
    for k, label, color in LOCI:
        ax.axhline(centers[k], color=GRID, lw=.8, zorder=0)
        ax.text(1977.6, centers[k], label, ha='right', va='center', fontsize=10, color=INK)
        ax.scatter(1978.1, centers[k], s=55, color=color, clip_on=False, zorder=3)
    for r, title, y, k, color in placed:
        agreed = r.fear_weight == 1 and r[f'threat_locus:{k}'] == 1
        ax.scatter(r.year, y, s=58, color=color if agreed else SURFACE, edgecolor=color if not agreed else SURFACE,
                   linewidth=1.8 if not agreed else 1.5, zorder=3)
        ax.text(r.year + .55, y, title, fontsize=9, color=INK2, va='center', ha='left', zorder=4)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlim(1979.3, 2030.5); ax.set_ylim(height - .3, -.45)
    ax.set_xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2026])
    ax.tick_params(axis='x', labelsize=9)
    ax.text(1977.6, -.4, 'afuera', ha='right', fontsize=9, color=MUTED, style='italic')
    ax.text(1977.6, height - .35, 'adentro', ha='right', fontsize=9, color=MUTED, style='italic')
    legend = [Line2D([], [], marker='o', ls='', color=INK2, markersize=7.5, label='Ambas lecturas coinciden'),
              Line2D([], [], marker='o', ls='', markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.6,
                     markersize=7.5, label='Una sola lectura ve miedo, o difiere el lugar')]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(2.0 / 13, .38 / fig.get_figheight()), ncol=2, fontsize=9)
    save(fig, 'fear-locus-timeline')


# ---------------------------------------------------------------- 2. composición por período

def fig_locus_periods(table: pd.DataFrame):
    fear = table[table.fear_weight > 0].copy()
    loc = balanced_shares(fear, [f'threat_locus:{k}' for k, _, _ in LOCI])
    for name, members, _ in RESP_GROUPS:
        fear[name] = fear[[f'response:{m}' for m in members]].sum(axis=1)
    resp = balanced_shares(fear, [g for g, _, _ in RESP_GROUPS])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, gridspec_kw=dict(wspace=.07))
    frame(fig, 'La amenaza se acerca: de la calle a la pareja, y de la pareja a la propia mente',
          'Composición de las canciones con miedo en cada período. Igual peso por idioma; votos fraccionales de dos lecturas.',
          'Una canción con dos lecturas distintas aporta mitad y mitad. «Canciones»: suma de presencia de miedo '
          '(1 = ambas lecturas lo ven; 0,5 = una).', top=1.25, bottom=1.45, left=1.35, right=.2)
    for ax, data, spec, title in [
            (axes[0], loc, [(f'threat_locus:{k}', l, c) for k, l, c in LOCI], '¿Dónde está la amenaza?'),
            (axes[1], resp, [(g, g, c) for g, _, c in RESP_GROUPS], '¿Cómo responde la voz?')]:
        stacked_rows(ax, data, spec)
        ax.set_title(title)
        ax.legend(handles=[Patch(color=c, label=l) for _, l, c in spec], loc='upper left', bbox_to_anchor=(0, -.1),
                  ncol=3 if len(spec) > 5 else 2, fontsize=8.5, handlelength=1, columnspacing=1.1)
    axes[0].set_yticks(range(len(PERIODS)))
    axes[0].set_yticklabels([f'{PLAB[p]}\n{n:g} canciones'.replace('.', ',') for p, n in zip(PERIODS, loc.songs)], fontsize=9)
    axes[0].invert_yaxis()
    save(fig, 'fear-locus-periods')


# ---------------------------------------------------------------- 3. brújula (geometría)

ZONE = [('afuera', ['external_collective', 'supernatural_fiction', 'external_person'], SLOTS[1], 'Afuera: sociedad, monstruo, otra persona'),
        ('entre', ['relational_loss', 'existential_time'], SLOTS[2], 'Entre: un vínculo, el tiempo'),
        ('adentro', ['internal_self'], SLOTS[0], 'Adentro: la propia mente')]
# solo canciones en zonas poco densas: en el centro, una etiqueta podría leerse junto al punto equivocado
COMPASS_LABELS = ['Thriller', "Somebody's Watching Me", 'Sunday Bloody Sunday', 'Matador', 'Hurt', 'Zombie',
                  'Enter Sandman', 'Crawling', 'Bring Me to Life', 'Disturbia', 'Color Esperanza', 'La Puerta Violeta',
                  'This Is America', 'bury a friend', 'Ansiedad', 'Anti-Hero', 'Mad World', 'Lose Yourself', 'Your Power']


def zone_of(row) -> str:
    votes = {z: sum(row.get(f'threat_locus:{k}', 0) for k in ks) for z, ks, _, _ in ZONE}
    return max(votes, key=votes.get) if max(votes.values()) > 0 else 'entre'


def dot_whisker(ax, means: pd.DataFrame, column: str, title: str, left_label: str, right_label: str):
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    for i, p in enumerate(PERIODS):
        m = means[(means.column == column) & (means.language == 'both') & (means.period == p)].iloc[0]
        ax.plot([m.lo90, m.hi90], [i, i], color=PCOL[p], lw=2, solid_capstyle='round', zorder=2)
        ax.scatter(m['mean'], i, s=80, color=PCOL[p], edgecolor=SURFACE, linewidth=2, zorder=4)
        for lang, marker in [('en', 'o'), ('es', 's')]:
            ml = means[(means.column == column) & (means.language == lang) & (means.period == p)]
            if not ml.empty:
                ax.scatter(ml['mean'].iloc[0], i - .28, s=18, marker=marker, color=SURFACE, edgecolor=INK2,
                           linewidth=1, zorder=3)
    ax.set_yticks(range(5)); ax.set_yticklabels([PLAB[p] for p in PERIODS], fontsize=9)
    ax.set_ylim(4.6, -.6); ax.set_xlim(-1.35, 1.35)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([-1, 0, 1]); ax.set_xticklabels([f'−1\n{left_label}', '0', f'+1\n{right_label}'], fontsize=8.5)


def fig_compass(table: pd.DataFrame, means: pd.DataFrame):
    fear = table[table.fear_weight > 0].copy()
    fear['zone'] = fear.apply(zone_of, axis=1)
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=.75, wspace=.12)
    frame(fig, 'La brújula del miedo, medida por los embeddings',
          'Cada letra proyectada sobre dos ejes construidos con frases ancla bilingües. Los ejes no ven las etiquetas de los lectores; el color sí.',
          'E5 large instruct con instrucción de miedo; dirección de idioma retirada. Horizontal: pasajes de miedo; vertical: toda la letra. '
          'Unidades: desviaciones estándar sobre los 1.137 bloques.', top=1.2, bottom=1.6, left=.85, right=.25)
    ax = fig.add_subplot(gs[:, 0])
    ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
    for z, _, color, label in ZONE:
        s = fear[fear.zone == z]
        ax.scatter(s.interiorizacion_fear, s.agencia_all, s=48, color=color, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=f'{label} ({len(s)})')
    lab = fear[fear.title.isin(COMPASS_LABELS)]
    ax.set_xlim(fear.interiorizacion_fear.min() - .35, fear.interiorizacion_fear.max() + .75)
    place_labels(ax, lab.interiorizacion_fear.values, lab.agencia_all.values,
                 [f'{t} ’{str(y)[2:]}' for t, y in zip(lab.title, lab.year)])
    ax.set_xlabel('la amenaza está afuera   ·   ·   ·   la amenaza está adentro', fontsize=9.5)
    ax.set_ylabel('huir, suplicar, rendirse   ·   ·   ·   enfrentar, decidir', fontsize=9.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.09), ncol=2, fontsize=8.5, handletextpad=.3)
    ax.set_title('Cada canción con miedo; color = lugar de la amenaza según los lectores', fontsize=10)
    dot_whisker(fig.add_subplot(gs[0, 1]), means, 'interiorizacion_fear', 'Media por período: ¿afuera o adentro?', 'afuera', 'adentro')
    ax2 = fig.add_subplot(gs[1, 1])
    dot_whisker(ax2, means, 'agencia_all', 'Media por período: ¿huir o enfrentar?', 'huir', 'enfrentar')
    ax2.legend(handles=[Line2D([], [], marker='o', ls='', color=PCOL[2000], markersize=7, label='Media, IC 90 %'),
                        Line2D([], [], marker='o', ls='', markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=5, label='Inglés'),
                        Line2D([], [], marker='s', ls='', markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=5, label='Español')],
               loc='upper left', bbox_to_anchor=(0, -.42), ncol=3, fontsize=8.5, handletextpad=.3)
    save(fig, 'fear-compass')


# ---------------------------------------------------------------- 4. el idioma en la geometría

def fig_language(maps: pd.DataFrame, neighbors: pd.DataFrame):
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=.42)
    frame(fig, 'Lo primero que un embedding ve en una letra es el idioma',
          'UMAP de los 1.137 bloques, coloreados por idioma. Retirar la media de cada idioma libera la geometría para el contenido.',
          'UMAP solo para inspección; las medidas se calculan en el espacio original. Vecinos: centroides de canción, coseno, k = 5.',
          top=1.2, bottom=1.3, left=.2, right=.3)
    panels = [('e5_small', 'raw', 'E5 small, sin corregir'), ('e5_large_fear', 'raw', 'E5 large + instrucción'),
              ('e5_large_fear', 'centered', 'E5 large + instrucción,\nsin la media de cada idioma')]
    for i, (enc, var, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        m = maps[(maps.encoder == enc) & (maps.variant == var)]
        for lang in ['en', 'es']:
            s = m[m.language == lang]
            ax.scatter(s.x, s.y, s=6, color=LANG[lang][1], alpha=.55, edgecolor='none', label=LANG[lang][0])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        purity = neighbors[(neighbors.encoder == enc) & (neighbors.variant == var)].same_language_neighbors.iloc[0]
        ax.set_title(title, fontsize=10)
        ax.text(0, -.03, f'{purity:.0%} de los vecinos\ncomparten idioma', transform=ax.transAxes, fontsize=9, color=INK, va='top')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=LANG[l][1], markersize=6, label=LANG[l][0]) for l in LANG],
               loc='lower left', bbox_to_anchor=(.2 / 13.5, .38 / 5.6), ncol=2, fontsize=9)
    ax = fig.add_subplot(gs[0, 3])
    order = [('minilm', 'raw', 'MiniLM'), ('e5_small', 'raw', 'E5 small'), ('e5_large_plain', 'raw', 'E5 large'),
             ('e5_large_fear', 'raw', 'E5 large + instrucción'), ('e5_large_fear', 'anchor_projected', '− dirección de idioma'),
             ('e5_large_fear', 'centered', '− media de cada idioma')]
    vals = [neighbors[(neighbors.encoder == e) & (neighbors.variant == v)].same_language_neighbors.iloc[0] for e, v, _ in order]
    y = np.arange(len(order))
    ax.barh(y, vals, height=.55, color=SLOTS[0])
    ax.axvline(.5, color=INK2, lw=1)
    ax.text(.52, -.62, 'azar', fontsize=8.5, color=INK2, va='bottom')
    for yi, v in zip(y, vals):
        ax.text(v + .02, yi, f'{v:.0%}', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for _, _, l in order], fontsize=8.5)
    ax.set_ylim(len(order) - .5, -.75); ax.set_xlim(0, 1.18); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['0', '50 %', '100 %'], fontsize=8.5)
    ax.set_title('Vecinos del mismo idioma', fontsize=10)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    save(fig, 'fear-language-geometry')


# ---------------------------------------------------------------- 4b. mapa libre frente a pregunta

def fig_map_vs_question(maps: pd.DataFrame, table: pd.DataFrame, knn: pd.DataFrame, axis_auc: float):
    m = maps[maps.variant == 'fear_blocks_centered']
    fear = table[table.fear_weight > 0].copy()
    fear['zone'] = fear.apply(zone_of, axis=1)
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=.22)
    k = knn[(knn.encoder == 'e5_large_fear') & (knn.variant == 'centered')].iloc[0]
    frame(fig, 'El mapa libre no responde la pregunta; un eje sí',
          'Los mismos pasajes de miedo, sin la señal de idioma. Izquierda: UMAP, vecindarios sin pregunta. Derecha: proyección sobre el eje afuera–adentro.',
          f'kNN (k = 10, validación agrupada por canción) acierta la zona en {k.knn10_zone_accuracy:.0%} de los pasajes, '
          f'por debajo de asignar siempre «afuera» ({k.majority_baseline:.0%}). El eje separa canciones «adentro» de «afuera» '
          f'con AUC {axis_auc:.2f}'.replace('AUC 0.', 'AUC 0,') + '.',
          top=1.15, bottom=1.0, left=.3, right=.3)
    ax = fig.add_subplot(gs[0, 0])
    for z, _, color, label in ZONE:
        s = m[m.zone == z]
        ax.scatter(s.x, s.y, s=16, color=color, alpha=.75, edgecolor='none', label=label)
    cent = m.groupby('song_id')[['x', 'y']].median().join(table[['title', 'year']])
    show = cent[cent.title.isin(['Thriller', 'Crawling', 'Ansiedad', 'Matador', 'Malo', 'Anti-Hero', 'Sunday Bloody Sunday',
                                 'bury a friend', 'La Puerta Violeta', "Somebody's Watching Me", 'Hurt', 'A Dios Le Pido'])]
    place_labels(ax, show.x.values, show.y.values, [f'{t}' for t in show.title], fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f'UMAP de {len(m)} pasajes de miedo (color: lectores)', fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.02), ncol=3, fontsize=8.5, markerscale=1.6, handletextpad=.2)
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    for i, (z, _, color, label) in enumerate(ZONE):
        s = fear[fear.zone == z]
        y = i + rng.uniform(-.18, .18, len(s))
        ax.scatter(s.interiorizacion_fear, y, s=40, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        med = s.interiorizacion_fear.median()
        ax.plot([med, med], [i - .3, i + .3], color=INK, lw=2, zorder=4)
        ax.text(med, i - .36, f'mediana {med:+.2f}'.replace('.', ','), ha='center', va='bottom', fontsize=8, color=INK2)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_yticks(range(3)); ax.set_yticklabels([l.split(':')[0] for _, _, _, l in ZONE], fontsize=9.5)
    ax.set_ylim(2.6, -.7); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlabel('posición en el eje (desviaciones estándar)   afuera  ·  ·  ·  adentro', fontsize=9)
    ax.set_title('Cada canción sobre el eje «¿dónde está la amenaza?»', fontsize=10)
    save(fig, 'fear-map-vs-question')


# ---------------------------------------------------------------- 5. arcos

def fig_arcs(table: pd.DataFrame, traj: pd.DataFrame, readings: pd.DataFrame):
    fear = table[table.fear_weight > 0]
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.25], hspace=.85, wspace=.2)
    frame(fig, 'El arco argumental: del miedo que crece al miedo que se atraviesa',
          'Arriba: cómo se mueve cada forma, con los estados por bloque de los lectores (amenaza −2 … control +2) en diez puntos de la letra.',
          'Banda: rango intercuartílico de las lecturas; línea: mediana. Abajo: igual peso por idioma, votos fraccionales.',
          top=1.2, bottom=1.0, left=.95, right=.2)
    tr = traj.merge(readings[['song_id', 'reader', 'arc_shape']], on=['song_id', 'reader'])
    tr = tr[tr.song_id.isin(fear.index)]
    tcols = [f't{i}' for i in range(10)]
    xs = np.linspace(0, 1, 10)
    for i, (arc, name, color) in enumerate(ARCS):
        ax = fig.add_subplot(gs[0, i])
        s = tr[tr.arc_shape == arc][tcols].astype(float)
        line = '#52514e' if color in ('#c3c2b7', '#ebeae4') else color
        if len(s):
            ax.fill_between(xs, s.quantile(.25), s.quantile(.75), color=line, alpha=.14, lw=0)
            ax.plot(xs, s.median(), color=line, lw=2.2)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_ylim(-2.2, 2.2); ax.set_xticks([0, 1]); ax.set_xticklabels(['inicio', 'final'], fontsize=8)
        if i == 0:
            ax.set_yticks([-2, 0, 2]); ax.set_yticklabels(['amenaza', 'neutro', 'control'], fontsize=8)
        else:
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.set_title(f'{name}\n{len(s)} lecturas', fontsize=9.5)
    ax = fig.add_subplot(gs[1, :])
    sh = balanced_shares(fear, [f'arc_shape:{a}' for a, _, _ in ARCS])
    stacked_rows(ax, sh, [(f'arc_shape:{a}', n, c) for a, n, c in ARCS], min_label=.08)
    ax.set_yticks(range(5)); ax.set_yticklabels([PLAB[p] for p in PERIODS], fontsize=9); ax.invert_yaxis()
    ax.legend(handles=[Patch(color=c, label=n) for _, n, c in ARCS], ncol=6, loc='upper left', bbox_to_anchor=(0, -.16), fontsize=8.5)
    ax.set_title('Forma del arco entre las canciones con miedo, por período', fontsize=10)
    save(fig, 'fear-arcs')


# ---------------------------------------------------------------- 6. tiras narrativas

def fig_story_strips(table: pd.DataFrame, states: pd.DataFrame, readings: pd.DataFrame, picks: list[str]):
    cmap = matplotlib.colors.ListedColormap(DIVERGING)
    norm = matplotlib.colors.BoundaryNorm([-2.25, -1.75, -1.25, -.25, .25, .75, 1.25, 2.25], cmap.N)
    n = len(picks)
    names = dict((a, nm) for a, nm, _ in ARCS)
    fig, ax = plt.subplots(figsize=(13, .62 * n + 2.2))
    frame(fig, 'Cómo se mueve una canción: el estado de la voz, bloque a bloque',
          'Cada fila son dos tiras, una por lectura anonimizada, en el orden de la letra (estribillos repetidos incluidos). Derecha: forma del arco.',
          'Estados de los lectores sin textos de la letra. El ancho de cada bloque es proporcional a su posición, no a su duración en el audio.',
          top=1.05, bottom=1.05, left=2.6, right=1.5)
    for row, sid in enumerate(picks):
        s = table.loc[sid]
        st = states[states.song_id == sid]
        for j, rd in enumerate(sorted(st.reader.unique())):
            d = st[st.reader == rd].sort_values('position')
            nb = len(d)
            for _, b in d.iterrows():
                score = STATE_SCORE.get(b.state)
                color = SURFACE if score is None else cmap(norm(score))
                ax.add_patch(plt.Rectangle((b.position / nb + .0015, row + j * .36 - .34), 1 / nb - .003, .32,
                                           facecolor=color, edgecolor=GRID if score is None else 'none', lw=.5))
        arcs = readings[readings.song_id == sid].arc_shape
        arc = names.get(arcs.iloc[0], '—') if arcs.nunique() == 1 else ' / '.join(names.get(a, a) for a in arcs)
        ax.text(-.012, row - .02, f'{s.year} · {s.title}', ha='right', va='center', fontsize=9.5, color=INK)
        ax.text(-.012, row + .3, s.artist, ha='right', va='center', fontsize=8, color=MUTED)
        ax.text(1.012, row, arc, ha='left', va='center', fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(n - .4, -.6)
    ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['inicio de la letra', 'mitad', 'final'], fontsize=8.5)
    ax.set_yticks([])
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_visible(False)
    keys = [('amenaza, miedo', DIVERGING[0]), ('evitación', DIVERGING[1]), ('súplica, pérdida, confesión', DIVERGING[2]),
            ('reflexión', DIVERGING[3]), ('deseo, cuidado', DIVERGING[4]), ('desafío', DIVERGING[5]), ('agencia', DIVERGING[6])]
    fig.legend(handles=[Patch(color=c, label=l) for l, c in keys], ncol=7, loc='lower left',
               bbox_to_anchor=(2.6 / 13, .42 / (.62 * n + 2.2)), fontsize=8.5, handlelength=1.2)
    save(fig, 'fear-story-strips')


# ---------------------------------------------------------------- 7. subtemas y su eco en la geometría

MARKERS = [('war_conflict', 'Guerra y violencia política', 'guerra_politica'),
           ('state_police_surveillance', 'Policía y vigilancia', 'vigilancia'),
           ('urban_violence_crime', 'Calle y crimen', 'calle_crimen'),
           ('economy_class', 'Pobreza y clase', 'pobreza_crisis'),
           ('horror_fiction', 'Terror y monstruos', 'noche_monstruos'),
           ('religion_faith', 'Fe y plegaria', 'fe_dios'),
           ('family_home', 'Familia y hogar', 'familia_hogar'),
           ('romance_sex', 'Amor y deseo', 'deseo_cuerpo'),
           ('gender_violence', 'Violencia de género', 'control_pareja'),
           ('media_technology_fame', 'Fama y mirada ajena', 'fama_pantallas'),
           ('mental_health', 'Salud mental', 'ansiedad_salud_mental')]


def fig_context(markers: pd.DataFrame, concept_auc: pd.DataFrame):
    rows = [m for m in MARKERS if m[0] in markers.index]
    data = markers.loc[[m[0] for m in rows], [str(p) for p in PERIODS]].values
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1], wspace=.08)
    frame(fig, 'Los subtemas del miedo: de la guerra y la calle a la salud mental',
          'Porcentaje de canciones con miedo cuyo contexto incluye el subtema (lectores; multietiqueta). Filas agrupadas: afuera, cerca, adentro.',
          'Igual peso por idioma; votos fraccionales. AUC: similitud de los pasajes de miedo con un concepto-ancla, frente a la etiqueta de los lectores; '
          'en azul, AUC ≥ 0,7.', top=1.35, bottom=.6, left=2.1, right=.2)
    ax = fig.add_subplot(gs[0, 0])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#9ec5f4', '#3987e5', '#184f95', '#0d366b'])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=.6, aspect='auto')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, '·' if v == 0 else f'{v:.0%}', ha='center', va='center', fontsize=9,
                    color='white' if v > .33 else (MUTED if v == 0 else INK))
    ax.set_xticks(range(5)); ax.set_xticklabels([PLAB[p] for p in PERIODS], fontsize=9.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([m[1] for m in rows], fontsize=9.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y in (4.5, 8.5):
        ax.axhline(y, color=SURFACE, lw=4)
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [float(concept_auc.get(m[2], np.nan)) for m in rows]
    y = np.arange(len(rows))
    ax2.barh(y, [v - .5 for v in vals], left=.5, height=.5, color=[SLOTS[0] if v >= .7 else '#c3c2b7' for v in vals])
    for yi, v in zip(y, vals):
        ax2.text(max(v, .5) + .015, yi, f'{v:.2f}'.replace('.', ','), va='center', fontsize=8.5, color=INK)
    ax2.axvline(.5, color=INK2, lw=1)
    ax2.set_xlim(.4, 1.1); ax2.set_ylim(len(rows) - .5, -.5)
    ax2.set_yticks([]); ax2.set_xticks([.5, .75, 1]); ax2.set_xticklabels(['0,5 azar', '0,75', '1'], fontsize=8.5)
    ax2.spines['left'].set_visible(False)
    ax2.set_title('¿Lo reconoce una pregunta\nal espacio de embeddings?', fontsize=10)
    save(fig, 'fear-subtopics-context')


# ---------------------------------------------------------------- 8. qué protege la voz

VALUE_ORDER = ['Estar a salvo, sobrevivir', 'La vida y la paz de otros', 'El vínculo amoroso', 'El propio yo',
               'Dignidad, libertad, voz', 'Sanar, ser aceptado']


def fig_values(values: pd.DataFrame):
    fig, axes = plt.subplots(1, len(VALUE_ORDER), figsize=(13, 4.5), sharey=True, gridspec_kw=dict(wspace=.14))
    frame(fig, 'Qué protege la voz cuando tiene miedo',
          'Valor en juego según las paráfrasis de los lectores, codificado con reglas explícitas; una canción puede proteger varias cosas.',
          'Proporción de canciones con miedo por período, igual peso por idioma. 2020–26 reúne 6,5 canciones: sus barras son frágiles.',
          top=1.45, bottom=.75, left=.6, right=.2)
    for ax, name in zip(axes, VALUE_ORDER):
        v = values.loc[name, [str(p) for p in PERIODS]].values.astype(float)
        ax.bar(range(5), v, width=.62, color=[PCOL[p] for p in PERIODS])
        for i, x in enumerate(v):
            ax.text(i, x + .015, f'{x:.0%}' if x > 0 else '0', ha='center', fontsize=8, color=INK2)
        ax.set_xticks(range(5)); ax.set_xticklabels(['80s', '90s', '00s', '10s', '20s'], fontsize=8.5)
        ax.set_title('\n'.join(textwrap.wrap(name, 18)), fontsize=9.5)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([0, .25, .5, .75]); axes[0].set_yticklabels(['0', '25 %', '50 %', '75 %'], fontsize=8.5)
    save(fig, 'fear-values')


def main():
    global OUT_LANG
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['es', 'en'], default='es')
    OUT_LANG = parser.parse_args().lang
    style()
    table, readings = load()
    means = pd.read_csv(RES / 'period_means.csv')
    fig_locus_timeline(table)
    fig_locus_periods(table)
    fig_compass(table, means)
    fig_language(pd.read_csv(RES / 'umap_maps.csv'), pd.read_csv(RES / 'language_neighbors.csv'))
    av = pd.read_csv(RES / 'axis_validation.csv')
    fig_map_vs_question(pd.read_csv(RES / 'umap_maps.csv'), table, pd.read_csv(RES / 'knn_zone_check.csv'),
                        float(av[(av.encoder == 'e5_large_fear') & (av.axis == 'interiorizacion')].auc.iloc[0]))
    fig_arcs(table, pd.read_csv(RES / 'arc_trajectories_readers.csv'), readings)
    fig_story_strips(table, pd.read_csv(RES / 'block_states.csv'), readings,
                     json.loads((RES / 'story_strip_picks.json').read_text()))
    cv = pd.read_csv(RES / 'concept_validation.csv')
    fig_context(pd.read_csv(RES / 'context_markers_by_period.csv', index_col=0),
                cv[cv.encoder == 'e5_large_fear'].set_index('concept').auc)
    fig_values(pd.read_csv(RES / 'values_by_period.csv', index_col=0))
    print('figuras en', FIG / 'en' if OUT_LANG == 'en' else FIG)


if __name__ == '__main__':
    main()
