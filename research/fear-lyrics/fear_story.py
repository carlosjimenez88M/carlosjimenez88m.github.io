"""Lectura argumental del miedo, 1980–2026: subtemas, ejes, arcos y contexto.

Combina dos mediciones independientes de las mismas 120 canciones:
1. la lectura ciega por pares (story-reading-protocol.md), que decide si hay miedo, dónde está la amenaza,
   cómo responde la voz y qué estado ocupa cada bloque;
2. la geometría de los embeddings, con un encoder instruido local y la señal de idioma retirada.

Lee la caché privada (letras, vectores, lecturas) y escribe solo agregados públicos en results/story-v3:
IDs, categorías, puntuaciones y paráfrasis breves. Nunca escribe letras.
Uso: python fear_story.py
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('future.infer_string', False)  # arrays de texto como object: indexado con numpy
# Accelerate (macOS) emite avisos espurios de matmul en float64; los resultados se comprueban finitos.
warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)

STUDY = Path(__file__).resolve().parent
PRIVATE = STUDY.parents[1] / '.research-cache' / 'fear-lyrics'
REVIEW = PRIVATE / 'story-review'
EMB = PRIVATE / 'embeddings-v3'
OUT = STUDY / 'results' / 'story-v3'
SEED = 20261012

PERIODS = [1980, 1990, 2000, 2010, 2020]
PERIOD_LABELS = {1980: '1980–89', 1990: '1990–99', 2000: '2000–09', 2010: '2010–19', 2020: '2020–26'}
FEAR_REFS = {'explicit', 'implied'}
FEAR_STATES = {'threat', 'fear', 'avoidance', 'plea'}
LOCI = ['external_person', 'external_collective', 'supernatural_fiction', 'relational_loss',
        'internal_self', 'existential_time']
RESPONSES = ['flight_avoidance', 'paralysis', 'plea_prayer', 'resignation', 'confession_naming',
             'care_protection', 'defiance_confrontation', 'transformation_agency']
ARCS = ['loop', 'escalation', 'collapse', 'episodic', 'release', 'turn_to_agency']
WHO = ['speaker', 'speaker_collective', 'addressee', 'third_party', 'nobody', 'unclear']
# Escala ordinal amenaza ↔ control para reconstruir el arco desde los estados por bloque.
STATE_SCORE = {'threat': -2.0, 'fear': -2.0, 'avoidance': -1.5, 'plea': -1.0, 'loss': -1.0,
               'confession': -0.5, 'reflection': 0.0, 'desire': 0.5, 'care': 1.0, 'celebration': 1.0,
               'defiance': 1.5, 'agency': 2.0}
ENCODERS = {
    'minilm': 'MiniLM multilingüe',
    'e5_small': 'E5 small',
    'e5_large_plain': 'E5 large instruct, sin instrucción',
    'e5_large_fear': 'E5 large instruct, instrucción de miedo',
}


# Qué protege la voz: regla transparente sobre las paráfrasis de los lectores (multietiqueta).
VALUES = {
    'Estar a salvo, sobrevivir': r'sobreviv|segur|refugio|privacidad|tranquilidad|calma|ánimo|salir adelante|esperanza|intentar|triunfar',
    'La vida y la paz de otros': r'niñ|madre|paz|inocentes|famili|hij|su gente|comunidad|tierra|memoria|vida de stan|vida del fan',
    'El vínculo amoroso': r'pareja|vínculo|amor|amad|relación|unión|cercanía|compañ|querid|junto|placer|deseo|amig',
    'El propio yo': r'sí mism|uno mismo|autocontrol|propia mente|identidad|sentir|viva|ser visto|despertar|conservarse|compostura|inocencia|imaginación',
    'Dignidad, libertad, voz': r'dign|libertad|libre|voz|justicia|autonom|derecho|integridad|honra|negarse|venderse|cuerpo',
    'Sanar, ser aceptado': r'sanar|respirar|hablar|defectos|pese a|no quedarse sola|limpieza|alivio',
}


# ---------------------------------------------------------------- datos

def period_of(year: int) -> int:
    return min(year // 10 * 10, 2020)


def load_manifest() -> pd.DataFrame:
    rows = json.loads((STUDY / 'corpus-manifest.json').read_text())
    df = pd.DataFrame(rows)[['song_id', 'title', 'artist', 'year', 'language', 'selection_label']]
    df['period'] = df.year.map(period_of)
    return df.set_index('song_id')


def load_inventory() -> pd.DataFrame:
    inv = pd.read_csv(STUDY / 'expanded' / 'results' / 'section_inventory.csv')
    inv['position'] = inv.groupby('song_id').order.rank(method='first').astype(int) - 1
    inv['n_blocks'] = inv.groupby('song_id').order.transform('size')
    inv['rel_pos'] = np.where(inv.n_blocks > 1, inv.position / (inv.n_blocks - 1).clip(lower=1), 0.5)
    return inv


def load_readings(manifest: pd.DataFrame):
    mapping = json.loads((REVIEW / 'mapping.json').read_text())
    songs, blocks = [], []
    for path in sorted(REVIEW.glob('reading-L*.json')):
        reader = path.stem.split('-')[1]
        for r in json.loads(path.read_text()):
            case = mapping['cases'][r['case_id']]
            sid = case['song_id']
            songs.append(dict(song_id=sid, reader=reader, **{k: r[k] for k in [
                'fear_reference', 'fear_polarity', 'who_fears', 'threat_locus', 'threat_locus_secondary',
                'threat_object', 'response', 'arc_shape', 'value_at_stake', 'confidence', 'counterreading']},
                context_markers='|'.join(r['context_markers']), fear_support_ids='|'.join(r['fear_support_ids']),
                arc_steps=json.dumps(r['arc_steps'], ensure_ascii=False)))
            for bid, state in r['block_states'].items():
                b = case['blocks'][bid]
                blocks.append(dict(song_id=sid, reader=reader, block=bid, document_id=b['document_id'],
                                   canonical_id=b['canonical_id'], state=state,
                                   support=bid in r['fear_support_ids']))
    songs = pd.DataFrame(songs)
    blocks = pd.DataFrame(blocks)
    counts = songs.groupby('song_id').reader.nunique()
    assert (counts == 2).all() and len(counts) == len(manifest), 'cada canción necesita dos lecturas'
    songs['slot'] = songs.sort_values('reader').groupby('song_id').cumcount()
    return songs, blocks


def load_embeddings():
    index = json.loads((EMB / 'index.json').read_text())
    docs = index['document_ids']
    old = PRIVATE / 'embeddings'
    out = {}
    for key, folder in [('minilm', 'sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'),
                        ('e5_small', 'intfloat--multilingual-e5-small')]:
        meta = json.loads((old / folder / 'document-index.json').read_text())
        assert meta['document_ids'] == docs
        out[key] = (np.load(old / folder / Path(meta['array_path']).name).astype(np.float64),
                    np.load(EMB / f'{folder}-anchors.npy').astype(np.float64))
    for variant in ['plain', 'fear']:
        out[f'e5_large_{variant}'] = (np.load(EMB / f'e5-large-instruct-{variant}-blocks.npy').astype(np.float64),
                                      np.load(EMB / f'e5-large-instruct-{variant}-anchors.npy').astype(np.float64))
    return docs, index['anchor_ids'], out


# ---------------------------------------------------------------- geometría

def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def center_by_language(x: np.ndarray, lang: np.ndarray) -> np.ndarray:
    """Retira la media de cada idioma (Libovický et al., 2020). Útil para vecindarios y clustering."""
    y = x.copy()
    for value in np.unique(lang):
        y[lang == value] -= x[lang == value].mean(0)
    return normalize(y)


def language_subspace(anchors: np.ndarray, anchor_ids: list[str], k: int = 2) -> np.ndarray:
    """Dirección(es) de idioma estimadas con anclas paralelas EN/ES de igual contenido."""
    pairs = []
    for prefix in sorted({i.rsplit('::', 1)[0] for i in anchor_ids}):
        idx = [j for j, i in enumerate(anchor_ids) if i.startswith(prefix + '::')]
        half = len(idx) // 2
        if prefix.startswith('axis'):
            pairs += [anchors[idx[h + half]] - anchors[idx[h]] for h in range(half)]
        else:
            pairs.append(anchors[idx[1]] - anchors[idx[0]])
    d = np.array(pairs)
    mean = d.mean(0)
    _, _, vt = np.linalg.svd(d - mean, full_matrices=False)
    basis = np.vstack([mean / np.linalg.norm(mean), vt[:k - 1]]) if k > 1 else (mean / np.linalg.norm(mean))[None]
    q, _ = np.linalg.qr(basis.T)
    return q.T


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return normalize(x - (x @ basis.T) @ basis)


def axis_vectors(anchors: np.ndarray, anchor_ids: list[str], spec: dict) -> dict:
    axes = {}
    for name in spec['axes']:
        pos = [j for j, i in enumerate(anchor_ids) if i.startswith(f'axis::{name}::positive')]
        neg = [j for j, i in enumerate(anchor_ids) if i.startswith(f'axis::{name}::negative')]
        v = anchors[pos].mean(0) - anchors[neg].mean(0)
        axes[name] = dict(vector=v / np.linalg.norm(v), pos=pos, neg=neg)
    return axes


def anchor_loo_accuracy(anchors: np.ndarray, axes: dict) -> dict:
    """Validación interna: cada ancla se proyecta sobre un eje construido sin ella (ni su traducción)."""
    out = {}
    for name, ax in axes.items():
        pos, neg = ax['pos'], ax['neg']
        half = len(pos) // 2
        hits = []
        for h in range(half):
            drop_p, drop_n = {pos[h], pos[h + half]}, {neg[h], neg[h + half]}
            v = anchors[[p for p in pos if p not in drop_p]].mean(0) - anchors[[n for n in neg if n not in drop_n]].mean(0)
            mid = (anchors[[p for p in pos if p not in drop_p]].mean(0) + anchors[[n for n in neg if n not in drop_n]].mean(0)) / 2
            hits += [float((anchors[p] - mid) @ v > 0) for p in drop_p] + [float((anchors[n] - mid) @ v < 0) for n in drop_n]
        out[name] = float(np.mean(hits))
    return out


def concept_vectors(anchors: np.ndarray, anchor_ids: list[str], spec: dict) -> dict:
    return {c: normalize(anchors[[j for j, i in enumerate(anchor_ids) if i.startswith(f'concept::{c}::')]].mean(0))
            for c in spec['concepts']}


# ---------------------------------------------------------------- lecturas

def cohen_kappa(a: list, b: list) -> float:
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def song_level(manifest: pd.DataFrame, songs: pd.DataFrame) -> pd.DataFrame:
    songs = songs.copy()
    songs['fear'] = songs.fear_reference.isin(FEAR_REFS).astype(float)
    g = songs.groupby('song_id')
    table = manifest.copy()
    table['fear_weight'] = g.fear.mean()
    table['fear_votes'] = g.fear.sum().astype(int)
    table['fear_agreed'] = table.fear_votes == 2
    table['fear_status'] = table.fear_votes.map({2: 'agreed', 1: 'split', 0: 'absent'})
    for field, cats in [('threat_locus', LOCI), ('response', RESPONSES), ('arc_shape', ARCS), ('who_fears', WHO)]:
        for c in cats:
            # voto fraccional entre las lecturas que reconocen miedo
            votes = songs[songs.fear.eq(1)].assign(v=lambda d: d[field].eq(c).astype(float))
            table[f'{field}:{c}'] = votes.groupby('song_id').v.sum().reindex(table.index).fillna(0) / 2
        agree = g[field].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
        table[f'{field}_agreed'] = agree
    markers = songs.assign(m=songs.context_markers.str.split('|')).explode('m')
    markers = markers[songs.loc[markers.index, 'fear'].eq(1)]
    for m, sub in markers.groupby('m'):
        table[f'context:{m}'] = (sub.groupby('song_id').size() / 2).reindex(table.index).fillna(0)
    return table


def agreement_table(songs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fear = songs.assign(f=songs.fear_reference.isin(FEAR_REFS)).pivot(index='song_id', columns='slot', values='f')
    rows.append(dict(field='miedo presente (explícito/implícito)', songs=len(fear),
                     agreement=float((fear[0] == fear[1]).mean()), kappa=cohen_kappa(fear[0], fear[1])))
    for field in ['fear_reference', 'threat_locus', 'response', 'arc_shape', 'who_fears']:
        both = songs[songs.fear_reference.isin(FEAR_REFS)].groupby('song_id').filter(lambda d: len(d) == 2) \
            if field != 'fear_reference' else songs
        w = both.pivot(index='song_id', columns='slot', values=field)
        rows.append(dict(field=field, songs=len(w), agreement=float((w[0] == w[1]).mean()),
                         kappa=cohen_kappa(w[0], w[1])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- medidas

def balanced_period_mean(df: pd.DataFrame, value: str, weight: str = 'w') -> pd.DataFrame:
    """Media por período con igual peso por idioma; dentro de idioma, peso por canción."""
    rows = []
    for p in PERIODS:
        sub = df[df.period == p]
        per_lang = []
        for lang, s in sub.groupby('language'):
            if s[weight].sum() > 0:
                per_lang.append(np.average(s[value], weights=s[weight]))
        rows.append(dict(period=p, mean=float(np.mean(per_lang)) if per_lang else np.nan,
                         songs=float(sub[weight].sum())))
    return pd.DataFrame(rows)


def bootstrap_period(df: pd.DataFrame, value: str, weight: str = 'w', n: int = 4000, by_language: bool = False):
    rng = np.random.default_rng(SEED)
    rows = []
    groups = [('both', df)] if not by_language else list(df.groupby('language'))
    for lang, data in groups:
        for p in PERIODS:
            sub = data[data.period == p]
            if sub[weight].sum() == 0:
                continue
            parts = {l: s for l, s in sub.groupby('language') if s[weight].sum() > 0}
            point = np.mean([np.average(s[value], weights=s[weight]) for s in parts.values()])
            boots = []
            for _ in range(n):
                vals = []
                for s in parts.values():
                    idx = rng.integers(0, len(s), len(s))
                    w = s[weight].values[idx]
                    if w.sum() > 0:
                        vals.append(np.average(s[value].values[idx], weights=w))
                if vals:
                    boots.append(np.mean(vals))
            lo, hi = np.percentile(boots, [5, 95])
            rows.append(dict(language=lang, period=p, mean=float(point), lo90=float(lo), hi90=float(hi),
                             songs=float(sub[weight].sum()), n=int(len(sub))))
    return pd.DataFrame(rows)


def trend_test(df: pd.DataFrame, value: str, weight: str = 'w', n: int = 10000) -> dict:
    """Pendiente por década (WLS) con permutación de años dentro de idioma."""
    rng = np.random.default_rng(SEED)
    d = df[df[weight] > 0]
    x = d.year.values.astype(float)
    y = d[value].values
    w = d[weight].values
    lang = d.language.values

    def slope(xv):
        # efectos fijos de idioma: se retira la media ponderada de cada idioma
        xs, ys = xv.copy(), y.copy()
        for l in np.unique(lang):
            m = lang == l
            xs[m] -= np.average(xv[m], weights=w[m])
            ys[m] -= np.average(y[m], weights=w[m])
        return float(np.sum(w * xs * ys) / np.sum(w * xs * xs)) * 10

    observed = slope(x)
    perm = []
    for _ in range(n):
        xp = x.copy()
        for l in np.unique(lang):
            m = np.where(lang == l)[0]
            xp[m] = rng.permutation(x[m])
        perm.append(slope(xp))
    perm = np.array(perm)
    p = float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (n + 1))
    return dict(slope_per_decade=observed, p_two_sided=p, songs=int(len(d)), weight=float(w.sum()))


def auc(scores_pos, scores_neg) -> float:
    pos, neg = np.asarray(scores_pos), np.asarray(scores_neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + ties / 2)


# ---------------------------------------------------------------- subtemas

def cluster_fear_blocks(x: np.ndarray, k_range=range(5, 11), seeds=(17, 42, 73)):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    rows, labels = [], {}
    for k in k_range:
        runs = [KMeans(n_clusters=k, n_init=20, random_state=s).fit(x).labels_ for s in seeds]
        ari = np.mean([adjusted_rand_score(runs[i], runs[j]) for i in range(3) for j in range(i + 1, 3)])
        rows.append(dict(k=k, silhouette=float(silhouette_score(x, runs[0], metric='cosine')), seed_ari=float(ari)))
        labels[k] = runs[0]
    return pd.DataFrame(rows), labels


# ---------------------------------------------------------------- principal

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((STUDY / 'semantic-anchors.json').read_text())
    manifest = load_manifest()
    inv = load_inventory()
    songs_r, blocks_r = load_readings(manifest)
    docs, anchor_ids, emb = load_embeddings()
    doc_index = {d: i for i, d in enumerate(docs)}
    doc_song = np.array([d.split('::')[0] for d in docs])
    doc_lang = manifest.loc[doc_song, 'language'].values

    # ---- 1. lecturas por canción
    table = song_level(manifest, songs_r)
    agreement = agreement_table(songs_r)
    agreement.to_csv(OUT / 'reader_agreement.csv', index=False)

    # bloques: peso de miedo = fracción de lecturas que lo ubican en un estado de miedo o lo citan
    br = blocks_r.merge(songs_r[['song_id', 'reader', 'fear_reference']], on=['song_id', 'reader'])
    br['fear_reading'] = br.fear_reference.isin(FEAR_REFS)
    br['fear_block'] = br.fear_reading & (br.state.isin(FEAR_STATES) | br.support)
    canon_weight = br.groupby('canonical_id').fear_block.mean()  # sobre dos lecturas × repeticiones
    fear_docs = [d for d in docs if canon_weight.get(d, 0) > 0]

    # ---- 2. idioma: diagnóstico de vecindarios
    def song_vectors(x):
        sv = pd.DataFrame(x).groupby(doc_song).mean()
        return sv.index.values, normalize(sv.values)

    lang_rows = []
    variants = {}
    for key, (x, anchors) in emb.items():
        basis = language_subspace(anchors, anchor_ids, k=2)
        variants[key] = {'raw': x, 'centered': center_by_language(x, doc_lang),
                         'anchor_projected': project_out(x, basis)}
        for vname, xv in variants[key].items():
            ids, sv = song_vectors(xv)
            sims = sv @ sv.T
            np.fill_diagonal(sims, -np.inf)
            nn = np.argsort(-sims, axis=1)[:, :5]
            langs = manifest.loc[ids, 'language'].values
            purity = float((langs[nn] == langs[:, None]).mean())
            # concordancia de vecinos con la presencia de miedo según los lectores
            fw = table.loc[ids, 'fear_weight'].values
            fear_match = float(np.mean([(fw[nn[i]] >= .5) == (fw[i] >= .5) for i in range(len(ids))]))
            lang_rows.append(dict(encoder=key, variant=vname, same_language_neighbors=purity,
                                  fear_status_neighbor_agreement=fear_match))
    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(OUT / 'language_neighbors.csv', index=False)

    # ---- 3. ejes semánticos y su validación contra la lectura
    axis_rows, loo_rows, song_scores = [], [], {}
    fear_doc_mask = np.array([canon_weight.get(d, 0) > 0 for d in docs])
    nonfiller = br[br.state != 'filler'].canonical_id.unique()
    nonfiller_mask = np.isin(docs, nonfiller)
    for key, (x, anchors) in emb.items():
        basis = language_subspace(anchors, anchor_ids, k=2)
        anchors_p = project_out(anchors, basis)
        axes = axis_vectors(anchors_p, anchor_ids, spec)
        loo = anchor_loo_accuracy(anchors_p, axes)
        loo_rows += [dict(encoder=key, axis=a, loo_accuracy=v) for a, v in loo.items()]
        xp = variants[key]['anchor_projected']
        scores = {}
        for name, ax in axes.items():
            s = xp @ ax['vector']
            # z dentro del corpus completo, para que los ejes sean comparables entre encoders
            s = (s - s[nonfiller_mask].mean()) / s[nonfiller_mask].std()
            scores[name] = s
        per_song = []
        for sid in manifest.index:
            m = doc_song == sid
            fm = m & fear_doc_mask
            wts = np.array([canon_weight.get(d, 0) for d in np.array(docs)[fm]])
            row = dict(song_id=sid)
            for name, s in scores.items():
                row[f'{name}_all'] = float(s[m & nonfiller_mask].mean()) if (m & nonfiller_mask).any() else np.nan
                row[f'{name}_fear'] = float(np.average(s[fm], weights=wts)) if fm.any() else np.nan
            per_song.append(row)
        song_scores[key] = pd.DataFrame(per_song).set_index('song_id')
        # validación: AUC frente a categorías acordadas o mayoritarias de los lectores
        ss = song_scores[key].join(table)
        fear_s = ss[ss.fear_weight > 0]
        internal = fear_s[fear_s['threat_locus:internal_self'] >= .5]
        external = fear_s[(fear_s['threat_locus:external_person'] + fear_s['threat_locus:external_collective']
                           + fear_s['threat_locus:supernatural_fiction']) >= .5]
        agentic = fear_s[(fear_s['response:defiance_confrontation'] + fear_s['response:transformation_agency']) >= .5]
        passive = fear_s[(fear_s['response:flight_avoidance'] + fear_s['response:paralysis'] + fear_s['response:plea_prayer']
                          + fear_s['response:resignation']) >= .5]
        collective = fear_s[fear_s['threat_locus:external_collective'] >= .5]
        individual = fear_s[fear_s['threat_locus:external_collective'] == 0]
        axis_rows += [
            dict(encoder=key, axis='interiorizacion', contrast='internal_self vs externo', auc=auc(
                internal.interiorizacion_fear.dropna(), external.interiorizacion_fear.dropna()),
                 n_pos=len(internal), n_neg=len(external)),
            dict(encoder=key, axis='agencia', contrast='enfrentar/decidir vs huir/someterse', auc=auc(
                agentic.agencia_all.dropna(), passive.agencia_all.dropna()), n_pos=len(agentic), n_neg=len(passive)),
            dict(encoder=key, axis='colectividad', contrast='amenaza colectiva vs otras', auc=auc(
                collective.colectividad_fear.dropna(), individual.colectividad_fear.dropna()),
                 n_pos=len(collective), n_neg=len(individual)),
        ]
    pd.DataFrame(axis_rows).to_csv(OUT / 'axis_validation.csv', index=False)
    pd.DataFrame(loo_rows).to_csv(OUT / 'axis_anchor_loo.csv', index=False)

    main_key = 'e5_large_fear'
    scores = song_scores[main_key].join(table)
    for key, df in song_scores.items():
        df.add_prefix(f'{key}:').to_csv(OUT / f'song_axis_scores_{key}.csv')

    # ---- 4. tendencias: lectura y geometría, con los mismos pesos
    fear_songs = scores[scores.fear_weight > 0].copy()
    fear_songs['w'] = fear_songs.fear_weight
    trend_rows, period_rows = [], []
    measures = {
        'lectura: amenaza interior (internal_self)': 'threat_locus:internal_self',
        'lectura: amenaza colectiva': 'threat_locus:external_collective',
        'lectura: amenaza de otra persona': 'threat_locus:external_person',
        'lectura: ficción / sobrenatural': 'threat_locus:supernatural_fiction',
        'lectura: pérdida de un vínculo': 'threat_locus:relational_loss',
        'lectura: tiempo / incertidumbre': 'threat_locus:existential_time',
        'lectura: respuesta de confesión': 'response:confession_naming',
        'lectura: respuesta de enfrentar/decidir': None,
        'lectura: respuesta de huir/someterse': None,
        'eje: interiorización (bloques de miedo)': 'interiorizacion_fear',
        'eje: agencia (toda la letra)': 'agencia_all',
        'eje: colectividad (bloques de miedo)': 'colectividad_fear',
        'arco: giro a la agencia': 'arc_shape:turn_to_agency',
        'arco: escalada': 'arc_shape:escalation',
        'arco: bucle': 'arc_shape:loop',
        'arco: episodio': 'arc_shape:episodic',
        'voz: teme quien canta (yo o nosotros)': 'first_person',
        'voz: teme quien canta (yo)': 'who_fears:speaker',
    }
    fear_songs['agentic'] = fear_songs['response:defiance_confrontation'] + fear_songs['response:transformation_agency']
    fear_songs['passive'] = (fear_songs['response:flight_avoidance'] + fear_songs['response:paralysis']
                             + fear_songs['response:plea_prayer'] + fear_songs['response:resignation'])
    # las fracciones de categoría se normalizan por el peso de miedo de la canción
    fear_songs['first_person'] = fear_songs['who_fears:speaker'] + fear_songs['who_fears:speaker_collective']
    for col in [c for c in fear_songs if c.startswith(('threat_locus:', 'response:', 'arc_shape:', 'who_fears:'))] + [
            'agentic', 'passive', 'first_person']:
        fear_songs[col] = fear_songs[col] / fear_songs.fear_weight
    measures['lectura: respuesta de enfrentar/decidir'] = 'agentic'
    measures['lectura: respuesta de huir/someterse'] = 'passive'
    for label, col in measures.items():
        d = fear_songs.dropna(subset=[col])
        t = trend_test(d, col)
        trend_rows.append(dict(measure=label, column=col, **t))
        b = bootstrap_period(d, col)
        bl = bootstrap_period(d, col, by_language=True)
        period_rows += [dict(measure=label, column=col, **r) for r in b.to_dict('records')]
        period_rows += [dict(measure=label, column=col, **r) for r in bl.to_dict('records')]
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(OUT / 'trends.csv', index=False)
    pd.DataFrame(period_rows).to_csv(OUT / 'period_means.csv', index=False)

    # sensibilidad: solo canciones con miedo acordado; solo etiquetas de selección no-«fear»; sin el encoder principal
    sens = []
    for name, subset in [('miedo acordado por ambos lectores', fear_songs[fear_songs.fear_weight == 1]),
                         ('excluye canciones elegidas como «miedo»', fear_songs[fear_songs.selection_label != 'fear']),
                         ('solo inglés', fear_songs[fear_songs.language == 'en']),
                         ('solo español', fear_songs[fear_songs.language == 'es'])]:
        for col in ['threat_locus:internal_self', 'threat_locus:external_collective', 'interiorizacion_fear',
                    'agentic', 'agencia_all']:
            d = subset.dropna(subset=[col])
            if d.language.nunique() and len(d) > 8:
                sens.append(dict(subset=name, column=col, **trend_test(d, col, n=4000)))
    for key in ENCODERS:
        d = song_scores[key].join(table)
        d = d[d.fear_weight > 0].assign(w=lambda z: z.fear_weight)
        for col in ['interiorizacion_fear', 'agencia_all', 'colectividad_fear']:
            sens.append(dict(subset=f'encoder {key}', column=col, **trend_test(d.dropna(subset=[col]), col, n=4000)))
    pd.DataFrame(sens).to_csv(OUT / 'trend_sensitivity.csv', index=False)

    # ---- 5. subtemas: ¿los descubre el clustering libre o hace falta preguntar?
    fdocs = np.array(docs)[fear_doc_mask]
    fear_blocks = pd.DataFrame(dict(document_id=fdocs, song_id=[d.split('::')[0] for d in fdocs],
                                    weight=[canon_weight[d] for d in fdocs]))
    fear_blocks.to_csv(OUT / 'fear_blocks.csv', index=False)
    loc_cols = [f'threat_locus:{l}' for l in LOCI]
    fear_tbl = table[table.fear_weight > 0]
    majority = fear_tbl[loc_cols].idxmax(axis=1).str.split(':').str[1]
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rng = np.random.default_rng(SEED)
    clus_rows = []
    for key in ENCODERS:
        for vname in ['raw', 'centered']:
            xv = variants[key][vname]
            ids, vecs = [], []
            for sid, b in fear_blocks.groupby('song_id'):
                v = np.average(xv[[doc_index[d] for d in b.document_id]], axis=0, weights=b.weight)
                ids.append(sid); vecs.append(v / np.linalg.norm(v))
            vecs = np.array(vecs)
            m = majority.loc[ids].values
            langs = table.loc[ids, 'language'].values
            for k in (6, 8):
                lab_k = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(vecs)
                boot = []
                for _ in range(200):
                    idx = np.unique(rng.integers(0, len(vecs), len(vecs)))
                    boot.append(adjusted_rand_score(lab_k[idx], AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(vecs[idx])))
                clus_rows.append(dict(encoder=key, variant=vname, k=k, songs=len(ids),
                                      nmi_reader_locus=float(normalized_mutual_info_score(m, lab_k)),
                                      ari_reader_locus=float(adjusted_rand_score(m, lab_k)),
                                      nmi_language=float(normalized_mutual_info_score(langs, lab_k)),
                                      bootstrap_ari=float(np.mean(boot))))
    pd.DataFrame(clus_rows).to_csv(OUT / 'clustering_vs_readers.csv', index=False)
    # agrupamiento de bloques (estilo BERTopic) solo como diagnóstico de estabilidad
    k_scan, _ = cluster_fear_blocks(variants[main_key]['centered'][fear_doc_mask])
    k_scan.to_csv(OUT / 'block_cluster_k_scan.csv', index=False)

    # subtemas como preguntas: similitud con conceptos-prototipo frente a los marcadores de los lectores
    readers_fear = songs_r[songs_r.fear_reference.isin(FEAR_REFS)]
    markers = readers_fear.assign(m=readers_fear.context_markers.str.split('|')).explode('m')
    marker_share = (markers.groupby(['song_id', 'm']).size().unstack(fill_value=0) / 2).reindex(fear_tbl.index).fillna(0)
    rows = []
    for p in PERIODS:
        parts = []
        for lang, s in fear_tbl[fear_tbl.period == p].groupby('language'):
            w = s.fear_weight
            parts.append((marker_share.loc[s.index].div(w, axis=0) * w.values[:, None]).sum() / w.sum())
        rows.append(pd.concat(parts, axis=1).mean(axis=1).rename(p))
    pd.concat(rows, axis=1).to_csv(OUT / 'context_markers_by_period.csv')
    concept_marker = {'noche_monstruos': 'horror_fiction', 'vigilancia': 'state_police_surveillance',
                      'guerra_politica': 'war_conflict', 'calle_crimen': 'urban_violence_crime',
                      'pobreza_crisis': 'economy_class', 'control_pareja': 'gender_violence',
                      'ansiedad_salud_mental': 'mental_health', 'fama_pantallas': 'media_technology_fame',
                      'fe_dios': 'religion_faith', 'familia_hogar': 'family_home', 'deseo_cuerpo': 'romance_sex'}
    cv_rows = []
    for key, (x, anchors) in emb.items():
        basis = language_subspace(anchors, anchor_ids, k=2)
        cvec = concept_vectors(project_out(anchors, basis), anchor_ids, spec)
        xp = variants[key]['anchor_projected']
        for concept, marker in concept_marker.items():
            sims = xp @ cvec[concept]
            sc = pd.Series({sid: np.average(sims[[doc_index[d] for d in b.document_id]], weights=b.weight)
                            for sid, b in fear_blocks.groupby('song_id')})
            has = marker_share.reindex(sc.index)[marker].fillna(0) >= .5 if marker in marker_share else sc < -9
            cv_rows.append(dict(encoder=key, concept=concept, marker=marker, songs_with_marker=int(has.sum()),
                                auc=auc(sc[has], sc[~has])))
    pd.DataFrame(cv_rows).to_csv(OUT / 'concept_validation.csv', index=False)

    # ---- 5b. valor en juego por período (lecturas con miedo; igual peso por idioma)
    vr = readers_fear.copy()
    for name, pattern in VALUES.items():
        vr[name] = vr.value_at_stake.str.lower().str.contains(pattern).astype(float)
    per_song = vr.groupby('song_id')[list(VALUES)].sum().div(2).reindex(fear_tbl.index).fillna(0)
    rows = []
    for p in PERIODS:
        parts = []
        for lang, s_ in fear_tbl[fear_tbl.period == p].groupby('language'):
            w = s_.fear_weight
            parts.append((per_song.loc[s_.index].div(w, axis=0) * w.values[:, None]).sum() / w.sum())
        rows.append(pd.concat(parts, axis=1).mean(axis=1).rename(p))
    pd.concat(rows, axis=1).to_csv(OUT / 'values_by_period.csv')
    json.dump({'uncoded_readings': int((vr[list(VALUES)].sum(axis=1) == 0).sum()), 'readings': len(vr)},
              open(OUT / 'values_coding.json', 'w'))

    # ---- 6. arcos: estados por bloque → trayectoria amenaza ↔ control
    grid = np.linspace(0, 1, 10)
    traj_rows = []
    pos = inv.set_index('document_id')[['rel_pos', 'position']]
    br2 = br.join(pos, on='document_id')
    br2['score'] = br2.state.map(STATE_SCORE)
    for (sid, reader), d in br2.dropna(subset=['score']).groupby(['song_id', 'reader']):
        d = d.sort_values('rel_pos')
        if len(d) < 2:
            continue
        y = np.interp(grid, d.rel_pos.values, d.score.values)
        traj_rows.append(dict(song_id=sid, reader=reader, **{f't{i}': float(v) for i, v in enumerate(y)}))
    traj = pd.DataFrame(traj_rows)
    traj.to_csv(OUT / 'arc_trajectories_readers.csv', index=False)
    states_public = br2[['song_id', 'reader', 'block', 'document_id', 'canonical_id', 'position', 'rel_pos', 'state', 'support']]
    states_public.to_csv(OUT / 'block_states.csv', index=False)

    # concordancia bloque a bloque: ¿el eje agencia sigue los estados de los lectores dentro de la canción?
    xs = variants[main_key]['anchor_projected']
    anchors_p = project_out(emb[main_key][1], language_subspace(emb[main_key][1], anchor_ids, 2))
    agency_axis = axis_vectors(anchors_p, anchor_ids, spec)['agencia']['vector']
    block_agency = pd.Series(xs @ agency_axis, index=docs)
    from scipy.stats import spearmanr
    within = []
    for sid, d in br2.dropna(subset=['score']).groupby('song_id'):
        m = d.groupby('canonical_id').score.mean()
        if m.nunique() > 1 and len(m) >= 4:
            within.append(spearmanr(m.values, block_agency.loc[m.index].values).statistic)
    within = np.array([w for w in within if np.isfinite(w)])
    pooled = br2.dropna(subset=['score']).groupby('canonical_id').score.mean()
    arc_validation = dict(songs=int(len(within)), mean_within_song_spearman=float(within.mean()),
                          share_positive=float((within > 0).mean()),
                          pooled_block_spearman=float(spearmanr(pooled.values, block_agency.loc[pooled.index].values).statistic))
    json.dump(arc_validation, open(OUT / 'arc_embedding_validation.json', 'w'), indent=1)

    # ---- 7. contexto: con qué conceptos convive el miedo en cada período
    anchors_main = emb[main_key][1]
    concepts = concept_vectors(project_out(anchors_main, language_subspace(anchors_main, anchor_ids, 2)), anchor_ids, spec)
    cm = np.array([concepts[c] for c in concepts])
    sim = xs @ cm.T                                             # bloques × conceptos
    sim = (sim - sim[nonfiller_mask].mean(0)) / sim[nonfiller_mask].std(0)
    ctx_rows = []
    rng = np.random.default_rng(SEED)
    for p in PERIODS:
        fear_p = [d for d in fdocs if manifest.loc[d.split('::')[0], 'period'] == p]
        sids = sorted({d.split('::')[0] for d in fear_p})
        by_song = {s: [doc_index[d] for d in fear_p if d.startswith(s)] for s in sids}
        song_mean = np.array([sim[by_song[s]].mean(0) for s in sids])
        sw = table.loc[sids, 'fear_weight'].values
        point = np.average(song_mean, axis=0, weights=sw)
        boots = []
        for _ in range(3000):
            idx = rng.integers(0, len(sids), len(sids))
            if sw[idx].sum() > 0:
                boots.append(np.average(song_mean[idx], axis=0, weights=sw[idx]))
        lo, hi = np.percentile(boots, [5, 95], axis=0)
        for j, c in enumerate(concepts):
            ctx_rows.append(dict(period=p, concept=c, z=float(point[j]), lo90=float(lo[j]), hi90=float(hi[j]),
                                 songs=len(sids)))
    pd.DataFrame(ctx_rows).to_csv(OUT / 'fear_context_concepts.csv', index=False)

    # ---- 8. mapas UMAP para la figura del idioma y la del espacio del miedo
    import umap
    maps = []
    for key, vname in [('e5_small', 'raw'), ('e5_large_fear', 'raw'), ('e5_large_fear', 'centered')]:
        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(variants[key][vname])
        maps.append(pd.DataFrame(dict(encoder=key, variant=vname, document_id=docs, x=xy[:, 0], y=xy[:, 1],
                                      song_id=doc_song, language=doc_lang, fear_block_weight=[canon_weight.get(d, 0) for d in docs])))
    # mapa libre de los pasajes de miedo: ¿sus vecinos comparten el lugar de la amenaza?
    zone_of = {'external_collective': 'afuera', 'supernatural_fiction': 'afuera', 'external_person': 'afuera',
               'relational_loss': 'entre', 'existential_time': 'entre', 'internal_self': 'adentro'}
    zones = fear_tbl[loc_cols].T.groupby(lambda c: zone_of[c.split(':')[1]]).sum().T
    song_zone = zones.idxmax(axis=1).where(zones.max(axis=1) > 0, 'entre')
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    knn_rows = []
    y_zone = song_zone.loc[fear_blocks.song_id].values
    for key in ENCODERS:
        for vname in ['raw', 'centered']:
            xb = variants[key][vname][[doc_index[d] for d in fear_blocks.document_id]]
            acc = cross_val_score(KNeighborsClassifier(10, metric='cosine'), xb, y_zone,
                                  groups=fear_blocks.song_id.values, cv=GroupKFold(5)).mean()
            knn_rows.append(dict(encoder=key, variant=vname, knn10_zone_accuracy=float(acc),
                                 majority_baseline=float(pd.Series(y_zone).value_counts(normalize=True).max())))
    pd.DataFrame(knn_rows).to_csv(OUT / 'knn_zone_check.csv', index=False)
    xf = variants[main_key]['centered'][[doc_index[d] for d in fear_blocks.document_id]]
    xy = umap.UMAP(n_neighbors=12, min_dist=0.08, metric='cosine', random_state=42).fit_transform(xf)
    maps.append(pd.DataFrame(dict(encoder=main_key, variant='fear_blocks_centered', document_id=fear_blocks.document_id,
                                  x=xy[:, 0], y=xy[:, 1], song_id=fear_blocks.song_id,
                                  language=table.loc[fear_blocks.song_id, 'language'].values,
                                  fear_block_weight=fear_blocks.weight, zone=y_zone)))
    pd.concat(maps).to_csv(OUT / 'umap_maps.csv', index=False)
    song_zone.rename('zone').to_csv(OUT / 'song_zone.csv')

    # ---- tabla de canciones pública (sin letras)
    public = songs_r[['song_id', 'reader', 'fear_reference', 'fear_polarity', 'who_fears', 'threat_locus',
                      'threat_locus_secondary', 'threat_object', 'response', 'arc_shape', 'arc_steps',
                      'value_at_stake', 'context_markers', 'confidence', 'counterreading']]
    public.merge(manifest[['title', 'artist', 'year', 'language', 'period']], left_on='song_id', right_index=True) \
        .to_csv(OUT / 'readings_public.csv', index=False)
    scores.drop(columns=[c for c in scores if c.startswith('context:')], errors='ignore').to_csv(OUT / 'song_table.csv')
    summary = dict(
        songs=len(table), fear_agreed=int(table.fear_agreed.sum()), fear_split=int((table.fear_votes == 1).sum()),
        fear_absent=int((table.fear_votes == 0).sum()), fear_blocks=int(fear_doc_mask.sum()),
        main_encoder=main_key, arc_validation=arc_validation,
        fear_by_period={PERIOD_LABELS[p]: float(table[table.period == p].fear_weight.sum()) for p in PERIODS})
    json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(agreement.to_string())
    print(lang_df.to_string())
    print(pd.DataFrame(axis_rows).to_string())
    print(trends.to_string())
    print(k_scan.to_string())
    print(pd.DataFrame(clus_rows).round(3).to_string())
    print(pd.DataFrame(cv_rows).pivot(index='concept', columns='encoder', values='auc').round(2).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
