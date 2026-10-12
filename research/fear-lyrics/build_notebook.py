"""Author the final metadata-only notebook from completed public result receipts.

This builder does not execute notebook cells, read lyrics, or call APIs. Run it
only after the final corpus/model results and figures have been regenerated.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import csv
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def build(output=None, expected_songs=None, expected_models=None):
    manifest = json.loads((HERE / 'corpus-manifest.json').read_text())
    results = HERE / 'results'
    dirs = sorted(p for p in results.iterdir() if p.is_dir() and (p / 'model_manifest.json').exists())
    receipts = [json.loads((p / 'model_manifest.json').read_text()) for p in dirs]
    diagnostics = json.loads((results / 'corpus_diagnostics.json').read_text())
    if any(r.get('unique_documents') != diagnostics.get('unique_blocks') for r in receipts):
        raise ValueError('Model receipts do not share the final corpus of unique blocks')
    with (results/'section_inventory.csv').open() as handle:
        common_song_ids = {r['song_id'] for r in csv.DictReader(handle)}
    common_manifest = [r for r in manifest if r['song_id'] in common_song_ids]
    if len(common_manifest) != diagnostics.get('songs_with_eligible_blocks'):
        raise ValueError('The common result cohort has incomplete public metadata')
    expanded_results = HERE/'expanded/results'
    expanded_dirs = sorted(p for p in expanded_results.iterdir() if p.is_dir() and (p/'model_manifest.json').exists()) if expanded_results.exists() else []
    if expected_songs is not None and len(common_manifest) != expected_songs:
        raise ValueError('The common result cohort has not reached the requested corpus size')
    if expected_models is not None and len(dirs) != expected_models:
        raise ValueError('The final result receipts have not reached the requested model count')
    if any(r.get('results_status') != 'executed' for r in receipts):
        raise ValueError('The notebook requires executed model receipts')
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))
    md(f"""
# Qué conservan los embeddings cuando una canción habla de miedo

**Cuaderno en español · {len(manifest)} canciones seleccionadas · comparación común: {len(common_manifest)} canciones / {len(dirs)} encoders · corte: 11 de octubre de 2026.**

Lejos de criticar o alabar una época, me interesa comprender qué sentimientos perduran entre décadas, cómo se desarrolla la historia que cuenta una canción y qué considera valioso proteger su voz. Esta vez sigo canciones sobre el miedo mediante embeddings, subtemas y arcos argumentales. El tiempo histórico compara publicaciones; el tiempo argumental sigue pasos dentro de una letra.

Una canción puede nombrar el miedo y negarlo, prestar su voz a un personaje o volver a un estribillo sin resolver nada. La revisión anual uniforme reúne dos lecturas por cada una de las 120 letras: 53 referencias apoyadas, 37 ausencias y 30 casos inciertos. Los resultados anteriores del panel de 40 se conservan como fase histórica, sin mezclarlos con los nuevos votos. La comparación entre cuatro encoders usa exactamente las mismas 40 canciones; la expansión local constituye una cohorte separada.

Las tablas y conclusiones se calculan desde los recibos de las corridas guardadas. La revisión interpretativa añade votos de un panel automatizado con procedencia y evidencia por ID. Los resultados describen el corpus seleccionado; las diferencias entre períodos pueden reflejar artistas, versiones, idiomas y decisiones de selección.
""")
    code(r"""
from pathlib import Path
import json, os, sys, subprocess
import numpy as np
import pandas as pd
from IPython.display import display, Image, SVG, Markdown

# Soporta tanto el repositorio como un paquete descargado con estructura plana.
candidates = [Path.cwd(), *Path.cwd().parents]
STUDY = next((d for p in candidates for d in [p/'research/fear-lyrics', p]
              if (d/'analyze.py').exists() and (d/'corpus-manifest.json').exists()), None)
if STUDY is None:
    raise FileNotFoundError('Extrae el paquete completo y abre el notebook junto a sus scripts y resultados.')
ROOT = STUDY.parents[1] if STUDY.parent.name == 'research' else STUDY.parent
RESULTS = STUDY/'results'
FIGURES = STUDY/'figures'
PRIVATE = Path(os.environ.get('FEAR_PRIVATE_CACHE', ROOT/'.research-cache/fear-lyrics'))
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
full_manifest = pd.DataFrame(read(STUDY/'corpus-manifest.json'))
common_ids = set(pd.read_csv(RESULTS/'section_inventory.csv').song_id)
manifest = full_manifest[full_manifest.song_id.isin(common_ids)].copy()
EXPANDED = STUDY/'expanded'
EXPANDED_RESULTS = EXPANDED/'results'
expanded_modeldirs = sorted(p for p in EXPANDED_RESULTS.iterdir() if p.is_dir() and (p/'model_manifest.json').exists()) if EXPANDED_RESULTS.exists() else []
PANEL = next((p for p in [STUDY/'pilot-comparison/results/arc-panel', RESULTS/'arc-panel']
              if (p/'panel_votes.json').exists()), RESULTS/'arc-panel')
diag = read(RESULTS/'corpus_diagnostics.json')
modeldirs = sorted(p for p in RESULTS.iterdir() if p.is_dir() and (p/'model_manifest.json').exists())
model_receipts = {p.name: read(p/'model_manifest.json') for p in modeldirs}

# Abrir/ejecutar las celdas lee resultados guardados. No se ejecutan experimentos.
# Para recalcular, se requieren corpus privado autorizado y embeddings locales.
RERUN_EXPERIMENTS = os.environ.get('RERUN_FEAR_EXPERIMENTS') == '1'

def frame(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def figure(name, caption=None):
    options = [FIGURES/name, RESULTS/'bayes'/name]
    path = next((p for p in options if p.exists()), None)
    if path is None:
        display(Markdown('Figura no incluida en esta versión del paquete: `'+name+'`.'))
        return
    if path.suffix == '.svg': display(SVG(filename=str(path)))
    else: display(Image(filename=str(path)))
    if caption: display(Markdown('*'+caption+'*'))

def public_rows(value):
    if isinstance(value, list): return value
    if isinstance(value, dict):
        for key in ['rows', 'annotations', 'votes', 'records', 'models']:
            if isinstance(value.get(key), list): return value[key]
    return []

assert manifest.song_id.is_unique and modeldirs
print('Colección seleccionada:', len(full_manifest), 'canciones')
print('Comparación común:', len(manifest), 'canciones;', manifest.artist.nunique(), 'artistas')
print('Con letras recuperadas:', diag.get('songs_with_lyrics'), '| bloques únicos:', diag.get('unique_blocks'))
print('Modelos ejecutados:', ', '.join(r['encoder']['model_id'] for r in model_receipts.values()))
""")
    md("""
## 1. Una voz no es un artista; un vector no es una emoción

Distinguimos situación, emoción representada, postura y arco narrativo. Cada lectura necesita identificar quién habla, quién experimenta el miedo, a qué se enfrenta y qué contradice la interpretación. El oyente constituye otro objeto: este estudio no mide audio ni recepción.

La claridad de [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) inspira una explicación gradual: texto, vector, vecindad y decisiones observables. La cercanía entre vectores puede reflejar vocabulario compartido sin resolver diferencias de negación, roles o secuencia de eventos.
""")
    md("""
## 2. Corpus, períodos y versiones

La selección intencional se organiza por idioma y período: 1980s, 1990s, 2000s, 2010s y 2020–2026 parcial. La primera tabla presenta toda la colección y la segunda la cohorte común de encoders. El tamaño efectivo se presenta en la tabla; se conserva el piloto en su archivo separado y las corridas finales no se mezclan con sus denominadores.

Usamos el año de primera publicación de la grabación y distinguimos esa fecha del año de su éxito comercial. Las etiquetas iniciales orientan la selección; no entran en el índice de recuperación. El equilibrio de idiomas y períodos no iguala géneros, países o concentración por artista.
""")
    code("""
display(Markdown('### Colección ampliada'))
display(full_manifest.groupby(['decade','language']).size().unstack(fill_value=0).rename(columns={'en':'Inglés','es':'Español'}))
display(Markdown('### Cohorte común para comparar todos los encoders'))
display(manifest.groupby(['decade','language']).size().unstack(fill_value=0).rename(columns={'en':'Inglés','es':'Español'}))
display(manifest[['title','artist','year','language','selection_label']])
coverage = frame(STUDY/'provider-coverage.csv')
agreement = frame(STUDY/'provider-agreement.csv')
if not coverage.empty:
    display(coverage.assign(available=coverage.status.astype(str).eq('200')).groupby('provider').agg(
        consultadas=('song_id','size'), disponibles=('available','sum')))
if not agreement.empty:
    display(agreement.groupby(['provider_a','provider_b']).agg(
        pares=('song_id','size'), mediana_similitud=('sequence_similarity','median'),
        mínimo=('sequence_similarity','min'), idénticas=('normalized_exact_match','sum')))
figure('corpus-providers.png')
""")
    md("""
**Alcance de la tabla y la figura.** `provider-coverage.csv` registra las 120 canciones de la colección vigente. La figura `corpus-providers.png` corresponde a la cohorte emparejada de 40 canciones.

La procedencia forma parte de la medida. [LyricsGenius](https://lyricsgenius.readthedocs.io/en/master/how_it_works.html) usa Genius para metadatos y búsqueda, y extrae la letra de la página. Disponibilidad y acuerdo entre proveedores no certifican la edición correcta ni la independencia de las fuentes.

Las letras permanecen en caché privada. El paquete público contiene IDs, hashes, metadatos, agregados, figuras y scripts. Distinguir grabación de estudio y versión en vivo, o conservar pronombres y negaciones, cambia el objeto del análisis.
""")
    md("""
## 3. Bloques para representar; canciones para ponderar

Se preservan párrafos y orden; los segmentos largos se dividen en ventanas. Los bloques repetidos exactamente dentro de una canción se deduplican para ajustar tópicos. Las comparaciones conservan dos variantes: primera aparición y todas las repeticiones.

El ajuste global recibe cada bloque único una vez. Las proporciones temporales dan igual peso a cada canción dentro de idioma y después a los idiomas disponibles. Los intervalos bootstrap remuestrean esas canciones; no corrigen el diseño de selección.
""")
    code("""
keys = [k for k in ['blocks','unique_blocks','exact_repeated_blocks','cutoff','corpus_identity'] if k in diag]
display(pd.Series({k:diag[k] for k in keys}))
inventory = frame(RESULTS/'section_inventory.csv')
if not inventory.empty: assert inventory.document_id.is_unique
models = []
for directory in modeldirs:
    receipt = model_receipts[directory.name]
    encoder = receipt['encoder']
    models.append({'modelo':encoder['model_id'], 'dimensiones':encoder.get('dimensions'),
        'ventana_tokens':encoder.get('max_seq_length'), 'bloques':receipt.get('unique_documents'),
        'bloques_truncados':receipt.get('truncated_documents'),
        'segundos_corrida':receipt.get('runtime_seconds'), 'estado':receipt['results_status']})
display(pd.DataFrame(models))
""")
    md("""
## 4. Modelos y contrastes de significado

Los dos modelos locales, [MiniLM multilingüe](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) y [E5 small multilingüe](https://huggingface.co/intfloat/multilingual-e5-small), tienen tareas y ventanas diferentes. E5 distingue los prefijos `query:` y `passage:`. Los encoders OpenAI se identifican en la tabla por modelo y dimensiones realmente utilizadas.

Se usa normalización L2 dentro de cada modelo. Comparamos rankings, particiones y sensibilidad: un coseno más alto en otro espacio no significa más miedo. Los límites de tokens importan especialmente cuando el contraste aparece al final del bloque.

Las veinte tripletas originales EN/ES examinan traducción, paráfrasis, contexto, motivos, experienciador, negación, roles, repetición, orden y truncación. A no es siempre una respuesta correcta: negación y roles pueden conservar cercanía semántica mientras cambian la afirmación.
""")
    code("""
probe_tables = [pd.DataFrame(read(p/'semantic_probes.json')['probes']).assign(model=model_receipts[p.name]['encoder']['model_id'])
                for p in modeldirs if (p/'semantic_probes.json').exists()]
probes = pd.concat(probe_tables, ignore_index=True) if probe_tables else pd.DataFrame()
if not probes.empty:
    display(probes[['model','id','category','related_cosine','contrast_cosine','related_minus_contrast']])
    display(probes[probes.category.isin(['narrative_order','long_ending'])][
        ['model','id','related_cosine','contrast_cosine','query_tokenizer_tokens','query_exceeds_max_seq_length']])
figure('semantic-probes.png')
""")
    md("""
## 5. UMAP, PCA 3D y la varianza que queda fuera

UMAP 2D dispone bloques en una proyección exploratoria ajustada separadamente del clustering. El color corresponde al tópico local; los IDs y colores no alinean grupos entre modelos.

PCA usa una sola base global por modelo sobre todos sus bloques únicos normalizados. Un ajuste full-SVD proporciona hasta veinte componentes y el mapa usa los primeros tres de ese mismo ajuste. Las tablas separan la varianza de tres ejes del espectro acumulado. Ni los ejes PCA ni las distancias 2D/3D constituyen escalas de miedo.

Los centroides temporales se calculan en el embedding original: media de bloques dentro de canción, de canciones dentro de idioma y de los idiomas presentes. Se proyectan con la misma base PCA, sin reajustarla por año. Los puntos con menos de tres canciones se marcan como soporte escaso; las líneas sólo conectan períodos observados consecutivos con al menos tres canciones en ambos extremos. Una ausencia no se imputa como cero.

El selector anual cambia qué grabaciones observamos. Las trayectorias describen una composición cambiante, no movimiento de las mismas canciones. Los representantes de cada centroide son hasta tres canciones distintas elegidas por coseno en el espacio original, y se publican sus IDs y metadatos.
""")
    code("""
projection_receipts = []
scree = []
for directory in modeldirs:
    path = directory/'projections.json'
    if not path.exists(): continue
    p = read(path); r = p['receipt']
    a = r['explained_variance_ratio']
    spectrum = r.get('scree_explained_variance_ratio', a)
    projection_receipts.append({'modelo':r['model_id'],'canciones':r['input_songs'],
        'bloques':r['input_documents'],'dimensiones_origen':r['input_dimensions'],
        'PC1':a[0],'PC2':a[1],'PC3':a[2], 'acumulada_3D':r['explained_variance_total'],
        'acumulada_hasta_20':sum(spectrum),'PCs_en_recibo':len(spectrum),
        'SHA_corpus':r['corpus_sha256'], 'semilla':r['seed'],'solver':r['solver']})
    scree.extend({'modelo':r['model_id'],'PC':i+1,'varianza':value,'acumulada':sum(spectrum[:i+1])}
                 for i,value in enumerate(spectrum))
    assert len(p['points']) == r['input_documents']
    assert all(c['line_eligible'] == (c['songs'] >= 3) and not c['imputed'] for c in p['centroids'])
projection_table = pd.DataFrame(projection_receipts)
display(projection_table)
if not projection_table.empty:
    for r in projection_receipts:
        display(Markdown('**'+r['modelo']+'** conserva **'+f"{r['acumulada_3D']:.1%}"+
            '** de la varianza en tres dimensiones. El resto queda fuera de este mapa.'))
figure('embeddings-umap-all.svg', 'UMAP exploratorio separado del ajuste de tópicos; no comparar distancias visuales entre modelos.')
figure('embeddings-pca3-all.svg', 'PCA global por modelo; mismos ejes para todos los años de esa representación.')
figure('embedding-variance.svg', 'Varianza explicada y acumulada de los componentes calculados; fracciones dentro de cada modelo.')
display(pd.DataFrame(scree))
""")
    md("""
## 6. BERTopic, palabras y nombres interpretativos

[BERTopic](https://arxiv.org/abs/2203.05794) combina embeddings, UMAP, HDBSCAN y descriptores c-TF-IDF. Cada modelo se ajusta globalmente; los parámetros y semillas se registran con sus resultados. El ruido `-1` permanece visible.

La [jerarquía de tópicos](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) conserva fusiones por similitud. Los nombres propuestos por un modelo lector son otra capa: no cambian las asignaciones ni reemplazan las palabras c-TF-IDF. Su procedencia, muestras e IDs permiten examinar interpretaciones y contralecturas.
""")
    code("""
labels = read(RESULTS/'topic_labels.json') if (RESULTS/'topic_labels.json').exists() else []
labels = public_rows(labels)
for directory in modeldirs:
    model_id = model_receipts[directory.name]['encoder']['model_id']
    topic_payload = read(directory/'topics.json')
    names = {int(t['topic_id']):t for t in labels if t.get('model_id') == model_id}
    table = pd.DataFrame([{'tópico':t['topic_id'],'bloques':t['unique_blocks'],'canciones':t['songs'],
        'palabras':' · '.join(x['term'] for x in t['terms'][:6]),
        'nombre_propuesto':names.get(t['topic_id'],{}).get('label',''),
        'modelo_lector':names.get(t['topic_id'],{}).get('labeler_model','')}
        for t in topic_payload['topics']])
    display(Markdown('### '+model_id)); display(table)
    display(pd.DataFrame([topic_payload.get('parameters',{})]))
    if (FIGURES/(directory.name+'-hierarchy.png')).exists(): figure(directory.name+'-hierarchy.png')
cards_path = STUDY/'interpretive-cards.json'
if cards_path.exists():
    cards = read(cards_path)['cards']
    display(pd.DataFrame([{'canción':c['title'],'situación_propuesta':c['proposed_subtopic']['label'],
        'contralectura':c['interpretive_limit']} for c in cards]))
    figure('reading-cases.png')
""")
    md("""
## 7. Estabilidad, artistas e idioma

El adjusted Rand index describe acuerdo entre semillas sobre los mismos bloques. Se presenta también la fracción asignada en ambas corridas. Un acuerdo alto sobre un único grupo, sobre idioma o sobre ruido necesita lectura de composición.

Las comparaciones entre encoders se hacen con particiones, no con igualdad de IDs. La concentración por artista y la retirada de un artista describen cuánto depende un tópico de unos pocos intérpretes; esa retirada no equivale a reentrenar el clustering.
""")
    code("""
for directory in modeldirs:
    display(Markdown('### '+model_receipts[directory.name]['encoder']['model_id']))
    display(frame(directory/'seed_stability.csv'))
    concentration = frame(directory/'artist_concentration.csv')
    if not concentration.empty: display(concentration.sort_values('largest_artist_share',ascending=False).head(8))
for filename in ['topic_language_association.csv','embedding_model_cluster_agreement.csv']:
    table = frame(RESULTS/filename)
    if not table.empty: display(table)
figure('language-confound.png')
""")
    md("""
## 8. Proporciones de bloques y presencia de canciones

Las proporciones temporales usan los tópicos globales, con igual peso por canción y luego por idioma. Las curvas representan composición de bloques en los casos seleccionados. Se comparan primera aparición y repeticiones con intervalos bootstrap pareados.

La presencia binaria pregunta algo distinto: ¿cuántas canciones contienen al menos un bloque del tópico? Una canción puede contener varios tópicos; esas tasas no suman uno. El denominador es el número efectivo de canciones de cada celda, que aparece como `k/n`.
""")
    code("""
for directory in modeldirs:
    figure(directory.name+'-decades.png')
    refrains = frame(directory/'repeated_refrain_sensitivity.csv')
    if not refrains.empty:
        display(refrains.assign(magnitud=refrains.repeated_minus_unique.abs()).sort_values('magnitud',ascending=False).head(5))
""")
    md("""
## 8.1 El contraste bayesiano

[Julia Silge, Bayesian Blues](https://juliasilge.com/blog/bayesian-blues/) inspira la comparación entre tasas observadas, pooling y posteriores. Aquí los ensayos son canciones: `y_g | θ_g ~ Binomial(n_g, θ_g)` y `θ_g ~ Beta(α,β)`. El posterior condicional es `Beta(α+y_g, β+n_g−y_g)`.

Empirical Bayes estima α y β y sus intervalos plug-in condicionan a esa estimación. La integración jerárquica propaga incertidumbre de hiperparámetros bajo hiperpriors explícitos. El protocolo fija priors, límites, resolución e idioma del pooling; se contrastan Jeffreys y uniforme, pooling por idioma, retirada de una celda, perfiles de concentración y resolución numérica.

El gráfico no mezcla esos posteriores con proporciones de bloques. Los resultados condicionan a las asignaciones y al corpus seleccionado. La unidad canción evita contar versos como ensayos independientes, pero no elimina concentración por artista o dependencia de la elección de representación.
""")
    code("""
bayesdir = RESULTS/'bayes'
posterior = frame(bayesdir/'posterior_rates.csv')
if not posterior.empty:
    # Regla descriptiva fijada: primera celda intermedia en orden modelo/tópico/período/idioma.
    pooled = posterior[(posterior.method == 'hierarchical_grid') & (posterior.prior_context == 'pooled_languages')]
    intermediate = pooled[(pooled.successes > 0) & (pooled.successes < pooled.songs)]
    candidate = (intermediate if not intermediate.empty else pooled).sort_values(['model','topic_id','decade','language']).iloc[0]
    same = posterior[(posterior.model == candidate.model) & (posterior.topic_id == candidate.topic_id) &
        (posterior.decade == candidate.decade) & (posterior.language == candidate.language) &
        posterior.prior_context.isin(['pooled_languages','fixed'])]
    display(same[['model','topic_id','decade','language','method','prior_context','songs','successes',
                  'raw_rate','posterior_mean','credible_low','credible_high','identifiability']])
    display(Markdown('**Celda descriptiva:** '+candidate.model+' · T'+str(candidate.topic_id)+
        ' · '+str(candidate.decade)+'s · '+candidate.language.upper()+' · '+
        str(candidate.successes)+'/'+str(candidate.songs)+' canciones.'))
    for r in same.itertuples():
        display(Markdown(r.method+' ('+r.prior_context+'): media **'+f'{r.posterior_mean:.3f}'+
            '**; intervalo creíble **'+f'[{r.credible_low:.3f}, {r.credible_high:.3f}]'+ '**.'))
    display(frame(bayesdir/'grid_resolution.csv'))
    figure('beta_binomial_presence.svg')
else: display(Markdown('Esta versión no incluye posteriores guardados.'))
""")
    md("""
**Lectura del contraste.** Un intervalo plug-in estrecho no recibe más confianza por ser estrecho: condiciona a hiperparámetros estimados. Cambiar de método puede cambiar la media además del intervalo, porque introduce supuestos distintos. Los diagnósticos de límites y sensibilidad se conservan en los archivos. Las bandas no incluyen incertidumbre de selección histórica ni errores de asignación de tópicos.
""")
    md("""
## 8.2 Qué cambia al ampliar el corpus

La ampliación local se conserva en `expanded/results`. Compara la cohorte común y el corpus completo con los mismos dos encoders locales. Las corridas remotas que no cubren el corpus ampliado no se presentan como resultados de esa cohorte.

Se vuelve a ajustar el clustering y la base PCA sobre todos los bloques de la nueva cohorte. Por eso cambian los IDs de tópicos y los ejes geométricos: T0 y PC1 de una cohorte no se conectan automáticamente con T0 y PC1 de la otra. Comparamos número de grupos, soporte, ruido, idioma y varianza, además de situaciones concretas.
""")
    code("""
if expanded_modeldirs:
    expanded_diag = read(EXPANDED_RESULTS/'corpus_diagnostics.json')
    expanded_inventory = frame(EXPANDED_RESULTS/'section_inventory.csv')
    expanded_ids = set(expanded_inventory.song_id)
    expanded_manifest = full_manifest[full_manifest.song_id.isin(expanded_ids)]
    print('Cohorte ampliada:',len(expanded_manifest),'canciones;',expanded_diag['unique_blocks'],'bloques únicos')
    display(expanded_manifest.groupby(['decade','language']).size().unstack(fill_value=0))
    comparison_rows = []
    for name, cohort_dirs, cohort_diag in [('Común',modeldirs,diag),('Ampliada local',expanded_modeldirs,expanded_diag)]:
        for directory in cohort_dirs:
            receipt = read(directory/'model_manifest.json'); encoder = receipt['encoder']
            if name == 'Común' and not any(read(p/'model_manifest.json')['encoder']['model_id']==encoder['model_id'] for p in expanded_modeldirs): continue
            topics = read(directory/'topics.json')['topics']
            projection = read(directory/'projections.json')['receipt'] if (directory/'projections.json').exists() else {}
            comparison_rows.append({'cohorte':name,'modelo':encoder['model_id'],
                'canciones':cohort_diag['songs_with_lyrics'],'bloques':receipt['unique_documents'],
                'tópicos':sum(t['topic_id']!=-1 for t in topics),
                'ruido':sum(t['unique_blocks'] for t in topics if t['topic_id']==-1),
                'varianza_3D':projection.get('explained_variance_total'),
                'varianza_PC20':sum(projection.get('scree_explained_variance_ratio',[]))})
    cohort_comparison = pd.DataFrame(comparison_rows)
    display(cohort_comparison)
    for model_id, group in cohort_comparison.groupby('modelo'):
        before = group[group.cohorte.eq('Común')].iloc[0]; after = group[group.cohorte.eq('Ampliada local')].iloc[0]
        display(Markdown('**'+model_id+'**: '+str(before['tópicos'])+' → '+str(after['tópicos'])+
            ' tópicos al pasar de '+str(before['canciones'])+' a '+str(after['canciones'])+' canciones.'))
    expanded_bayes = frame(EXPANDED_RESULTS/'bayes/posterior_rates.csv')
    if not expanded_bayes.empty:
        display(expanded_bayes[expanded_bayes.method.eq('hierarchical_grid') &
            expanded_bayes.prior_context.eq('pooled_languages')][['model','topic_id','decade','language',
            'songs','successes','raw_rate','posterior_mean','credible_low','credible_high']])
    expanded_bayes_figure = EXPANDED_RESULTS/'bayes/beta_binomial_presence.svg'
    if expanded_bayes_figure.exists(): display(SVG(filename=str(expanded_bayes_figure)))
    for directory in expanded_modeldirs:
        table = frame(directory/'seed_stability.csv')
        display(Markdown('### '+read(directory/'model_manifest.json')['encoder']['model_id']+' · ampliación'))
        display(table)
    expanded_retrieval = []
    for path in sorted(EXPANDED_RESULTS.glob('rag*/summary.csv')):
        table = frame(path)
        if not table.empty: expanded_retrieval.append(table.assign(experimento=path.parent.name))
    if expanded_retrieval:
        table = pd.concat(expanded_retrieval,ignore_index=True)
        display(table[table.split.eq('test')][[c for c in ['experimento','method','arm','language',
            'recall_at_5','context_support_recall','negative_safe_abstention_rate'] if c in table]])
else: display(Markdown('Esta edición incluye únicamente la cohorte común.'))
""")
    md("""
## 9. Del perfil semántico al arco narrativo

Las curvas usan cuatro prototipos bilingües originales: miedo, amenaza, evitación y afrontamiento. El contraste `afrontamiento − (miedo + amenaza) / 2` es geométrico; su pendiente no mide intensidad emocional. Se comparan posiciones relativas de bloques, sin duración de audio.

La permutación de orden mantiene textos y prototipos y cambia su secuencia. Los valores p se ajustan por Benjamini–Hochberg dentro de cada variante. El resultado informa sobre esa representación; no demuestra por sí solo una trama. Las [formas afectivas de Reagan y colaboradores](https://link.springer.com/article/10.1140/epjds/s13688-016-0093-1) se estudiaron en libros extensos: dominio, extensión y unidad son distintos aquí.

[EmotionArcs (2024)](https://aclanthology.org/2024.latechclfl-1.7/) estudia trayectorias en novelas; [Social Story Frames (2026)](https://aclanthology.org/2026.acl-long.1934/) aborda marcos interpretativos de relatos sociales. Estos trabajos motivan distinguir situaciones, inferencias y trayectorias. Sus dominios y unidades difieren de una letra breve y repetitiva; sus resultados no validan el método sobre este corpus de canciones.

Dos lectores OpenAI examinan de forma separada bloques anónimos y ordenados, sin título, artista, año ni hipótesis de selección. Una tercera lectura de mini, acotada a desacuerdos predeclarados, reutiliza un modelo del panel; no es un tercer modelo independiente. El panel examina situaciones, voces, cambios de postura y contralecturas con evidencia por ID. El reconocimiento de letras conocidas sigue siendo posible aunque se omitan metadatos. Una distribución de votos muestra acuerdo o discrepancia entre los lectores automáticos; no convierte una curva semántica en arco narrativo.

La corrida primaria conserva su validación conjunta de categorías, IDs y cronología. El diagnóstico **posterior por campo**, calculado sin nuevas llamadas, separa validez de tipo/enum de cada categoría y validez de los anclajes argumentales. Una respuesta que falla cronología puede aportar un voto categórico de voz o situación; su arco sigue excluido. Los resultados primarios y esta lectura posterior aparecen en tablas separadas. La mayoría expresa acuerdo entre estos lectores y no exactitud semántica.
""")
    code("""
figure('semantic-trajectories.png')
for directory in modeldirs:
    display(Markdown('### '+model_receipts[directory.name]['encoder']['model_id']))
    display(frame(directory/'narrative_order_tests.csv'))
    songtests = frame(directory/'narrative_song_order_tests.csv')
    if not songtests.empty:
        tested = songtests[songtests.tested.astype(str).str.lower().eq('true')]
        display(tested.groupby('variant').agg(canciones=('song_id','size'),
            q_menor_005=('permutation_q_bh_within_variant',lambda x:int((x<.05).sum()))))

panel_dir = PANEL
votes_path = panel_dir/'panel_votes.json'
annotations_path = panel_dir/'song_annotations.json'
if votes_path.exists():
    votes = pd.DataFrame(public_rows(read(votes_path)))
    valid = votes[votes.status.eq('executed')] if 'status' in votes else pd.DataFrame()
    display(Markdown('### Corrida primaria: validación conjunta y cronología'))
    print('Lecturas solicitadas:', len(votes), '| lecturas con esquema y cronología válidos:', len(valid))
    print('La tercera lectura reutiliza mini; no es un tercer modelo independiente.')
    if (panel_dir/'summary.json').exists():
        primary_summary = read(panel_dir/'summary.json')
        display(pd.Series({k:primary_summary[k] for k in ['songs','initial_core_agreement_songs',
            'eligible_disagreements','third_reviews','unresolved_core_songs'] if k in primary_summary}))
    view = []
    for r in valid.to_dict('records'):
        reading = r.get('reading', {})
        meta = full_manifest[full_manifest.song_id.eq(r['song_id'])].iloc[0]
        view.append({'canción':meta.title, 'año':r['year'], 'idioma':r['language'],
            'modelo':r['model_id'], 'etapa':r['stage'], 'voz':reading.get('voice'),
            'postura':reading.get('stance'), 'miedo':reading.get('fear_relevance'),
            'polaridad':reading.get('fear_polarity'), 'arco':reading.get('arc'),
            'subtemas':reading.get('subtopics')})
    display(pd.DataFrame(view))
    annotations = public_rows(read(annotations_path)) if annotations_path.exists() else []
    consensus_table = pd.json_normalize(annotations)
    if not consensus_table.empty:
        display(consensus_table[['song_id','year','language','valid_votes','initial_core_agreement',
            'third_review_attempted','consensus.arc','consensus.subtopics','unresolved_fields']])
    # Caso descriptivo: primer ID con dos votos válidos. Cambia el ID para explorar otro.
    SELECTED_SONG_ID = next((r['song_id'] for r in annotations if r.get('valid_votes',0)>=2), None)
    if SELECTED_SONG_ID:
        meta = full_manifest[full_manifest.song_id.eq(SELECTED_SONG_ID)].iloc[0]
        display(Markdown('### Leer los pasos: '+meta.title+' · '+meta.artist))
        for r in valid[valid.song_id.eq(SELECTED_SONG_ID)].to_dict('records'):
            reading = r['reading']
            display(Markdown('**'+r['model_id']+' · '+r['stage']+'**: '+reading['label']))
            display(pd.DataFrame([{'paso':i+1,'estado':step['state'],
                'lectura':step['reading'],'IDs_evidencia':step.get('source_document_ids',[])}
                for i,step in enumerate(reading.get('steps',[]))]))
            display(pd.DataFrame([{'desde':link['from']+1,'hasta':link['to']+1,
                'relación':link['relation'],'lectura':link['reading']} for link in reading.get('links',[])]))
    temporal_readings = frame(panel_dir/'decade_readings.csv')
    if not temporal_readings.empty:
        display(temporal_readings)
else: display(Markdown('No se distribuyen votos narrativos ejecutados en esta versión.'))
""")
    md("""
### 9.1 Diagnóstico posterior: votos por campo y preocupaciones

Esta tabla reutiliza los votos existentes. Las categorías de la ontología permanecen congeladas y el arco exige anclajes y orden válidos en la corrida primaria. Los votos de mini en una segunda llamada comparten modelo; no aumentan el número de lectores independientes.

Una preocupación puede estar presente en un control donde el miedo está ausente. Por eso se distinguen todos los motivos, los motivos cruzados con consenso de miedo explícito/implícito y una sensibilidad que exige que los mismos dos votos sostengan ambos campos. La referencia al miedo incluye negarlo o plantearlo como hipótesis: no equivale a experimentarlo ni afirmarlo. Cada proporción conserva todas las canciones del período e idioma en el denominador, incluidos desacuerdos; los subtemas se solapan.
""")
    code("""
fieldwise_dir = PANEL/'fieldwise-posthoc'
if (fieldwise_dir/'summary.json').exists():
    fieldwise_summary = read(fieldwise_dir/'summary.json')
    assert fieldwise_summary['status']=='executed_offline_posthoc'
    assert fieldwise_summary['new_api_calls']==0
    fields = pd.DataFrame(fieldwise_summary['fields'])
    display(fields[['field','songs','initial_agree_songs','majority_resolved_songs','unresolved_songs','validity']])
    fieldwise_annotations = public_rows(read(fieldwise_dir/'song_annotations.json'))
    fieldwise_table = pd.json_normalize(fieldwise_annotations)
    display(fieldwise_table[['song_id','year','language','consensus.voice','consensus.fear_relevance',
        'consensus.fear_polarity','consensus.subtopics','consensus.arc','unresolved_fields']])
    print('Relevancia del miedo:', fieldwise_table['consensus.fear_relevance'].value_counts().to_dict())
    print('Arcos con acuerdo y anclajes válidos:',fieldwise_table.loc[
        ~fieldwise_table['consensus.arc'].eq('unresolved'),'consensus.arc'].value_counts().to_dict())
    concern_rates = frame(fieldwise_dir/'concern_presence_by_period.csv')
    if not concern_rates.empty:
        display(concern_rates[concern_rates.language.eq('balanced')][
            ['decade','view','label','songs','count','share']])
    figure('voted-subtopics-time.svg', 'Todas las preocupaciones y referencia al miedo son lecturas distintas; todos los casos conservan su denominador.')
    anchored = public_rows(read(fieldwise_dir/'anchored_cases.json'))
    display(pd.DataFrame([{'canción':r['title'],'año':r['year'],'idioma':r['language'],
        'arco':r['consensus']['arc'],'votos_con_anclajes':r['supporting_vote_count'],
        'selección':r['selection']} for r in anchored]))
    # Caso ilustrativo: primer ID ordenado entre todos los arcos con >=2 votos válidos.
    chosen = min(anchored,key=lambda r:r['song_id']) if anchored else None
    if chosen:
        display(Markdown('### Anclajes de un caso: '+chosen['title']))
        for index, reading in enumerate(chosen['readings']):
            display(Markdown('**'+chosen['models'][index]+' · '+chosen['stages'][index]+'**: '+reading['label']))
            display(pd.DataFrame([{'paso':i+1,'estado':step['state'],'lectura':step['reading'],
                'IDs_evidencia':step.get('source_document_ids',[])} for i,step in enumerate(reading['steps'])]))
            display(pd.DataFrame([{'desde':link['from']+1,'hasta':link['to']+1,
                'relación':link['relation'],'lectura':link['reading']} for link in reading['links']]))
else: display(Markdown('Esta edición conserva únicamente el panel primario.'))
""")
    md("""
## 10. Recuperación y lectura del apoyo son tareas distintas

El benchmark usa preguntas originales EN/ES por familias. BM25, denso y RRF recuperan bloques; el contexto puede añadir vecinos de la misma canción. Las etiquetas de selección no entran en el índice. Recall, MRR y nDCG se refieren a los IDs designados por el analista; el corpus ampliado puede aportar evidencia adicional.

La descomposición reformulación × vecinos conserva identificables los brazos y sus presupuestos. Es una comparación exploratoria sobre las mismas familias. Se muestran resultados por modelo, método, brazo e idioma, sin convertir cada paráfrasis en una réplica independiente.
""")
    code("""
retrieval_tables = []
for path in sorted(RESULTS.glob('rag*/summary.csv')):
    table = frame(path)
    if not table.empty: retrieval_tables.append(table.assign(experimento=path.parent.name))
retrieval = pd.concat(retrieval_tables, ignore_index=True) if retrieval_tables else pd.DataFrame()
if not retrieval.empty:
    cols = ['experimento','method','arm','language','positive_queries','negative_queries',
            'recall_at_5','context_support_recall','negative_safe_abstention_rate']
    display(retrieval[retrieval.split=='test'][[c for c in cols if c in retrieval]])
figure('rag-recall-safety.png')
sensitivity = frame(RESULTS/'rag_family_sensitivity.csv')
if not sensitivity.empty: display(sensitivity)
""")
    md("""
## 11. Dos grafos LangGraph ejecutados

El grafo de recuperación de `rag.py` sigue `route → retrieve → assess`. La suficiencia puede llevar a reformulación acotada, contexto o abstención; reformular vuelve a recuperar. Contexto y paquete de respuesta pasan después por resolución de IDs. Las decisiones de ese controlador son reglas; el paquete conserva evidencia y presupuestos.

El grafo lector de `context_review.py` es otro: `attach_frozen_context → support_reader → citation_validation → answer/abstain`. Los lectores reciben la misma pregunta original y los mismos bloques congelados. No recuperan, reformulan ni eligen herramientas. Las declaraciones y votos de apoyo se registran por modelo.

La validación de cita verifica pertenencia al contexto. El guard de respuesta requiere además al menos una afirmación no vacía y ningún slot solicitado pendiente. Una cita existente no prueba entailment de una afirmación, y `supported` es una decisión del lector. El panel automatizado permite examinar discrepancias; no intercambiamos esas decisiones con exactitud de recuperación.

La [documentación de LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) distingue workflows acotados y agentes con decisión dinámica. Las ampliaciones de [Self-RAG](https://arxiv.org/abs/2310.11511), [CRAG](https://arxiv.org/abs/2401.15884) y [Adaptive-RAG](https://arxiv.org/abs/2403.14403) tienen mecanismos propios. La [recuperación contextual de Anthropic](https://www.anthropic.com/engineering/contextual-retrieval) incorpora contexto antes de indexar; añadir vecinos después de recuperar es otra operación.
""")
    code("""
figure('langgraph-workflow.png')
# Compila el grafo real con un callback local; no se invoca ningún lector ni API.
if str(STUDY) not in sys.path: sys.path.insert(0, str(STUDY))
import context_review
reader = context_review.reader_graph(lambda model_id,payload: {'status':'not_invoked'})
graph = reader.get_graph()
display(pd.DataFrame([{'origen':edge.source,'destino':edge.target,'condicional':edge.conditional}
                     for edge in graph.edges]))
print(graph.draw_mermaid())

context_dir = next((p for p in [STUDY/'pilot-comparison/results/llm-comparison-context',
    RESULTS/'llm-comparison-context'] if (p/'summary.json').exists()), RESULTS/'llm-comparison-context')
if (context_dir/'summary.json').exists():
    summary = read(context_dir/'summary.json')
    table = pd.DataFrame(public_rows(summary))
    columns = [c for c in ['model_id','requested','valid_reviews','positive_queries','negative_queries',
        'positive_abstentions','negative_safe_abstentions','positive_complete_contexts',
        'complete_context_abstentions','positive_incomplete_contexts','incomplete_context_abstentions',
        'invalid_citation_reviews'] if c in table]
    display(table[columns])
    paired = pd.DataFrame(summary.get('pairwise',[]))
    if not paired.empty:
        display(paired[['model_a','model_b','paired_valid_queries','support_decision_agreement',
            'final_abstention_agreement']])
        print('El acuerdo compara decisiones sobre preguntas emparejadas; no mide exactitud.')
    records = pd.DataFrame(read(context_dir/'runs.json'))
    if not records.empty:
        available = [c for c in ['model','query_id','family','language','support_decision','final_abstained',
            'citation_ids_valid','claim_count','missing_slot_count','context_words'] if c in records]
        display(records[available])
        display(records.groupby(['query_id','support_decision']).size().rename('votos_automáticos').reset_index())
""")
    md("""
## Apéndice A. Controles originales de alcance y formato

Los controles ficticios distinguen lo que una voz reporta de lo que un registro estipulado como autoritativo establece sobre un performer ficticio. Sus cuatro familias de slots se cruzan con tres alcances y permanecen congeladas. Su autoridad sólo existe dentro de ese escenario.

Los IDs de estos controles nombran el alcance: la prueba mide seguir una estipulación explícita, no autenticar documentos reales.

La revisión de formato V2 explicita cinco claves JSON para todos los modelos; OpenAI añade un esquema estricto y Anthropic sigue la restricción textual. Esa diferencia de interfaz permanece documentada. Las decisiones, las citas existentes y la fidelidad del contenido son objetos distintos. Las doce condiciones inglesas son controles de mecanismo, dependientes dentro de cada familia.
""")
    code("""
control_dir = RESULTS/'llm-comparison-detail-controls-v2'
if (control_dir/'summary.json').exists():
    summary = read(control_dir/'summary.json')
    table = pd.DataFrame(public_rows(summary))
    columns = [c for c in ['model_id','requested','valid','assistant_expected_matches','expected_constraints',
        'voice_report_supported','authoritative_supported','verified_missing_safe_abstentions',
        'supported_expected_id_cited','invalid_citation_reviews'] if c in table]
    display(table[columns])
    display(Markdown('La reparación de interfaz conserva los casos y sus expectativas; no constituye un conjunto independiente.'))
synthetic_dir = RESULTS/'llm-comparison-synthetic'
if (synthetic_dir/'summary.json').exists():
    summary = read(synthetic_dir/'summary.json')
    table = pd.DataFrame(public_rows(summary))
    columns = [c for c in ['model_id','requested','valid','core_assistant_constraints',
        'core_constraint_matches','coexistence_false_contradictions'] if c in table]
    display(table[columns])
    if summary.get('pairwise'): display(pd.DataFrame(summary['pairwise']))
""")
    md("""
## 12. Diagnóstico anterior: mapa del panel de 40

El mapa sitúa años de publicación y categorías de preocupación, con formas para los arcos. Las 23 referencias explícitas/implícitas aparecen separadas de los controles; no son 120 canciones anotadas. El eje vertical nombra categorías y no mide intensidad del miedo. Las curvas cuentan canciones con denominadores explícitos. El mapa interactivo está en `static/research/fear-lyrics/evolution-map.html`.

Los cinco ejemplos de valor subjetivo son lecturas editoriales de lo que se desea proteger o conservar. Las flechas ilustran pasos de un lector con anclajes válidos; el consenso corresponde al tipo de arco, no a cada estado.
""")
    code("""
focus = read(RESULTS/'historical-map/fear-focus-summary.json')
display(pd.DataFrame({'relevancia':list(focus['fear_reference_counts']),
                      'canciones':[focus['fear_reference_counts'][k] for k in focus['fear_reference_counts']]}))
display(pd.DataFrame({'arco_en_referencias_al_miedo':list(focus['arcs_fear_reference']),
                      'canciones':[focus['arcs_fear_reference'][k] for k in focus['arcs_fear_reference']]}))
assert focus['fear_reference_songs'] == 23 and sum(focus['arcs_fear_reference'].values()) == 23
for name in ['fear-reference-arcs.svg','historical-subtopics-map.svg',
             'historical-subtopics-presence.svg','narrative-value-evolution.svg']:
    figure(name)
""")
    md("""
## 13. Comparativa aplicada de los cuatro embeddings

Cada canción se representa por la media de bloques únicos normalizados y se normaliza el centroide. Los vecinos se calculan por coseno en el espacio completo de cada encoder. Se comparan identidades y rankings, no magnitudes de coseno entre espacios distintos.

El diagnóstico posterior de miedo restringe alternativas al mismo idioma y luego omite etiquetas sin resolver. Las barras tienen distintos denominadores de aristas utilizables. Su referencia uniforme está condicionada al idioma y a candidatos etiquetados. Concordancia con lecturas existentes no es exactitud de un clasificador, y un subtema frecuente puede explicar parte del acuerdo.
""")
    code("""
comparison = RESULTS/'encoder-comparison-new'
overlap = frame(comparison/'model_neighbor_overlap.csv')
display(overlap[overlap.k.eq(5)][['model_a','model_b','candidate_policy','mean_overlap_fraction']])
display(frame(comparison/'model_neighbor_diagnostics.csv'))
display(frame(comparison/'fear_neighbor_checks.csv')[['model','metric','retained_labeled_directed_edges',
    'observed_matching_edges','observed_fraction','uniform_conditional_expected_fraction']])
for name in ['encoder-comparison-neighborhoods.svg','encoder-comparison-language-control.svg',
             'encoder-comparison-cases.svg','encoder-comparison-fear-checks.svg']:
    figure(name)
""")
    md("""
## 14. Bayes legible: recuento, estimación y tres varianzas distintas

La comparación inspirada en Julia Silge pone `c/n` en horizontal y la estimación posterior en vertical. La diagonal indica igualdad. Se conservan todas las celdas y los ceros; los tópicos no están alineados entre encoders.

La varianza temporal `Var(c/n)` describe cinco recuentos observados (ddof=0). La varianza posterior `Var(θ|datos)` proviene de momentos Beta o de la ley de varianza total sobre la mezcla jerárquica. La varianza de embeddings depende de sus coordenadas y se estudia mediante PCA. El ancho de un intervalo no se usa como sustituto de varianza.
""")
    code("""
variance_receipt = read(FIGURES/'bayes-readable-receipt.json')
display(pd.DataFrame([{'cohorte':r['cohort'],'canciones_por_celda':r['songs_per_cell'],
    'filas_posteriores':r['posterior_rows'],'error_maximo_media':r['moments_mean_max_absolute_disagreement_with_executed_posterior']}
    for r in variance_receipt['cohorts']]))
for cohort in ['matched40','expanded120']:
    for kind in ['shrinkage','priors','variance']:
        figure('bayes-readable-'+kind+'-'+cohort+'.svg')
""")
    md("""
## 15. LangGraph: recorrido observado de la consulta

El esquema y las trazas muestran decisiones realmente guardadas: recuperación, gate léxico, hasta dos reformulaciones, contexto limitado y paquete de IDs o abstención. Los vecinos se incorporan por posición y presupuesto. El grafo de recuperación no produce por sí solo una lectura literaria.

La lectura mini/nano usa otros contextos híbridos congelados; su figura conserva esa fase separada. Recuperar soporte, reconocer insuficiencia y resolver la interpretación se examinan como tareas distintas.
""")
    code("""
process_receipt = read(RESULTS/'langgraph-process-new/receipt.json')
display(Markdown('Recibo del proceso: `results/langgraph-process-new/receipt.json`. Trazas y presupuestos auditados sin nuevas APIs.'))
for name in ['langgraph-process-overview.svg','langgraph-process-traces.svg','langgraph-process-reader.svg']:
    figure(name)
""")
    md("""
## 16. Revisión anual uniforme: 120 canciones y 240 lecturas

El protocolo fijó referencia, polaridad, voz, respuesta, hasta dos motivos y pasos apoyados antes de revisar lotes sin título, artista ni fecha. Tres procesos del mismo asistente configurado leyeron 80 casos cada uno; cada canción recibió dos lecturas. No son proveedores independientes. Un motivo cuenta cuando ambas lecturas apoyan miedo y ese motivo; los casos inciertos permanecen en el denominador de canciones seleccionadas.

El mapa anual conserva 47 años, con seis sin observaciones. Las proporciones de años con pocas canciones se presentan con n; no se interpolan huecos. La comparación de extremos equilibra idiomas y conserva sensibilidad de una lectura y de retiro de artista. Los arcos sólo entran al subconjunto con referencia apoyada y siguen conservando los desacuerdos.

Las nuevas comprobaciones de embeddings reutilizan matrices originales: cuatro encoders sobre los mismos 40 casos y dos locales sobre 120, con consultas compartidas por métrica e idioma controlado. La tasa bayesiana estima la detección consensuada de miedo y motivo entre todas las canciones de cada celda; no la prevalencia histórica de miedo. Jeffreys y uniforme son priors fijos y los intervalos no integran incertidumbre de categorías ni selección.
""")
    code("""
ANNUAL = RESULTS/'annual-120'
annual = read(ANNUAL/'summary.json')
annotations = pd.DataFrame(read(ANNUAL/'annotations.json'))
display(pd.DataFrame([{'referencia':k,'canciones':v} for k,v in annual['fear_reference_counts'].items()]))
display(frame(ANNUAL/'periods.csv'))
display(frame(ANNUAL/'endpoint_sensitivity.csv'))
assert annual['songs']==120 and annual['readings']==240
assert annual['schema_invalid_readings']==annual['arc_structure_invalid_readings']==0
assert annotations.song_id.is_unique
assert len(read(ANNUAL/'votes.json'))==2*len(annotations)
assert sum(annual['arcs_in_fear'].values())==53
print('Años sin canciones seleccionadas:', annual['empty_years'])
print('Acuerdo de presencia con el panel anterior:', annual['prior_panel_presence_agreement'], '/40')
embedding_checks = frame(ANNUAL/'embeddings/neighbor_checks.csv')
if not embedding_checks.empty: display(embedding_checks)
posterior_rates = frame(ANNUAL/'bayes/annual-bayes-posterior-rates.csv')
endpoint_posteriors = frame(ANNUAL/'bayes/annual-bayes-endpoint-posteriors.csv')
display(frame(ANNUAL/'bayes/annual-bayes-variance.csv'))
display(endpoint_posteriors)
for name in ['annual-subtopics-map.svg','annual-arcs-stances.svg','annual-argumental-cases.svg',
             'annual-embeddings-neighbor-checks.svg','annual-bayes-priors.svg',
             'annual-bayes-endpoints.svg','annual-bayes-variance.svg']:
    assert (FIGURES/name).exists(), name
    figure(name)
""")
    md("""
## 17. Conclusiones y trazabilidad

Las cuatro preguntas de lectura siguen separadas: qué relaciones preserva cada embedding, qué bloques agrupa cada tópico, qué casos componen cada período y qué afirmación sostiene el contexto recuperado. El espectro PCA cuantifica lo que queda fuera del mapa; los cambios de posterior explicitan dependencia del prior; los votos del panel dejan visibles interpretaciones y discrepancias.

La contribución es vincular búsqueda semántica, lectura del miedo y comparación temporal de preocupaciones y respuestas. Los vecindarios de OpenAI muestran concordancia con subtemas por encima de la referencia uniforme del diagnóstico controlado por idioma; sus centroides no conservan posiciones de bloques, de modo que los arcos requieren volver al orden y a las relaciones argumentales.

En la revisión anual de 53 referencias apoyadas hay 14 repeticiones, ocho confrontaciones, cuatro contrastes, cuatro giros, tres relaciones mixtas y veinte arcos inciertos. Fragilidad aparece en cuatro ventanas; coerción, en cinco. Los 28 intervalos estrictos de diferencias entre extremos por motivo, idioma y prior incluyen cero. La evidencia sostiene recurrencias y contrastes de historias, sin una dirección histórica uniforme.

Vigilancia en Every Breath You Take, desafío en Malo, salida en La Puerta Violeta y recuperación en Ansiedad sitúan distintas voces frente a amenazas relacionadas. Aperture queda incierta en la nueva revisión: la historia anterior no se conserva como hecho. Los valores son interpretaciones apoyadas de qué protege una voz, no puntuaciones ni afirmaciones de novedad histórica. Audio, mercados y respuesta de oyentes requieren preguntas y diseños propios.
""")
    code("""
# Comprobaciones de resultados y denominadores; no se imprimen letras.
assert all(r['results_status']=='executed' for r in model_receipts.values())
assert diag['songs_with_lyrics'] <= len(manifest)
assert set(manifest.language) == {'en','es'}
for directory in modeldirs:
    composition = frame(directory/'decade_composition.csv')
    if not composition.empty:
        totals = composition.groupby(['variant','decade','language']).song_normalized_share.sum()
        assert np.allclose(totals,1), 'Las proporciones no conservan el denominador.'
    projection_file = directory/'projections.json'
    if projection_file.exists():
        p = read(projection_file); r = p['receipt']
        assert r['seed'] is None and r['solver']=='full'
        assert len(p['points']) == r['input_documents']
        assert len(r.get('scree_explained_variance_ratio',r['explained_variance_ratio'])) == min(20,r['input_documents'],r['input_dimensions'])
        assert np.isclose(sum(r['explained_variance_ratio']),r['explained_variance_total'])
    print(directory.name,'✓ recibos, proporciones y proyección')
print('Resultados guardados inspeccionados. Las celdas no ejecutaron APIs ni descargaron letras.')
""")
    md("""
## 18. Recalcular con el corpus privado autorizado

Esta celda permanece desactivada por defecto. Requiere una caché privada cuyo corpus coincida con el manifiesto final, pesos locales y bundles API ya calculados. No descarga letras ni genera nuevas llamadas de embeddings o de lectura. `FEAR_PRIVATE_CACHE` y `FEAR_PRIVATE_CORPUS` permiten señalar la caché de la corrida final.
""")
    code(r"""
if RERUN_EXPERIMENTS:
    default_corpus = PRIVATE/'pilot-40/corpus.jsonl' if len(manifest)==40 else PRIVATE/'corpus.jsonl'
    corpus_path = Path(os.environ.get('FEAR_PRIVATE_CORPUS', default_corpus))
    if not corpus_path.exists(): raise FileNotFoundError('Se requiere el corpus privado autorizado de esta corrida.')
    private_rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    private_ids = {r['song_id'] for r in private_rows}
    assert private_ids <= set(manifest.song_id) and len(private_ids) == diag['songs_with_lyrics'], 'El corpus privado no coincide con esta edición.'
    bundles = []
    for path in sorted(PRIVATE.rglob('*.npz'), key=lambda p:(p.parent.name!='pilot-comparison',str(p))):
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            metadata = read(sidecar)
            if set(metadata.get('document_ids',[])) == set(inventory[inventory.document_id.eq(inventory.canonical_id)].document_id):
                bundles.append((path,sidecar,metadata.get('model_id')))
    projection_bundles = []
    for directory in modeldirs:
        encoder = model_receipts[directory.name]['encoder']; model_id = encoder['model_id']
        match = next((b for b in bundles if b[2] in [model_id,model_id.split('/',1)[-1]]),None)
        if match is not None: projection_bundles += ['--bundle',model_id+'='+str(match[0])]
        command = [sys.executable,str(STUDY/'analyze.py'),'--input',str(corpus_path),'--output',str(RESULTS),
                   '--private-cache',str(PRIVATE),'--model-cache',str(PRIVATE/'model-cache'),'--offline']
        if encoder.get('input_type') == 'verified_external_embedding_bundle':
            if match is None: raise FileNotFoundError('No existe un bundle local verificado para '+model_id)
            command += ['--embedding-file',str(match[0]),'--embedding-metadata',str(match[1]),'--model-label',model_id]
        else: command += ['--models',model_id]
        subprocess.run(command,cwd=ROOT,check=True)
    ids = [r['encoder']['model_id'] for r in model_receipts.values()]
    subprocess.run([sys.executable,str(STUDY/'projections.py'),'--results',str(RESULTS),
                    '--manifest',str(STUDY/'corpus-manifest.json'),'--cache',str(PRIVATE),*projection_bundles],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(STUDY/'bayes.py'),'--results',str(RESULTS),
                    '--private-cache',str(PRIVATE),'--models',*ids],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(STUDY/'evaluate.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(STUDY/'visualize.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(STUDY/'atlas_v2.py')],cwd=ROOT,check=True)
else: print('Reejecución desactivada: se conservaron resultados y presupuestos.')
""")
    appendix_index = next(i for i,c in enumerate(cells) if c.cell_type == 'markdown' and c.source.startswith('## Apéndice A.'))
    appendix = cells[appendix_index:appendix_index+2]
    del cells[appendix_index:appendix_index+2]
    cells.extend(appendix)
    notebook = nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3 (study environment)','language':'python'},
        'language_info':{'name':'python','version':'3.12'},
        'study':{'cutoff':'2026-10-11','language':'es','privacy':'metadata_only_no_full_lyrics',
                 'selected_songs':len(manifest),'common_comparison_songs':len(common_manifest),'expanded_model_count':len(expanded_dirs),'embedding_models':[r['encoder']['model_id'] for r in receipts],
                 'model_count':len(dirs),'execution':'saved_results_no_remote_calls_by_default'}})
    nbf.validate(notebook)
    destination = Path(output) if output else HERE/'anexo-auditoria-metodologica.ipynb'
    nbf.write(notebook,destination)
    print(json.dumps({'notebook':str(destination),'selected_songs':len(manifest),
                      'models':len(dirs),'common_comparison_songs':len(common_manifest),'expanded_models':len(expanded_dirs),'cells':len(cells),'executed':False},ensure_ascii=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expected-songs',type=int,help='Expected songs in the common encoder result cohort, not the full collection manifest.')
    parser.add_argument('--expected-models',type=int)
    args = parser.parse_args()
    build(args.output,args.expected_songs,args.expected_models)
