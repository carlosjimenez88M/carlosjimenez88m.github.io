"""Historical views of existing categorical votes; no inference or network."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import html
import json
import shutil
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import visualize as v

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / 'pilot-comparison/results/arc-panel/fieldwise-posthoc'
OUT = ROOT / 'results/historical-map'
PERIODS = [1980, 1990, 2000, 2010, 2020]
PERIOD_LABELS = ['1980–89', '1990–99', '2000–09', '2010–19', '2020–26*']
COLORS = ['#d47840', '#b49318', '#3c9670', '#4282b5', '#96588c']
VIEWS = {'fear_paired_joint_votes': 'paired_joint_vote_subtopics',
         'fear_consensus_cross': 'fear_conditioned_subtopics',
         'all_concerns': 'consensus'}
ARCS = {'repetition': 'Repetición', 'turn': 'Giro', 'mixed': 'Relación mixta',
        'unresolved': 'Arco sin resolver', 'no_supported_arc': 'Sin arco apoyado'}
VALUE_READINGS = {
    "Somebody's Watching Me": 'Seguridad e intimidad',
    'Rayando el Sol': 'Vínculo afectivo',
    'A Dios Le Pido': 'Cuidado de vínculos',
    'Out Of The Woods': 'Seguridad del vínculo',
    'Aperture': 'Seguridad y unión',
}
STATES = {'exposure': 'Exposición', 'fear': 'Miedo', 'avoidance': 'Evitación',
          'threat': 'Amenaza', 'loss': 'Pérdida', 'agency': 'Acción',
          'reflection': 'Reflexión', 'care': 'Cuidado', 'unclear': 'Estado incierto',
          'celebration': 'Celebración', 'negation': 'Negación'}


def read(path):
    return json.loads(path.read_text())


def labels(row, view):
    if view == 'all_concerns':
        return row['consensus']['subtopics']
    return row[VIEWS[view]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT / 'corpus-manifest.json')
    meta = {r['song_id']: r for r in manifest}
    rows = read(PANEL / 'song_annotations.json')
    votes = read(PANEL / 'categorical_votes.json')
    cases = read(PANEL / 'anchored_cases.json')
    ontology = read(PANEL.parent / 'ontology.json')['subtopics']
    keys = [key for key in ontology if key not in ['none', 'uncertain', 'other']]
    assert len(rows) == 40
    assert Counter((r['decade'], r['language']) for r in rows) == Counter({
        (period, lang): 4 for period in PERIODS for lang in ['en', 'es']})
    song_rows = []
    for row in rows:
        item = dict(row)
        item.update({key: meta[row['song_id']][key] for key in ['title', 'artist']})
        supporting = {}
        for key in keys:
            supporting[key] = sum(
                key in vote['categorical_votes'].get('subtopics', []) and
                vote['categorical_votes'].get('fear_relevance') in ['explicit', 'implied']
                for vote in votes if vote['song_id'] == row['song_id'])
        item['joint_support_votes'] = supporting
        item['editorial_value_reading'] = VALUE_READINGS.get(item['title'])
        song_rows.append(item)
    counts = []
    for view in VIEWS:
        for language in ['balanced', 'en', 'es']:
            for period in PERIODS:
                group = [r for r in song_rows if r['decade'] == period and
                         (language == 'balanced' or r['language'] == language)]
                for key in keys:
                    count = sum(key in labels(r, view) for r in group)
                    counts.append(dict(view=view, language=language, decade=period,
                                       subtopic=key, label=ontology[key], count=count,
                                       songs=len(group), share=count / len(group)))
    # Check independently against the existing, frozen posthoc counting rules.
    reference = list(csv.DictReader((PANEL / 'concern_presence_by_period.csv').open()))
    lookup = {(r['view'], r['language'], int(r['decade']), r['value']): int(r['count'])
              for r in reference}
    assert all(r['count'] == lookup[(r['view'], r['language'], r['decade'], r['subtopic'])]
               for r in counts)
    with (OUT / 'presence.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader(); writer.writerows(counts)
    value_cases = []
    for case in sorted(cases, key=lambda r: r['year']):
        if case['title'] not in VALUE_READINGS:
            continue
        example = case['readings'][0]
        value_cases.append(dict(song_id=case['song_id'], title=case['title'],
                                artist=case['artist'], year=case['year'],
                                arc=case['consensus']['arc'],
                                supporting_votes=case['supporting_vote_count'],
                                editorial_value=VALUE_READINGS[case['title']],
                                example_reader=case['models'][0],
                                steps=example['steps'], links=example['links']))
    fear_rows = [r for r in song_rows if r['consensus']['fear_relevance'] in ['explicit', 'implied']]
    focus_summary = {
        'selected_songs': len(song_rows), 'fear_reference_songs': len(fear_rows),
        'fear_reference_counts': dict(Counter(r['consensus']['fear_relevance'] for r in song_rows)),
        'arcs_all_selected': dict(Counter(r['consensus']['arc'] for r in song_rows)),
        'arcs_fear_reference': dict(Counter(r['consensus']['arc'] for r in fear_rows)),
        'polarity_fear_reference': dict(Counter(r['consensus']['fear_polarity'] for r in fear_rows)),
        'fear_reference_by_period': [{
            'decade': period, 'selected_songs': 8,
            'fear_reference_songs': sum(r['decade'] == period for r in fear_rows),
            'arcs_fear_reference': dict(Counter(r['consensus']['arc'] for r in fear_rows if r['decade'] == period))}
            for period in PERIODS],
        'definition': 'Existing field consensus explicit/implied; includes denied/hypothetical polarity. No new model calls.',
        'arc_definition': 'Strict chronological category consensus; unresolved remains in denominator.',
    }
    assert len(fear_rows) == 23
    (OUT / 'fear-focus-summary.json').write_text(json.dumps(focus_summary, ensure_ascii=False, indent=2) + '\n')
    payload = {'schema_version': 1, 'cutoff': '2026-10-11', 'songs': song_rows,
               'corpus': [{k: r[k] for k in ['song_id', 'title', 'artist', 'year', 'language', 'decade']}
                          for r in manifest],
               'ontology': {k: ontology[k] for k in keys}, 'presence': counts,
               'value_cases': value_cases, 'fear_focus': focus_summary, 'periods': PERIODS,
               'period_labels': PERIOD_LABELS, 'colors': COLORS,
               'scope': '40 songs with votes, nested in 120-song corpus. Not 120 annotated songs.',
               'value_scope': 'Five editorial interpretations, not model scores or historical novelty labels.'}
    (OUT / 'evolution-data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    render_static(payload, keys)
    render_html()
    for name in ['historical-subtopics-map', 'historical-subtopics-presence', 'narrative-value-evolution', 'fear-reference-arcs']:
        for suffix in ['svg', 'png']:
            shutil.copy2(v.FIG / f'{name}.{suffix}', v.SITE / f'{name}.{suffix}')
    shutil.copy2(OUT / 'evolution-data.json', v.SITE / 'evolution-data.json')
    receipt = {
        'status': 'executed_from_existing_votes', 'network_calls': 0,
        'songs_annotated': 40, 'corpus_songs': len(manifest), 'year_axis': [1980, 2026],
        'years_observed': sorted({r['year'] for r in song_rows}),
        'line_rule': 'Period aggregates only; no interpolation of absent years or connecting individual songs.',
        'primary_view': 'fear_consensus_cross_only_explicit_implied_songs',
        'point_area': 'Equal; no intensity scale. Interactive paired-joint sensitivity can show supporting votes.',
        'categorical_y': 'Named concerns, no intensity or utility scale.',
        'arc_shape': 'Strict chronological consensus; unresolved retained.',
        'editorial_value_cases': len(value_cases),
        'inputs': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in [PANEL / 'song_annotations.json', PANEL / 'categorical_votes.json',
                             PANEL / 'anchored_cases.json', ROOT / 'corpus-manifest.json']],
        'figures': ['historical-subtopics-map', 'historical-subtopics-presence', 'narrative-value-evolution', 'fear-reference-arcs'],
    }
    (OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'songs_with_votes': len(song_rows), 'editorial_value_cases': len(value_cases),
                      'figures': receipt['figures'], 'interactive': 'evolution-map.html'}))


def render_static(data, keys):
    songs = data['songs']; ylabels = [data['ontology'][k] for k in keys] + ['Subtema sin resolver', 'Ausente o referencia sin resolver']
    fig, ax = plt.subplots(figsize=(13.7, 7.4))
    for j, period in enumerate(PERIODS):
        ax.axvspan(period, min(period + 10, 2026.7), color=COLORS[j], alpha=.07)
    symbols = {'repetition': 'o', 'turn': '^', 'mixed': 'D'}
    for row in songs:
        in_focus = row['consensus']['fear_relevance'] in ['explicit', 'implied']
        chosen = [k for k in labels(row, 'fear_consensus_cross') if k in keys] if in_focus else []
        if not chosen:
            ax.scatter(row['year'], len(keys) + (not in_focus), s=30, marker='x', color='#84929c', alpha=.65)
        for k in chosen:
            y = keys.index(k)
            marker = symbols.get(row['consensus']['arc'], 'x')
            ax.scatter(row['year'], y, s=90, marker=marker,
                       color=COLORS[PERIODS.index(row['decade'])], alpha=.82, zorder=3)
        if row['title'] in VALUE_READINGS and chosen:
            y = keys.index(chosen[0]); offset = 14 if row['title'] != 'A Dios Le Pido' else -24
            label = f"{row['title']} · {row['year']}"
            ax.annotate(label, (row['year'], y), xytext=(3, offset), textcoords='offset points',
                        fontsize=8.5, ha='right' if row['year'] > 2018 else 'left',
                        arrowprops=dict(arrowstyle='-', lw=.7, color='#596575'))
    # Show acquisition coverage as a separate rug, not as annotated subtopics.
    for r in data['corpus']:
        ax.plot(r['year'], len(keys) + 1.65, '|', color='#667380', alpha=.28)
    ax.text(1980.4, len(keys) + 1.87, 'Rug inferior: años de las 120 canciones recuperadas', fontsize=8, color='#596575')
    ax.set(xlim=(1979.7, 2027), ylim=(len(keys) + 2.12, -.8),
           yticks=range(len(ylabels)), yticklabels=ylabels,
           xticks=[1980, 1990, 2000, 2010, 2020, 2026],
           xlabel='Año de primera publicación de la grabación',
           title='Mapa histórico del miedo: 23 referencias explícitas o implícitas (1980–2026)')
    handles = [plt.Line2D([], [], marker=s, linestyle='', color='#3d5468', label=t, markersize=7)
               for s, t in [('o', 'Repetición'), ('^', 'Giro'), ('D', 'Relación mixta'), ('x', 'Arco sin resolver')]]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.12), ncol=4, frameon=False, fontsize=9)
    ax.grid(axis='x', alpha=.2); ax.grid(axis='y', alpha=.13)
    fig.tight_layout(rect=(0, .12, 1, 1))
    v.save(fig, 'historical-subtopics-map',
           '23 referencias al miedo con consenso por campo; otros 17 casos en gris. Igual tamaño; categorías solapadas. Color: período; *2020–26 parcial.')
    fig, axes = plt.subplots(4, 2, figsize=(12.9, 11), sharex=True, sharey=True)
    for ax, key in zip(axes.flat, keys + ['fear_reference']):
        if key == 'fear_reference':
            group = [sum(r['consensus']['fear_relevance'] in ['explicit', 'implied']
                         for r in songs if r['decade'] == period) for period in PERIODS]
            title = 'Miedo nombrado o implícito (consenso por campo)'
        else:
            group = [next(r['count'] for r in data['presence'] if r['view'] == 'fear_paired_joint_votes'
                          and r['language'] == 'balanced' and r['decade'] == period and r['subtopic'] == key)
                     for period in PERIODS]
            title = data['ontology'][key]
        values = np.array(group) / 8
        ax.plot(range(5), values, color='#425f78', linewidth=1.4)
        ax.scatter(range(5), values, s=64, c=COLORS, zorder=3)
        for j, n in enumerate(group):
            ax.annotate(f'{n}/8', (j, values[j]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
        ax.axvspan(3.7, 4.3, color=COLORS[-1], alpha=.06)
        ax.set_title(title, loc='left', fontsize=10, weight='bold')
        ax.set(ylim=(-.05, 1.05), yticks=[0, .25, .5, .75, 1],
               yticklabels=['0 %', '25 %', '50 %', '75 %', '100 %'],
               xticks=range(5), xticklabels=PERIOD_LABELS)
        ax.grid(alpha=.14)
    fig.suptitle('Qué subtemas reaparecen: presencia por canción, con denominador visible', fontsize=14, weight='bold')
    fig.tight_layout(rect=(0, .06, 1, .97))
    v.save(fig, 'historical-subtopics-presence',
           'Subtemas: dos votos que vinculan motivo y miedo. Último panel: consenso de relevancia. Líneas unen períodos observados, sin suavizado ni inferencia poblacional.')
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 9.2))
    for ax, case in zip(axes, data['value_cases']):
        ax.axis('off'); steps = case['steps']; count = len(steps)
        ax.text(-.03, .65, f"{case['year']} · {case['title']}", transform=ax.transAxes,
                fontsize=10, weight='bold', color=COLORS[PERIODS.index(case['year'] // 10 * 10)])
        ax.text(-.03, .4, case['editorial_value'], transform=ax.transAxes, fontsize=9)
        ax.text(-.03, .12, f"{ARCS[case['arc']]} · {case['supporting_votes']} votos", transform=ax.transAxes, fontsize=8, color='#5e6c79')
        locations = np.linspace(.38, .87, count)
        for j, (x, step) in enumerate(zip(locations, steps)):
            ax.text(x, .48, STATES.get(step['state'], step['state']) + '\n' +
                    '\n'.join(textwrap.wrap(step['reading'], 26)), transform=ax.transAxes,
                    ha='center', va='center', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=.7', fc='#edf3f6', ec='#aabac5'))
            if j < count - 1:
                ax.annotate('', xy=(locations[j+1]-.07, .48), xytext=(x+.07, .48), xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='#5e7386', lw=1.2))
    fig.suptitle('Tiempo histórico y tiempo argumental: qué está en juego para la voz', fontsize=14, weight='bold')
    fig.tight_layout(rect=(.04, .1, 1, .96))
    v.save(fig, 'narrative-value-evolution',
           'Cinco casos ilustrativos. Valor: lectura editorial. Pasos: ejemplo de un lector con anclajes válidos; el consenso corresponde al tipo de arco, no a todos los estados.')
    focus = data['fear_focus']
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.7))
    for ax, mapping, labels_map, denominator, title in [
        (axes[0], focus['fear_reference_counts'], {'explicit': 'Referencia explícita', 'implied': 'Referencia implícita',
         'absent': 'Ausencia de referencia', 'unresolved': 'Sin resolución'}, 40, '¿La letra refiere al miedo?'),
        (axes[1], focus['arcs_fear_reference'], {'repetition': 'Repetición', 'turn': 'Giro',
         'mixed': 'Relación mixta', 'unresolved': 'Sin arco resuelto'}, 23, '¿Cómo se desarrolla la referencia al miedo?')]:
        keys_here = list(labels_map); values = [mapping.get(k, 0) for k in keys_here]
        ax.barh(range(len(keys_here)), values, color=['#4282b5', '#3c9670', '#96588c', '#a6afb6'])
        for j, n in enumerate(values): ax.text(n + .3, j, f'{n}/{denominator}', va='center', fontsize=11)
        ax.set(yticks=range(len(keys_here)), yticklabels=list(labels_map.values()), xlim=(0, denominator*.78),
               xlabel='Canciones', title=title)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.15)
    fig.tight_layout(rect=(0, .15, 1, .98))
    v.save(fig, 'fear-reference-arcs',
           'Izquierda: 40 casos. Derecha: sólo 23 referencias explícitas/implícitas; incluye negaciones y posibilidades. Arcos conservan anclajes y orden estrictos.')


def render_html():
    page = r'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mapa histórico del miedo · canciones, subtemas y arcos</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#243747;background:#f5f7f8;margin:0}main{max-width:1360px;margin:auto;padding:24px}h1{font-size:clamp(26px,4vw,40px);line-height:1.16}h2{font-size:24px}p{max-width:950px}.intro,.panel{background:white;border:1px solid #d9e1e5;border-radius:14px;padding:22px;margin:18px 0}.muted{color:#5b6d79;font-size:14px}.controls{display:flex;gap:18px;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:6px}select{font:inherit;padding:8px;max-width:400px}button{font:inherit;padding:10px}.chart{width:100%;min-height:600px}#presence{min-height:420px}.reading{border-left:4px solid #438184;padding:0 14px;margin:20px 0}.tag{display:inline-block;background:#edf3f6;padding:3px 8px;border-radius:4px;margin:3px;font-size:13px}a{color:#24597b}@media(max-width:600px){main{padding:12px}.intro,.panel{padding:14px}.chart{min-height:720px}}</style>
<script src="plotly.min.js"></script><main><a href="/post/2026-10-11-miedo-embeddings-bertopic-rag/">← Volver al artículo</a>
<section class="intro"><p class="muted">1980–2026 · corte 11/10/2026 · EN / ES</p><h1>Qué permanece en el miedo y qué cambia en la historia</h1>
<p>El eje horizontal sitúa la publicación. Las filas nombran preocupaciones; la forma distingue arcos con consenso de casos sin resolución. Haz clic en una canción para revisar qué está en juego y qué pasos están apoyados.</p>
<p><b>40 canciones tienen votos interpretativos.</b> Forman parte del corpus de 120 letras. Los años restantes aparecen como cobertura del corpus, sin adjudicarles subtemas que no se evaluaron.</p>
<div class="controls"><label>Lectura de los subtemas<select id="view"><option value="fear_paired_joint_votes">Motivo y miedo en los mismos votos</option><option value="fear_consensus_cross">Consenso de motivo y relevancia por separado</option><option value="all_concerns">Todas las preocupaciones (incluye ausencia de miedo)</option></select></label><label>Idioma<select id="language"><option value="balanced">Inglés + español</option><option value="en">Inglés</option><option value="es">Español</option></select></label><label>Etiquetas<select id="names"><option value="selected">Cinco casos ilustrativos</option><option value="all">Todas las canciones</option><option value="none">Sin etiquetas</option></select></label></div>
<p class="muted">Área: número de votos conjuntos cuando esa vista está seleccionada; las demás usan puntos de igual tamaño. Acuerdo no es probabilidad de corrección. Un caso puede aparecer en dos filas. Las categorías del eje vertical no tienen orden de intensidad.</p></section>
<section class="panel"><div id="timeline" class="chart"></div></section>
<section class="panel"><h2>Presencia por período</h2><p>Cada curva cuenta canciones de la selección. El denominador incluye casos sin resolución. Activa o desactiva un subtema en la leyenda; los años vacíos no se interpolan.</p><div id="presence" class="chart"></div><p id="denominator" class="muted"></p></section>
<section id="inspector" class="panel"><h2>Una canción, una preocupación, una respuesta</h2><p>Selecciona un punto en el mapa. Las lecturas de valor subjetivo aparecen en cinco casos como interpretación editorial, junto a pasos reales de uno de los lectores.</p></section>
<section class="panel"><h2>Los dos tiempos, juntos</h2><img src="narrative-value-evolution.svg" alt="Cinco canciones en orden histórico, con lecturas de valor y pasos argumentales anclados" style="width:100%"><p class="muted">La secuencia de canciones compara casos publicados en épocas distintas. Las flechas dentro de cada fila siguen pasos de esa canción; no establecen una genealogía o una relación causal entre artistas.</p><p><a href="atlas.html">Comparar los cuatro encoders</a> · <a href="expanded/atlas.html">Explorar las 120 canciones</a> · <a href="/examples/fear-lyrics-study.zip">Notebook y materiales</a></p></section></main>
<script>
const $=id=>document.getElementById(id);let D;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const arcLabels={repetition:'Repetición',turn:'Giro',mixed:'Relación mixta',unresolved:'Arco sin resolver',no_supported_arc:'Sin arco apoyado'};
const symbols={repetition:'circle',turn:'triangle-up',mixed:'diamond'};
function selected(){return D.songs.filter(r=>$('language').value==='balanced'||r.language===$('language').value)}
function topics(r){return $('view').value==='all_concerns'?r.consensus.subtopics:($('view').value==='fear_consensus_cross'?r.fear_conditioned_subtopics:r.paired_joint_vote_subtopics)}
function render(){const songs=selected(),keys=Object.keys(D.ontology),x=[],y=[],colors=[],sizes=[],shapes=[],texts=[],hover=[],ids=[];for(const r of songs){let ts=topics(r).filter(t=>keys.includes(t));if(!ts.length)ts=['unresolved'];for(const t of ts){x.push(r.year);y.push(t==='unresolved'?'Sin motivo resuelto':D.ontology[t]);colors.push(D.colors[D.periods.indexOf(r.decade)]);sizes.push($('view').value==='fear_paired_joint_votes'&&t!=='unresolved'?8+3*r.joint_support_votes[t]:10);shapes.push(symbols[r.consensus.arc]||'x');texts.push($('names').value==='all'||($('names').value==='selected'&&r.editorial_value_reading)?esc(r.title):'');hover.push(esc(r.title)+' · '+esc(r.artist)+'<br>'+r.year+' · '+r.language.toUpperCase()+'<br>'+esc(t==='unresolved'?'Sin motivo resuelto':D.ontology[t])+'<br>'+esc(arcLabels[r.consensus.arc]||r.consensus.arc)+'<br>Relevancia: '+esc(r.consensus.fear_relevance));ids.push(r.song_id)}}
const rug=D.corpus.filter(r=>$('language').value==='balanced'||r.language===$('language').value);
Plotly.react('timeline',[{type:'scatter',mode:'markers+text',x,y,text:texts,textposition:'top center',textfont:{size:10},marker:{color:colors,size:sizes,symbol:shapes,opacity:.8},customdata:ids,hovertext:hover,hoverinfo:'text',name:'Canciones con votos'},{type:'scatter',mode:'markers',x:rug.map(r=>r.year),y:rug.map(r=>'Cobertura del corpus'),marker:{symbol:'line-ns',size:9,color:'#a6b1b7'},hovertext:rug.map(r=>esc(r.title)+' · '+r.year+'<br>Metadatos de cobertura; no es una anotación de subtema'),hoverinfo:'text',name:'120 canciones recuperadas'}],{margin:{l:185,r:25,t:25,b:70},height:630,showlegend:false,xaxis:{title:'Año de publicación',range:[1979.5,2027],tickvals:[1980,1990,2000,2010,2020,2026]},yaxis:{categoryorder:'array',categoryarray:['Cobertura del corpus','Sin motivo resuelto',...Object.values(D.ontology).reverse()],automargin:true},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false});
const trace=keys.map((k,i)=>{const rows=D.presence.filter(r=>r.view===$('view').value&&r.language===$('language').value&&r.subtopic===k).sort((a,b)=>a.decade-b.decade);return{type:'scatter',mode:'lines+markers',x:rows.map(r=>D.period_labels[D.periods.indexOf(r.decade)]),y:rows.map(r=>r.share),name:D.ontology[k],hovertext:rows.map(r=>r.count+'/'+r.songs+' canciones'),hovertemplate:'%{x}<br>%{hovertext}<extra>%{fullData.name}</extra>',line:{width:2},marker:{size:8}}});
Plotly.react('presence',trace,{margin:{l:65,r:20,t:15,b:100},height:420,yaxis:{title:'Presencia en la selección',range:[-.02,1.02],tickformat:'.0%'},legend:{orientation:'h',y:-.2},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displaylogo:false});$('denominator').textContent='Denominador: '+(songs.length/5)+' canciones por período. Las categorías se solapan; las tasas no suman uno. 2020–26 es parcial.';}
function showSong(id){const r=D.songs.find(r=>r.song_id===id),c=D.value_cases.find(r=>r.song_id===id);if(!r)return;const box=$('inspector');box.replaceChildren();function add(tag,text,cls){const el=document.createElement(tag);el.textContent=text;if(cls)el.className=cls;box.append(el);return el}add('h2',r.title+' · '+r.year);add('p',r.artist+' · '+r.language.toUpperCase());add('p','Subtemas mostrados: '+topics(r).map(t=>D.ontology[t]||'Sin motivo resuelto').join(' / '));add('p','Voz: '+r.consensus.voice+' · postura: '+r.consensus.stance+' · miedo: '+r.consensus.fear_relevance+' · polaridad: '+r.consensus.fear_polarity,'muted');add('p','Arco: '+(arcLabels[r.consensus.arc]||r.consensus.arc));if(c){add('h3','Valor en juego: '+c.editorial_value);add('p','Lectura editorial, no puntuación automática. Tipo de arco apoyado por '+c.supporting_votes+' votos; pasos ilustrados del lector '+c.example_reader+'.','muted');for(let i=0;i<c.steps.length;i++){const s=c.steps[i];add('p',(i+1)+'. '+s.state+' · '+s.reading,'reading');add('p','IDs: '+s.source_document_ids.join(' · '),'muted')}}else{add('p','No se adjudicó una lectura editorial de valor a esta canción. El atlas conserva sus lecturas categóricas y, cuando existen, pasos anclados.','muted')}if(r.unresolved_fields.length)add('p','Sin resolución: '+r.unresolved_fields.join(', '),'muted');}
async function init(){try{D=await(await fetch('evolution-data.json',{cache:'no-store'})).json();render();$('timeline').on('plotly_click',e=>{if(e.points[0].customdata)showSong(e.points[0].customdata)});for(const id of ['view','language','names'])$(id).onchange=render;}catch(e){$('timeline').textContent='No se pudo cargar el mapa. Sirve esta carpeta con un servidor local junto a Plotly y evolution-data.json.';console.error(e)}}init();
</script></html>'''
    page = page.replace('<option value="fear_paired_joint_votes">Motivo y miedo en los mismos votos</option><option value="fear_consensus_cross">Consenso de motivo y relevancia por separado</option>',
                        '<option value="fear_consensus_cross">Consenso de motivo y relevancia por separado</option><option value="fear_paired_joint_votes">Motivo y miedo en los mismos votos</option>')
    page = page.replace('<label>Etiquetas<select id="names">',
                        '<label>Casos del mapa<select id="scope"><option value="fear">23 referencias al miedo</option><option value="all">40 casos: incluye controles y sin resolución</option></select></label><label>Etiquetas<select id="names">')
    page = page.replace("const songs=selected(),keys=", "const songs=selected().filter(r=>$('scope').value==='all'||['explicit','implied'].includes(r.consensus.fear_relevance)),keys=")
    page = page.replace("(songs.length/5)", "(selected().length/5)")
    page = page.replace("Las categorías se solapan; las tasas no suman uno. 2020–26 es parcial.",
                        "El filtro de casos afecta sólo al mapa. Las curvas conservan toda la selección por idioma. Las categorías se solapan; 2020–26 es parcial.")
    page = page.replace("['view','language','names']", "['view','language','names','scope']")
    page = page.replace('<section class="panel"><div id="timeline"',
                        '<p class="muted">Formas del mapa: ● repetición · ▲ giro · ◆ relación mixta · × arco sin resolver. Color: período de publicación.</p><section class="panel"><div id="timeline"')
    dictionary = "const readingLabels={first_singular:'Primera persona singular',first_collective:'Primera persona colectiva',third_person_story:'Relato en tercera persona',dialogue_or_multiple:'Diálogo o varias voces',impersonal:'Voz impersonal',avoidance:'Evitación',resistance:'Resistencia',care:'Cuidado',ambivalence:'Ambivalencia',resignation:'Resignación',other:'Otra lectura',explicit:'Explícita',implied:'Implícita',absent:'Ausente',unresolved:'Sin resolución',affirmed:'Afirmado',denied:'Negado',hypothetical:'Hipotético',mixed:'Mixto',unclear:'Incierto',reflection:'Reflexión',exposure:'Exposición',fear:'Miedo',threat:'Amenaza',loss:'Pérdida',agency:'Acción',celebration:'Celebración',negation:'Negación',stance:'Postura',fear_polarity:'Polaridad',voice:'Voz',fear_relevance:'Referencia al miedo',subtopics:'Subtemas',arc:'Arco'};const readable=k=>readingLabels[k]||k;"
    page = page.replace('function selected()', dictionary + 'function selected()')
    page = page.replace("r.consensus.voice+' · postura: '+r.consensus.stance+' · miedo: '+r.consensus.fear_relevance+' · polaridad: '+r.consensus.fear_polarity",
                        "readable(r.consensus.voice)+' · postura: '+readable(r.consensus.stance)+' · miedo: '+readable(r.consensus.fear_relevance)+' · polaridad: '+readable(r.consensus.fear_polarity)")
    page = page.replace("(i+1)+'. '+s.state", "(i+1)+'. '+readable(s.state)")
    page = page.replace("r.unresolved_fields.join(', ')", "r.unresolved_fields.map(readable).join(', ')")
    page = page.replace('<label>Etiquetas<select id="names">',
                        '<label>Inspeccionar una canción<select id="case"><option value="">Selecciona un caso</option></select></label><label>Etiquetas<select id="names">')
    # Serialize Plotly updates so rapid filter changes cannot leave stale traces.
    page = page.replace('function render(){', 'async function draw(){')
    page = page.replace("Plotly.react('timeline'", "await Plotly.react('timeline'")
    page = page.replace("Plotly.react('presence'", "await Plotly.react('presence'")
    render_queue = "let drawing=false,pending=false;async function render(){if(drawing){pending=true;return}drawing=true;try{do{pending=false;await draw()}while(pending)}finally{drawing=false}}"
    page = page.replace('function showSong(id)', render_queue + 'function showSong(id)')
    page = page.replace("D=await(await fetch('evolution-data.json',{cache:'no-store'})).json();render();",
                        "D=await(await fetch('evolution-data.json',{cache:'no-store'})).json();for(const r of D.songs.slice().sort((a,b)=>a.year-b.year||a.title.localeCompare(b.title,'es'))){const option=document.createElement('option');option.value=r.song_id;option.textContent=r.year+' · '+r.title+' · '+r.artist;$('case').append(option)}$('case').onchange=()=>{if($('case').value)showSong($('case').value)};await render();")
    (v.SITE / 'evolution-map.html').write_text(page)


if __name__ == '__main__':
    main()
