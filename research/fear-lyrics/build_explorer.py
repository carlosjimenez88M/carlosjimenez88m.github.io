"""Explorador interactivo de las 56 canciones con miedo (static/research/fear-lyrics/story/explorador.html).

Solo datos públicos: metadatos, categorías, puntuaciones de los ejes, estados por bloque y paráfrasis de los lectores.
Uso: python build_explorer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

pd.set_option('future.infer_string', False)

HERE = Path(__file__).resolve().parent
RES = HERE / 'results' / 'story-v3'
OUT = HERE.parents[1] / 'static' / 'research' / 'fear-lyrics' / 'story' / 'explorador.html'

LOCUS = {'external_collective': 'Sociedad, Estado, calle', 'supernatural_fiction': 'Monstruo o presagio',
         'external_person': 'Otra persona', 'relational_loss': 'Perder un vínculo',
         'existential_time': 'Tiempo, futuro, fracaso', 'internal_self': 'La propia mente', 'none': 'Ninguno',
         'uncertain': 'Incierto'}
WHO = {'speaker': 'quien canta (yo)', 'speaker_collective': 'quien canta (nosotros)', 'addressee': 'un «tú»',
       'third_party': 'un personaje', 'nobody': 'nadie', 'unclear': 'no está claro'}
RESPONSE = {'flight_avoidance': 'huir, evitar', 'paralysis': 'paralizarse', 'plea_prayer': 'suplicar, rezar',
            'resignation': 'resignarse', 'confession_naming': 'confesar, nombrar', 'care_protection': 'cuidar, proteger',
            'defiance_confrontation': 'enfrentar', 'transformation_agency': 'transformar, decidir', 'none': 'ninguna',
            'unclear': 'no está clara'}
ARC = {'loop': 'bucle', 'escalation': 'escalada', 'collapse': 'derrumbe', 'episodic': 'episodio', 'release': 'alivio',
       'turn_to_agency': 'giro a la agencia', 'no_fear_arc': 'sin arco de miedo'}
LOCUS_EN = {'external_collective': 'Society, state, street', 'supernatural_fiction': 'Monster or omen',
            'external_person': 'Another person', 'relational_loss': 'Losing a bond', 'existential_time': 'Time, future, failure',
            'internal_self': "One's own mind", 'none': 'None', 'uncertain': 'Uncertain'}
WHO_EN = {'speaker': 'the singer (I)', 'speaker_collective': 'the singer (we)', 'addressee': 'a "you"',
          'third_party': 'a character', 'nobody': 'nobody', 'unclear': 'unclear'}
RESPONSE_EN = {'flight_avoidance': 'flee, avoid', 'paralysis': 'freeze', 'plea_prayer': 'plead, pray', 'resignation': 'resign',
               'confession_naming': 'confess, name it', 'care_protection': 'care, protect', 'defiance_confrontation': 'confront',
               'transformation_agency': 'transform, decide', 'none': 'none', 'unclear': 'unclear'}
ARC_EN = {'loop': 'loop', 'escalation': 'escalation', 'collapse': 'collapse', 'episodic': 'episode', 'release': 'relief',
          'turn_to_agency': 'turn to agency', 'no_fear_arc': 'no fear arc'}
ZONE = {'external_collective': 'afuera', 'supernatural_fiction': 'afuera', 'external_person': 'afuera',
        'relational_loss': 'entre', 'existential_time': 'entre', 'internal_self': 'adentro'}


def zone(row) -> str:
    votes = {'afuera': 0.0, 'entre': 0.0, 'adentro': 0.0}
    for k, z in ZONE.items():
        votes[z] += row.get(f'threat_locus:{k}', 0)
    return max(votes, key=votes.get) if max(votes.values()) > 0 else 'entre'


def build(lang: str = 'es') -> dict:
    en = lang == 'en'
    para = json.loads((RES / 'paraphrases_en.json').read_text()) if en else {}
    t = (lambda x: para.get(x, x)) if en else (lambda x: x)
    locus, who, response, arc = (LOCUS_EN, WHO_EN, RESPONSE_EN, ARC_EN) if en else (LOCUS, WHO, RESPONSE, ARC)
    table = pd.read_csv(RES / 'song_table.csv', index_col=0)
    readings = pd.read_csv(RES / 'readings_public.csv')
    states = pd.read_csv(RES / 'block_states.csv')
    fear = table[table.fear_weight > 0]
    songs = []
    for sid, r in fear.sort_values('year').iterrows():
        rd = readings[readings.song_id == sid].sort_values('reader')
        reads = []
        for _, x in rd.iterrows():
            st = states[(states.song_id == sid) & (states.reader == x.reader)].sort_values('position')
            reads.append(dict(
                fear=x.fear_reference, locus=locus.get(x.threat_locus, x.threat_locus), who=who.get(x.who_fears, x.who_fears),
                response=response.get(x.response, x.response), arc=arc.get(x.arc_shape, x.arc_shape),
                object=t(x.threat_object), value=t(x.value_at_stake), counter=t(x.counterreading),
                steps=[t(s['reading']) for s in json.loads(x.arc_steps)], states=st.state.tolist()))
        songs.append(dict(id=sid, title=r.title, artist=r.artist, year=int(r.year), lang=r.language,
                          period=int(r.period), weight=float(r.fear_weight), zone=zone(r),
                          x=round(float(r.interiorizacion_fear), 3), y=round(float(r.agencia_all), 3), reads=reads))
    return dict(songs=songs)


PAGE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explorador del miedo</title>
<meta name="description" content="Las 56 canciones con miedo de 1980 a 2026: dónde está la amenaza, quién teme, cómo responde la voz y cómo se mueve la letra.">
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--afuera:#eb6834;--entre:#1baf7a;--adentro:#2a78d6;--chip:#f0efec;--focus:#2a78d6;
--s-threat:#b3261e;--s-avoid:#e66767;--s-plea:#f2a7a2;--s-neutral:#f0efec;--s-care:#b7d3f6;--s-defy:#5598e7;--s-agency:#184f95;--s-filler:transparent}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--afuera:#d95926;--entre:#199e70;--adentro:#3987e5;--chip:#2c2c2a;--focus:#3987e5;--s-neutral:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--afuera:#d95926;--entre:#199e70;--adentro:#3987e5;--chip:#2c2c2a;--focus:#3987e5;--s-neutral:#383835}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:clamp(24px,3.4vw,34px);line-height:1.15;margin:0 0 6px}
.lede{color:var(--ink2);max-width:860px;margin:0 0 18px}
a{color:var(--adentro)}
.controls{display:flex;flex-wrap:wrap;gap:10px 18px;margin:8px 0 14px;align-items:center}
.group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.group span{font-size:13px;color:var(--muted);margin-right:2px}
button.chip{font:inherit;font-size:13px;border:1px solid var(--border);background:var(--surface);color:var(--ink);padding:5px 11px;border-radius:999px;cursor:pointer}
button.chip[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
button.chip:focus-visible,svg circle:focus-visible,tr:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:16px}
@media (max-width:860px){.layout{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px}
.panel h2{font-size:15px;margin:0 0 4px}
.panel .sub{font-size:13px;color:var(--ink2);margin:0 0 8px}
svg{display:block;max-width:100%}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink2);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:7px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;max-width:260px;z-index:5}
.card h3{margin:0;font-size:20px;line-height:1.2}
.card .meta{color:var(--ink2);font-size:13px;margin:2px 0 10px}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.tag{background:var(--chip);border-radius:6px;padding:2px 8px;font-size:12.5px;color:var(--ink)}
.reading{border-top:1px solid var(--grid);padding-top:10px;margin-top:10px}
.reading h4{margin:0 0 4px;font-size:13px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.reading dl{display:grid;grid-template-columns:110px 1fr;gap:3px 10px;margin:0;font-size:14px}
.reading dt{color:var(--muted)}
.reading dd{margin:0}
.strip{display:flex;gap:2px;margin:6px 0 2px;height:16px}
.strip i{flex:1;border-radius:3px}
.strip i.filler{border:1px dashed var(--axis)}
.keys{display:flex;flex-wrap:wrap;gap:4px 10px;font-size:12px;color:var(--ink2);margin-top:8px}
.keys i{display:inline-block;width:12px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
ol.steps{margin:4px 0 0 18px;padding:0;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12.5px;position:sticky;top:0;background:var(--surface)}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.sel td{background:var(--chip)}
tbody tr{cursor:pointer}
.tablewrap{max-height:420px;overflow:auto}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
footer{color:var(--muted);font-size:12.5px;margin-top:18px;max-width:900px}
</style>
</head>
<body>
<main>
<h1>Explorador: ¿dónde vive el miedo?</h1>
<p class="lede">Las 56 canciones con miedo explícito o implícito entre 120 éxitos en inglés y español, de 1980 a 2026. Cada punto es una canción: la posición la decide la geometría de los embeddings; el color, dos lecturas ciegas. Elige una canción para ver sus lecturas y cómo se mueve la voz bloque a bloque. Acompaña al artículo <a href="/post/2026-10-11-miedo-embeddings-bertopic-rag/">«La evolución de la canción que trata sobre el miedo»</a>.</p>
<div class="controls" role="toolbar" aria-label="Filtros">
  <div class="group" data-key="period"><span>Período</span></div>
  <div class="group" data-key="lang"><span>Idioma</span></div>
  <div class="group" data-key="zone"><span>Amenaza</span></div>
</div>
<div class="layout">
  <section class="panel" aria-labelledby="t-compass">
    <h2 id="t-compass">La brújula del miedo</h2>
    <p class="sub">Horizontal: ¿la amenaza está afuera o adentro? Vertical: ¿la voz huye o enfrenta? Desviaciones estándar sobre el corpus.</p>
    <svg id="compass" role="img" aria-label="Dispersión de canciones en dos ejes semánticos"></svg>
    <div class="legend" id="legend"></div>
  </section>
  <section class="panel card" id="card" aria-live="polite"></section>
</div>
<section class="panel" style="margin-top:16px" aria-labelledby="t-table">
  <h2 id="t-table">Todas las canciones con miedo</h2>
  <p class="sub">Vista de tabla de los mismos datos. Peso 1: ambas lecturas ven miedo; 0,5: una.</p>
  <div class="tablewrap"><table id="table"><thead><tr><th>Año</th><th>Canción</th><th>Idioma</th><th>Dónde (lecturas)</th><th>Arco</th><th class="num">Afuera→adentro</th><th class="num">Huir→enfrentar</th></tr></thead><tbody></tbody></table></div>
</section>
<footer>Datos públicos del estudio: categorías y paráfrasis breves de los lectores, sin citas de las letras. Modelo de embeddings: multilingual-e5-large-instruct con instrucción de miedo y dirección de idioma retirada. Muestra intencional; 2020–26 parcial. Código y resultados en <a href="/examples/fear-lyrics-study.zip">el paquete del estudio</a>.</footer>
</main>
<div class="tip" id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const ZONES = {afuera:'Afuera: sociedad, monstruo, otra persona', entre:'Entre: un vínculo, el tiempo', adentro:'Adentro: la propia mente'};
const PERIODS = {1980:'80s',1990:'90s',2000:'00s',2010:'10s',2020:'20s'};
const LANGS = {en:'Inglés', es:'Español'};
const STATE = {threat:['--s-threat','amenaza'],fear:['--s-threat','miedo'],avoidance:['--s-avoid','evitación'],plea:['--s-plea','súplica'],
  loss:['--s-plea','pérdida'],confession:['--s-plea','confesión'],reflection:['--s-neutral','reflexión'],desire:['--s-care','deseo'],
  care:['--s-care','cuidado'],celebration:['--s-care','celebración'],defiance:['--s-defy','desafío'],agency:['--s-agency','agencia'],filler:['--s-filler','relleno']};
const filters = {period:'all', lang:'all', zone:'all'};
let selected = DATA.songs.find(s => s.title === 'Ansiedad') || DATA.songs[0];
const esc = t => String(t).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2).replace('.', ',');

function chips(){
  const opts = {period:[['all','Todos'],...Object.entries(PERIODS)], lang:[['all','Ambos'],...Object.entries(LANGS)],
                zone:[['all','Todas'],['afuera','Afuera'],['entre','Entre'],['adentro','Adentro']]};
  document.querySelectorAll('.group').forEach(g => {
    const key = g.dataset.key;
    opts[key].forEach(([v,l]) => {
      const b = document.createElement('button'); b.className = 'chip'; b.textContent = l; b.dataset.v = v;
      b.setAttribute('aria-pressed', String(filters[key] === v));
      b.onclick = () => { filters[key] = v; g.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.v === v))); render(); };
      g.appendChild(b);
    });
  });
}
const visible = s => (filters.period === 'all' || String(s.period) === filters.period) && (filters.lang === 'all' || s.lang === filters.lang) && (filters.zone === 'all' || s.zone === filters.zone);
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function compass(){
  const svg = document.getElementById('compass');
  const W = Math.max(300, Math.round(svg.parentElement.clientWidth - 28)), H = Math.round(Math.max(300, Math.min(520, W * .8))), m = {l:40, r:14, t:14, b:44};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  const xs = DATA.songs.map(s => s.x), ys = DATA.songs.map(s => s.y);
  const x0 = Math.min(...xs) - .3, x1 = Math.max(...xs) + .3, y0 = Math.min(...ys) - .3, y1 = Math.max(...ys) + .3;
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  let g = `<line x1="${X(0)}" x2="${X(0)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--grid)"/><line x1="${m.l}" x2="${W-m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--grid)"/>`;
  for (let t = Math.ceil(x0); t <= x1; t++) g += `<text x="${X(t)}" y="${H-m.b+16}" font-size="11" fill="var(--muted)" text-anchor="middle">${t}</text>`;
  for (let t = Math.ceil(y0); t <= y1; t++) g += `<text x="${m.l-8}" y="${Y(t)+4}" font-size="11" fill="var(--muted)" text-anchor="end">${t}</text>`;
  g += `<text x="${m.l}" y="${H-6}" font-size="12" fill="var(--ink2)">← afuera</text><text x="${W-m.r}" y="${H-6}" font-size="12" fill="var(--ink2)" text-anchor="end">adentro →</text>`;
  g += `<text transform="translate(12 ${H-m.b}) rotate(-90)" font-size="12" fill="var(--ink2)">← huir, suplicar</text><text transform="translate(12 ${m.t+70}) rotate(-90)" font-size="12" fill="var(--ink2)" text-anchor="end">enfrentar →</text>`;
  const pts = DATA.songs.map(s => {
    const on = visible(s), sel = s === selected, r = s.weight === 1 ? 6.5 : 5;
    return `<circle cx="${X(s.x)}" cy="${Y(s.y)}" r="${sel ? r + 3 : r}" fill="${s.weight === 1 ? `var(--${s.zone})` : 'var(--surface)'}" stroke="${s.weight === 1 ? 'var(--surface)' : `var(--${s.zone})`}" stroke-width="${s.weight === 1 ? 2 : 2.2}" opacity="${on ? 1 : .12}" tabindex="${on ? 0 : -1}" data-id="${s.id}" style="cursor:pointer"><title>${esc(s.title)}</title></circle>` +
      (sel ? `<circle cx="${X(s.x)}" cy="${Y(s.y)}" r="${r + 7}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>` : '');
  }).join('');
  svg.innerHTML = g + pts;
  const tip = document.getElementById('tip');
  svg.querySelectorAll('circle[data-id]').forEach(c => {
    const s = DATA.songs.find(z => z.id === c.dataset.id);
    c.onmouseenter = e => { tip.style.display = 'block'; tip.innerHTML = `<strong>${esc(s.title)}</strong> · ${s.year}<br>${esc(s.artist)}<br><span style="color:var(--muted)">${ZONES[s.zone]}</span><br>Afuera→adentro ${fmt(s.x)} · Huir→enfrentar ${fmt(s.y)}`; };
    c.onmousemove = e => { tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; };
    c.onmouseleave = () => tip.style.display = 'none';
    c.onclick = () => { selected = s; render(); };
    c.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selected = s; render(); } };
  });
  document.getElementById('legend').innerHTML = Object.entries(ZONES).map(([k,l]) => `<span><i style="background:var(--${k})"></i>${l}</span>`).join('') +
    `<span><i style="background:var(--surface);border:2px solid var(--ink2)"></i>Hueco: una sola lectura ve miedo</span>`;
}

function card(){
  const s = selected;
  const reads = s.reads.map((r, i) => `
    <div class="reading"><h4>Lectura ${i + 1}${r.fear === 'explicit' || r.fear === 'implied' ? '' : ' · no ve miedo'}</h4>
      <div class="strip" aria-label="Estados por bloque">${r.states.map(st => `<i class="${st === 'filler' ? 'filler' : ''}" style="background:var(${STATE[st][0]})" title="${STATE[st][1]}"></i>`).join('')}</div>
      <dl><dt>A qué teme</dt><dd>${esc(r.object)}</dd><dt>Dónde</dt><dd>${esc(r.locus)}</dd><dt>Quién teme</dt><dd>${esc(r.who)}</dd>
      <dt>Responde</dt><dd>${esc(r.response)}</dd><dt>Arco</dt><dd>${esc(r.arc)}</dd><dt>Qué protege</dt><dd>${esc(r.value)}</dd>
      <dt>Contralectura</dt><dd>${esc(r.counter)}</dd></dl>
      ${r.steps.length ? `<ol class="steps">${r.steps.map(t => `<li>${esc(t)}</li>`).join('')}</ol>` : ''}
    </div>`).join('');
  document.getElementById('card').innerHTML = `
    <h3>${esc(s.title)}</h3><p class="meta">${esc(s.artist)} · ${s.year} · ${LANGS[s.lang]}</p>
    <div class="tags"><span class="tag">${ZONES[s.zone]}</span><span class="tag">Afuera→adentro ${fmt(s.x)}</span><span class="tag">Huir→enfrentar ${fmt(s.y)}</span></div>
    <p class="sub" style="margin:0">Tiras: estado de la voz en cada bloque de la letra, en orden; estribillos repetidos incluidos.</p>
    ${reads}
    <div class="keys">${[['--s-threat','amenaza, miedo'],['--s-avoid','evitación'],['--s-plea','súplica, pérdida, confesión'],['--s-neutral','reflexión'],['--s-care','deseo, cuidado'],['--s-defy','desafío'],['--s-agency','agencia']].map(([v,l]) => `<span><i style="background:var(${v})"></i>${l}</span>`).join('')}</div>`;
}

function table(){
  const tb = document.querySelector('#table tbody');
  tb.innerHTML = DATA.songs.filter(visible).map(s => `<tr tabindex="0" data-id="${s.id}" class="${s === selected ? 'sel' : ''}">
    <td class="num">${s.year}</td><td><span class="dot" style="background:var(--${s.zone})"></span><strong>${esc(s.title)}</strong><br><span style="color:var(--muted)">${esc(s.artist)}</span></td>
    <td>${LANGS[s.lang]}</td><td>${[...new Set(s.reads.map(r => r.locus))].map(esc).join(' / ')}</td><td>${[...new Set(s.reads.map(r => r.arc))].map(esc).join(' / ')}</td>
    <td class="num">${fmt(s.x)}</td><td class="num">${fmt(s.y)}</td></tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr => {
    const pick = () => { selected = DATA.songs.find(z => z.id === tr.dataset.id); render(); document.getElementById('card').scrollIntoView({block:'nearest'}); };
    tr.onclick = pick; tr.onkeydown = e => { if (e.key === 'Enter') pick(); };
  });
}
function render(){ compass(); card(); table(); }
chips(); render();
let resizeTimer; addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(compass, 120); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
</script>
</body>
</html>
"""


UI_EN = [
    ('<html lang="es">', '<html lang="en">'),
    ('<title>Explorador del miedo</title>', '<title>Fear Explorer</title>'),
    ('content="Las 56 canciones con miedo de 1980 a 2026: dónde está la amenaza, quién teme, cómo responde la voz y cómo se mueve la letra."',
     'content="The 56 fear songs from 1980 to 2026: where the threat is, who is afraid, how the voice responds, and how the lyric moves."'),
    ('<h1>Explorador: ¿dónde vive el miedo?</h1>', '<h1>Explorer: where does fear live?</h1>'),
    ('Las 56 canciones con miedo explícito o implícito entre 120 éxitos en inglés y español, de 1980 a 2026. Cada punto es una canción: la posición la decide la geometría de los embeddings; el color, dos lecturas ciegas. Elige una canción para ver sus lecturas y cómo se mueve la voz bloque a bloque. Acompaña al artículo <a href="/post/2026-10-11-miedo-embeddings-bertopic-rag/">«La evolución de la canción que trata sobre el miedo»</a>.',
     'The 56 songs with explicit or implied fear among 120 hits in English and Spanish, from 1980 to 2026. Each dot is a song: its position comes from the embedding geometry; its color, from two blind readings. Pick a song to see its readings and how the voice moves block by block. Companion to the essay <a href="/post/2026-10-11-fear-songs-embeddings/">"The Evolution of Songs About Fear"</a>. Readers\' notes are translated from Spanish.'),
    ('aria-label="Filtros"', 'aria-label="Filters"'),
    ('<span>Período</span>', '<span>Period</span>'), ('<span>Idioma</span>', '<span>Language</span>'), ('<span>Amenaza</span>', '<span>Threat</span>'),
    ('La brújula del miedo</h2>', 'The compass of fear</h2>'),
    ('Horizontal: ¿la amenaza está afuera o adentro? Vertical: ¿la voz huye o enfrenta? Desviaciones estándar sobre el corpus.',
     'Horizontal: is the threat outside or inside? Vertical: does the voice flee or confront? Standard deviations over the corpus.'),
    ('aria-label="Dispersión de canciones en dos ejes semánticos"', 'aria-label="Scatter of songs on two semantic axes"'),
    ('Todas las canciones con miedo</h2>', 'All fear songs</h2>'),
    ('Vista de tabla de los mismos datos. Peso 1: ambas lecturas ven miedo; 0,5: una.', 'Table view of the same data. Weight 1: both readings see fear; 0.5: one.'),
    ('<th>Año</th><th>Canción</th><th>Idioma</th><th>Dónde (lecturas)</th><th>Arco</th><th class="num">Afuera→adentro</th><th class="num">Huir→enfrentar</th>',
     '<th>Year</th><th>Song</th><th>Language</th><th>Where (readings)</th><th>Arc</th><th class="num">Outside→inside</th><th class="num">Flee→confront</th>'),
    ('Datos públicos del estudio: categorías y paráfrasis breves de los lectores, sin citas de las letras. Modelo de embeddings: multilingual-e5-large-instruct con instrucción de miedo y dirección de idioma retirada. Muestra intencional; 2020–26 parcial. Código y resultados en <a href="/examples/fear-lyrics-study.zip">el paquete del estudio</a>.',
     'Public study data: readers\' categories and short paraphrases, with no lyric quotations. Embedding model: multilingual-e5-large-instruct with a fear instruction and the language direction removed. Purposive sample; 2020–26 partial. Code and results in <a href="/examples/fear-lyrics-study.zip">the study package</a>.'),
    ("const ZONES = {afuera:'Afuera: sociedad, monstruo, otra persona', entre:'Entre: un vínculo, el tiempo', adentro:'Adentro: la propia mente'};",
     "const ZONES = {afuera:'Outside: society, monster, another person', entre:'In between: a bond, time', adentro:\"Inside: one's own mind\"};"),
    ("const LANGS = {en:'Inglés', es:'Español'};", "const LANGS = {en:'English', es:'Spanish'};"),
    ("const STATE = {threat:['--s-threat','amenaza'],fear:['--s-threat','miedo'],avoidance:['--s-avoid','evitación'],plea:['--s-plea','súplica'],\n  loss:['--s-plea','pérdida'],confession:['--s-plea','confesión'],reflection:['--s-neutral','reflexión'],desire:['--s-care','deseo'],\n  care:['--s-care','cuidado'],celebration:['--s-care','celebración'],defiance:['--s-defy','desafío'],agency:['--s-agency','agencia'],filler:['--s-filler','relleno']};",
     "const STATE = {threat:['--s-threat','threat'],fear:['--s-threat','fear'],avoidance:['--s-avoid','avoidance'],plea:['--s-plea','plea'],\n  loss:['--s-plea','loss'],confession:['--s-plea','confession'],reflection:['--s-neutral','reflection'],desire:['--s-care','desire'],\n  care:['--s-care','care'],celebration:['--s-care','celebration'],defiance:['--s-defy','defiance'],agency:['--s-agency','agency'],filler:['--s-filler','filler']};"),
    (".toFixed(2).replace('.', ',')", ".toFixed(2)"),
    ("[['all','Todos'],...Object.entries(PERIODS)], lang:[['all','Ambos'],...Object.entries(LANGS)],",
     "[['all','All'],...Object.entries(PERIODS)], lang:[['all','Both'],...Object.entries(LANGS)],"),
    ("zone:[['all','Todas'],['afuera','Afuera'],['entre','Entre'],['adentro','Adentro']]};",
     "zone:[['all','All'],['afuera','Outside'],['entre','In between'],['adentro','Inside']]};"),
    ('>← afuera</text>', '>← outside</text>'), ('>adentro →</text>', '>inside →</text>'),
    ('>← huir, suplicar</text>', '>← flee, plead</text>'), ('>enfrentar →</text>', '>confront →</text>'),
    ('<br>Afuera→adentro ${fmt(s.x)} · Huir→enfrentar ${fmt(s.y)}', '<br>Outside→inside ${fmt(s.x)} · Flee→confront ${fmt(s.y)}'),
    ('Hueco: una sola lectura ve miedo', 'Hollow: only one reading sees fear'),
    ("<h4>Lectura ${i + 1}${r.fear === 'explicit' || r.fear === 'implied' ? '' : ' · no ve miedo'}</h4>",
     "<h4>Reading ${i + 1}${r.fear === 'explicit' || r.fear === 'implied' ? '' : ' · sees no fear'}</h4>"),
    ('aria-label="Estados por bloque"', 'aria-label="States by block"'),
    ('<dl><dt>A qué teme</dt>', '<dl><dt>Fears</dt>'), ('<dt>Dónde</dt>', '<dt>Where</dt>'), ('<dt>Quién teme</dt>', '<dt>Who fears</dt>'),
    ('<dt>Responde</dt>', '<dt>Response</dt>'), ('<dt>Arco</dt>', '<dt>Arc</dt>'), ('<dt>Qué protege</dt>', '<dt>Protects</dt>'),
    ('<dt>Contralectura</dt>', '<dt>Counter-reading</dt>'),
    ('<span class="tag">Afuera→adentro ${fmt(s.x)}</span><span class="tag">Huir→enfrentar ${fmt(s.y)}</span>',
     '<span class="tag">Outside→inside ${fmt(s.x)}</span><span class="tag">Flee→confront ${fmt(s.y)}</span>'),
    ('Tiras: estado de la voz en cada bloque de la letra, en orden; estribillos repetidos incluidos.',
     "Strips: the voice's state in each block of the lyric, in order; repeated choruses included."),
    ("[['--s-threat','amenaza, miedo'],['--s-avoid','evitación'],['--s-plea','súplica, pérdida, confesión'],['--s-neutral','reflexión'],['--s-care','deseo, cuidado'],['--s-defy','desafío'],['--s-agency','agencia']]",
     "[['--s-threat','threat, fear'],['--s-avoid','avoidance'],['--s-plea','plea, loss, confession'],['--s-neutral','reflection'],['--s-care','desire, care'],['--s-defy','defiance'],['--s-agency','agency']]"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['es', 'en'], default='es')
    lang = parser.parse_args().lang
    data = build(lang)
    page = PAGE
    out = OUT
    if lang == 'en':
        for es, en in UI_EN:
            assert es in page, f'texto de interfaz no encontrado: {es[:60]}'
            page = page.replace(es, en)
        out = OUT.parent / 'en' / 'explorer.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace('__DATA__', json.dumps(data, ensure_ascii=False)))
    print(out, len(data['songs']), 'canciones', f'{out.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
