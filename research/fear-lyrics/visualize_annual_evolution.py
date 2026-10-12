"""Annual, song-weighted views of the new paired readings. No inference/network."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from annual_analysis import TOPICS, LABELS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/annual-120'
SITE = ROOT.parents[1] / 'static/research/fear-lyrics'
FIG = ROOT / 'figures'
COLORS = ['#426a8c','#a84948','#ab7841','#8a5795','#388476','#c2586c','#888335']
ARC = {'repetition':'Repetición','contrast':'Contraste','confrontation':'Confrontación',
       'turn':'Giro','mixed':'Mixto','unresolved':'Sin resolver','no_supported_arc':'Sin arco apoyado'}
PERIODS = [1980,1990,2000,2010,2020]

def load(name): return json.loads((DATA / name).read_text())

def save(fig, name):
    for ext in ['svg','png']:
        path=FIG/f'{name}.{ext}'
        fig.savefig(path,dpi=170,bbox_inches='tight',facecolor=fig.get_facecolor())
        shutil.copy2(path,SITE/path.name)
    plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True);SITE.mkdir(exist_ok=True)
    a=load('annotations.json');votes=load('votes.json');summary=load('summary.json')
    assert len(a)==120 and len(votes)==240
    byid={r['song_id']:r for r in a}
    yearly=[]
    for year in range(1980,2027):
        g=[r for r in a if r['year']==year]
        yearly.append(dict(year=year,n=len(g),fear=sum(r['fear_reference_consensus'] is True for r in g),
                           unresolved=sum(r['fear_reference_consensus'] is None for r in g),
                           en=sum(r['language']=='en' for r in g),es=sum(r['language']=='es' for r in g)))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,(ax,cov)=plt.subplots(2,1,figsize=(16,8.8),gridspec_kw={'height_ratios':[4,1]},sharex=True)
    for year in summary['empty_years']:
        ax.axvspan(year-.43,year+.43,color='#eeeeed',zorder=0)
        cov.axvspan(year-.43,year+.43,color='#eeeeed',zorder=0)
    for i,(key,label,color) in enumerate(zip(TOPICS,LABELS,COLORS)):
        for y in range(1980,2027):
            g=[r for r in a if r['year']==y and key in r['consensus']['subtopics']]
            if g:ax.scatter(y,i,s=130*len(g),c=color,edgecolors='white',linewidth=1,zorder=3)
            if len(g)>1:ax.text(y,i,str(len(g)),ha='center',va='center',color='white',fontsize=9,weight='bold')
    ax.set_yticks(range(7),LABELS);ax.invert_yaxis();ax.set_xlim(1979.4,2026.6)
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.set_title('Siete motivos en las canciones seleccionadas, año por año',loc='left',pad=34,fontsize=17)
    ax.text(0,1.045,'Área = canciones con miedo + motivo apoyados por ambas lecturas. Un caso puede tener dos motivos.',transform=ax.transAxes,color='#555')
    selected=[('Somebody\'s Watching Me','Vigilancia · 1983',-.2,-.38),
              ('Malo','Malo · 2004',-2,.42),('La Puerta Violeta','Rozalén · 2017',-2,.42),
              ('Algo tú','Algo tú · 2026',-2.2,-.38)]
    for title,label,dx,dy in selected:
        r=next(r for r in a if r['title']==title); key=r['consensus']['subtopics'][-1];i=TOPICS.index(key)
        ax.annotate(label,(r['year'],i),(r['year']+dx,i+dy),fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#888','lw':.6},ha='center')
    xs=[r['year'] for r in yearly]
    fear=np.array([r['fear'] for r in yearly]);u=np.array([r['unresolved'] for r in yearly]);n=np.array([r['n'] for r in yearly])
    cov.bar(xs,fear,color='#315b5b',width=.65,label='Referencia apoyada (53)')
    cov.bar(xs,u,bottom=fear,color='#c8a675',width=.65,label='Sin resolver (30)')
    cov.bar(xs,n-fear-u,bottom=fear+u,color='#d8dcdf',width=.65,label='Ausencia apoyada (37)')
    cov.set_ylabel('Canciones\nseleccionadas');cov.set_xticks(range(1980,2027,2));cov.tick_params(axis='x',rotation=45)
    cov.set_xlabel('Año de publicación · zonas grises = sin canciones seleccionadas; corte 11/10/2026')
    cov.legend(loc='upper left',bbox_to_anchor=(0,-.46),ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(hspace=.14,bottom=.16);save(fig,'annual-subtopics-map')
    # Keep unresolved arcs in the denominator; inspect exact categories, not scored emotions.
    fig,axes=plt.subplots(1,2,figsize=(14,5.4))
    arc_order=['repetition','confrontation','contrast','turn','mixed','unresolved']
    arc_colors=['#426a8c','#bd7150','#7b6a93','#388476','#ac8b34','#d5d9dc']
    for ax,field,order,colors,labels in [(axes[0],'arc',arc_order,arc_colors,ARC),
       (axes[1],'stance',['resistance','avoidance','care','ambivalence','resignation','other','unresolved'],
        ['#388476','#bd7150','#426a8c','#7b6a93','#b09a68','#81929b','#d5d9dc'],
        {'resistance':'Resistencia','avoidance':'Evitación','care':'Cuidado','ambivalence':'Ambivalencia','resignation':'Resignación','other':'Otra','unresolved':'Sin resolver'})]:
        bottom=np.zeros(5)
        for key,color in zip(order,colors):
            count=np.array([sum(r['consensus'][field]==key for r in a if r['decade']==d and r['fear_reference_consensus'] is True) for d in PERIODS])
            den=np.array([p['fear_songs'] for p in summary['periods']]);share=count/den*100
            ax.bar(range(5),share,bottom=bottom,color=color,width=.65,label=labels[key])
            for j,(c,s,b) in enumerate(zip(count,share,bottom)):
                if c:ax.text(j,b+s/2,str(c),ha='center',va='center',fontsize=10)
            bottom+=share
        ax.set_xticks(range(5),[f'{d}–{str(d+9)[-2:] if d<2020 else "26*"}\nn={p["fear_songs"]}' for d,p in zip(PERIODS,summary['periods'])])
        ax.set_ylim(0,101);ax.set_ylabel('% de las canciones con referencia apoyada')
        ax.set_title('Relación argumental' if field=='arc' else 'Respuesta de la voz',loc='left')
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.18),ncol=3,fontsize=8,frameon=False)
    fig.suptitle('La forma de responder: recuentos sobre 53 letras, con desacuerdos visibles',x=.08,ha='left',fontsize=15,weight='bold')
    fig.subplots_adjust(wspace=.27,bottom=.3,top=.84);save(fig,'annual-arcs-stances')
    # Two parallel readings demonstrate that arc agreement does not imply identical segmentations.
    titles=['Every Breath You Take','Malo','La Puerta Violeta','Ansiedad']
    fig,axes=plt.subplots(4,1,figsize=(15,11.6))
    for ax,title in zip(axes,titles):
        row=next(r for r in a if r['title']==title);pair=[v for v in votes if v['song_id']==row['song_id']]
        ax.set_xlim(-.1,4.25);ax.set_ylim(-.45,1.9);ax.axis('off')
        ax.text(0,1.58,f'{row["year"]} · {title} · {row["artist"]}   /   {ARC[row["consensus"]["arc"]]}',fontsize=12,weight='bold')
        for j,vote in enumerate(pair):
            ax.text(-.08,1-j*.85,f'Lector {vote["reader"].upper()}',fontsize=8,color='#777',ha='right')
            steps=vote['steps'];positions=np.linspace(.0,2.35,len(steps))
            for k,(x,step) in enumerate(zip(positions,steps)):
                ax.text(x,1-j*.85,textwrap.fill(step['reading'],29),fontsize=9,va='center',ha='left',
                        bbox=dict(boxstyle='round,pad=.35',fc='#f0f4f3',ec='#a2b8b5'))
                if k<len(steps)-1:ax.annotate('',(positions[k+1]-.06,1-j*.85),(x+.77,1-j*.85),arrowprops={'arrowstyle':'->','color':'#819997'})
        ax.text(3.8,.62,'EN JUEGO\n'+textwrap.fill(pair[0]['value_reading'],24),ha='center',va='center',fontsize=9,color='#695542')
    fig.suptitle('Un motivo puede persistir mientras cambia quién actúa frente a él',x=.08,ha='left',fontsize=16,weight='bold')
    fig.text(.08,.015,'Paráfrasis de dos lecturas; secuencias de bloques en orden. El valor es una interpretación con IDs de apoyo, no una escala cuantitativa.',fontsize=9,color='#666')
    fig.subplots_adjust(top=.94,hspace=.2,bottom=.045);save(fig,'annual-argumental-cases')
    payload=dict(songs=a,votes=votes,yearly=yearly,topics=dict(zip(TOPICS,LABELS)),summary=summary,colors=dict(zip(TOPICS,COLORS)))
    (SITE/'annual-evolution-data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
    # Preserve the older 40-song artifact at an explicit historical URL.
    previous=SITE/'evolution-map.html'
    if previous.exists() and not (SITE/'pilot40-evolution-map.html').exists():shutil.copy2(previous,SITE/'pilot40-evolution-map.html')
    (SITE/'evolution-map.html').write_text(HTML)
    receipt=dict(status='executed',songs=120,readings=240,years=[1980,2026],empty_years=summary['empty_years'],
                 inputs={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['annotations.json','votes.json','summary.json']},
                 denominator='Annual selected songs, not all releases. Period styles condition on 53 paired fear references.',
                 project_api_calls=0,figures=['annual-subtopics-map','annual-arcs-stances','annual-argumental-cases'])
    (DATA/'visualization-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))

HTML=r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>El miedo, año por año · 1980–2026</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f6f5f0;color:#202d33;font-family:system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px 30px 60px}a{color:#32766b}h1{font-size:clamp(28px,4vw,50px);letter-spacing:-1.5px;max-width:950px;margin:20px 0 12px}h2{font-size:24px;margin:0 0 12px}p{line-height:1.65;max-width:1000px}.eyebrow{letter-spacing:2px;font-size:12px;text-transform:uppercase;color:#7b6550}.lead{font-size:18px;color:#53616a}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:25px 0 20px;padding:16px;background:#fff;border:1px solid #e0e2db;border-radius:10px}label{font-size:12px;text-transform:uppercase;letter-spacing:.8px;display:flex;flex-direction:column;gap:6px}select{padding:9px;color:#26343a;background:#fafbf8;border:1px solid #c9d1cf;border-radius:4px;max-width:400px;font-size:14px}.stats{display:flex;gap:25px;flex-wrap:wrap;color:#55666b;margin:14px 0}.stats strong{font-size:28px;color:#285f55;display:block}.card{background:white;border:1px solid #e1e3dd;border-radius:10px;padding:20px;margin:20px 0}.plot{height:650px;width:100%}.compact{height:250px}.detail{display:grid;grid-template-columns:1fr 1fr;gap:20px}.reader{background:#f5f7f4;padding:18px;border-radius:6px}.reader p{font-size:14px}.muted{font-size:13px;color:#65737a}.pill{font-size:12px;background:#e8f1ee;padding:5px 8px;border-radius:6px;display:inline-block;margin:2px}.steps{margin:12px 0;list-style:none;padding:0}.steps li{margin:12px 0;border-left:3px solid #8daaa2;padding:0 0 0 12px;font-size:14px}.ids{font-family:monospace;font-size:10px;color:#72817a;overflow-wrap:anywhere}footer{font-size:13px;color:#6d787a;margin-top:40px}#notice{font-size:13px;color:#746953}@media(max-width:750px){main{padding:20px 12px}.detail{grid-template-columns:1fr}.plot{height:570px}.controls{gap:10px}select{max-width:330px}h1{letter-spacing:-.7px}}
</style><script src="plotly.min.js"></script></head><body><main>
<a href="/post/2026-10-11-miedo-embeddings-bertopic-rag/">← Volver al artículo</a>
<div class="eyebrow" style="margin-top:28px">Lecturas, subtemas y relaciones argumentales</div>
<h1>El miedo, año por año.<br>Qué vuelve, quién habla, qué cambia.</h1>
<p class="lead">120 canciones publicadas entre 1982 y 2026, dentro de una ventana de estudio 1980–2026. Cada canción recibió dos lecturas de sus bloques ordenados. El mapa registra motivos apoyados; el inspector conserva ambas interpretaciones y sus alternativas.</p>
<div class="stats"><div><strong>53</strong>referencias apoyadas</div><div><strong>37</strong>ausencias apoyadas</div><div><strong>30</strong>sin resolver</div><div><strong>240</strong>lecturas</div></div>
<div class="controls"><label>Idioma<select id="language"><option value="balanced">Inglés + español</option><option value="en">Inglés</option><option value="es">Español</option></select></label><label>Criterio<select id="criterion"><option value="paired">Ambas lecturas</option><option value="one">Al menos una lectura · sensibilidad</option></select></label><label>Vista<select id="view"><option value="bubbles">Recuento anual por subtema</option><option value="cases">Canciones y arcos por año</option><option value="rates">Proporción anual por subtema</option></select></label><label>Inspeccionar canción<select id="case"><option value="">Seleccionar un caso</option></select></label></div>
<p id="notice"></p><div class="card"><div id="timeline" class="plot"></div><div id="coverage" class="plot compact"></div></div>
<div class="card"><h2 id="caseTitle">Del punto a la lectura</h2><p id="caseMeta">Selecciona una canción o pulsa una marca. Cada subtema reúne motivos; cada arco necesita volver al orden de los bloques.</p><div id="detail" class="detail"></div></div>
<div class="card"><h2>Qué significa una diferencia entre años</h2><p>El área de la burbuja representa el número de canciones, no intensidad emocional. En la vista de proporciones, el denominador es el total de canciones seleccionadas de ese año e idioma; un año con un caso puede alcanzar 100 %. Una canción puede aportar hasta dos motivos. Los años sin canciones quedan grises: carecemos de observaciones.</p><p>La selección tiene 12 canciones por idioma y período, pero no está equilibrada dentro de cada año. La sensibilidad de una lectura amplía el criterio de inclusión; no constituye un intervalo de confianza. La referencia al miedo incluye su negación y amenazas implícitas: describir un peligro no demuestra que el narrador lo experimente.</p></div>
<footer>Ventana 1980–2026; corte 11/10/2026. Tres procesos del mismo asistente configurado; dos lecturas por canción. Categorías y paráfrasis públicas; letras originales privadas. <a href="annual-evolution-data.json">Datos del mapa</a> · <a href="atlas.html">Cuatro encoders del piloto</a> · <a href="expanded/atlas.html">Dos encoders sobre 120 canciones</a>.</footer>
</main><script>
const $=id=>document.getElementById(id);let D,drawing=false,pending=false;
const words={explicit:'Explícita',implied:'Implícita',explicit_or_implied:'Explícita o implícita',absent:'Ausencia',uncertain:'Incierta',unresolved:'Sin resolver',affirmed:'Afirmado',denied:'Negado',hypothetical:'Hipotético',mixed:'Mixto',unclear:'Incierto',first_singular:'Primera persona singular',first_collective:'Voz colectiva',third_person_story:'Relato en tercera persona',dialogue_or_multiple:'Diálogo o voces múltiples',impersonal:'Impersonal',avoidance:'Evitación',resistance:'Resistencia',care:'Cuidado',ambivalence:'Ambivalencia',resignation:'Resignación',other:'Otra',repetition:'Repetición',contrast:'Contraste',confrontation:'Confrontación',turn:'Giro',no_supported_arc:'Sin arco apoyado',not_in_fear_subset:'Fuera del subconjunto de miedo',fear:'Miedo',threat:'Amenaza',agency:'Acción',exposure:'Exposición',loss:'Pérdida',reflection:'Reflexión',celebration:'Celebración',negation:'Negación'};
const word=k=>words[k]||k;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
function selected(){return D.songs.filter(r=>$('language').value==='balanced'||r.language===$('language').value)}
function topics(r){return $('criterion').value==='paired'?r.consensus.subtopics:r.one_reader_subtopics}
function included(r){return $('criterion').value==='paired'?r.fear_reference_consensus===true:r.one_reader_fear}
async function draw(){let g=selected(),positive=g.filter(included),keys=Object.keys(D.topics),traces=[],view=$('view').value;
const shapes=D.summary.empty_years.map(y=>({type:'rect',xref:'x',yref:'paper',x0:y-.43,x1:y+.43,y0:0,y1:1,fillcolor:'#eeeee9',line:{width:0},layer:'below'}));
for(let i=0;i<keys.length;i++){let key=keys[i];if(view==='cases'){let hits=positive.filter(r=>topics(r).includes(key));let byyear={};for(let r of hits)(byyear[r.year]??=[]).push(r);traces.push({type:'scatter',mode:'markers',name:D.topics[key],x:hits.map(r=>r.year),y:hits.map(r=>i+(byyear[r.year].indexOf(r)-(byyear[r.year].length-1)/2)*.13),customdata:hits.map(r=>r.song_id),text:hits.map(r=>`${r.year} · ${esc(r.title)}<br>${esc(r.artist)} · ${r.language.toUpperCase()}<br>${word(r.consensus.arc)}`),hovertemplate:'%{text}<extra></extra>',marker:{color:D.colors[key],size:12,symbol:hits.map(r=>({repetition:'circle',turn:'triangle-up',confrontation:'diamond',contrast:'square',mixed:'star',unresolved:'x'}[r.consensus.arc]||'x')),line:{color:'white',width:.8}}});}
else {let points=[];for(let y=1980;y<=2026;y++){let all=g.filter(r=>r.year===y),hits=positive.filter(r=>r.year===y&&topics(r).includes(key));if(all.length)points.push({year:y,n:all.length,hits})}if(view==='bubbles')points=points.filter(p=>p.hits.length);traces.push({type:'scatter',mode:'markers',name:D.topics[key],x:points.map(p=>p.year),y:points.map(()=>i),customdata:points.map(p=>({year:p.year,key,ids:p.hits.map(r=>r.song_id)})),text:points.map(p=>`${p.year} · ${D.topics[key]}<br>${p.hits.length}/${p.n} canciones · ${(100*p.hits.length/p.n).toFixed(1)} %<br>${p.hits.map(r=>esc(r.title)).join('<br>')}`),hovertemplate:'%{text}<extra></extra>',marker:{color:view==='rates'?points.map(p=>p.hits.length/p.n):D.colors[key],colorscale:[[0,'#f2f5f0'],[1,'#295f55']],cmin:0,cmax:1,size:view==='rates'?20:points.map(p=>Math.sqrt(p.hits.length)*18),line:{color:'#aaa',width:.5}}});}}
const layout={paper_bgcolor:'#fff',plot_bgcolor:'#fff',margin:{l:185,r:30,t:65,b:55},showlegend:false,title:{text:view==='cases'?'Cada punto es una canción · forma = relación argumental':view==='rates'?'Color = proporción de canciones seleccionadas en ese año':'Área = canciones con miedo y motivo apoyados',font:{size:17},x:.02},xaxis:{range:[1979.4,2026.6],tickmode:'linear',tick0:1980,dtick:2,tickangle:-45,title:'Año de publicación',gridcolor:'#f0f0eb'},yaxis:{tickvals:keys.map((_,i)=>i),ticktext:keys.map(k=>D.topics[k]),autorange:'reversed',range:[6.7,-.7],gridcolor:'#e8ece7'},shapes,hovermode:'closest'};
await Plotly.react('timeline',traces,layout,config);
let ys=Array.from({length:47},(_,i)=>1980+i),f=ys.map(y=>g.filter(r=>r.year===y&&r.fear_reference_consensus===true).length),u=ys.map(y=>g.filter(r=>r.year===y&&r.fear_reference_consensus===null).length),a=ys.map(y=>g.filter(r=>r.year===y&&r.fear_reference_consensus===false).length);
await Plotly.react('coverage',[{type:'bar',name:'Referencia apoyada',x:ys,y:f,marker:{color:'#315b5b'}},{type:'bar',name:'Sin resolver',x:ys,y:u,marker:{color:'#c8a675'}},{type:'bar',name:'Ausencia apoyada',x:ys,y:a,marker:{color:'#d8dcdf'}}],{paper_bgcolor:'#fff',plot_bgcolor:'#fff',barmode:'stack',margin:{l:185,r:30,t:20,b:55},xaxis:{range:[1979.4,2026.6],tickmode:'linear',tick0:1980,dtick:2,tickangle:-45},yaxis:{title:'n por año',dtick:2},legend:{orientation:'h',x:0,y:1.15,font:{size:11}},shapes},config);
$('notice').textContent=`${g.length} canciones seleccionadas · ${positive.length} incluidas bajo el criterio actual. Cobertura inferior: clasificación por ambas lecturas. Gris = año sin observaciones en el corpus completo.`;
}
async function render(){if(drawing){pending=true;return}drawing=true;try{do{pending=false;await draw()}while(pending)}finally{drawing=false}}
function showSong(id){let r=D.songs.find(s=>s.song_id===id);if(!r)return;$('case').value=id;$('caseTitle').textContent=`${r.year} · ${r.title}`;$('caseMeta').innerHTML=`${esc(r.artist)} · ${r.language.toUpperCase()}<br><span class="pill">Referencia: ${word(r.consensus.fear_relevance)}</span><span class="pill">Polaridad: ${word(r.consensus.fear_polarity)}</span><span class="pill">Voz: ${word(r.consensus.voice)}</span><span class="pill">Arco: ${word(r.consensus.arc)}</span><br>Motivos compartidos: ${r.consensus.subtopics.map(k=>D.topics[k]).join(' · ')||'Ninguno apoyado por ambas lecturas'}`;
$('detail').innerHTML=D.votes.filter(v=>v.song_id===id).map(v=>`<section class="reader"><strong>Lector ${v.reader.toUpperCase()}</strong><p>Referencia ${word(v.fear_reference).toLowerCase()} · ${word(v.fear_polarity)} · ${word(v.voice)} · ${word(v.stance)}</p><p>Motivos: ${v.subtopics.map(k=>D.topics[k]||word(k)).join(' · ')}</p><strong>${word(v.arc_type)}</strong><ol class="steps">${v.steps.map((s,i)=>`<li>${esc(s.reading)}<div class="ids">${s.source_document_ids.map(esc).join(' · ')}</div>${i<v.links.length?`<span class="muted">↓ ${word(v.links[i].relation)}</span>`:''}</li>`).join('')||'<li>Sin secuencia argumental apoyada.</li>'}</ol><p><strong>En juego:</strong> ${esc(v.value_reading)}</p><div class="ids">${v.value_support_ids.map(esc).join(' · ')}</div><p><strong>Otra lectura:</strong> ${esc(v.counterreading)}</p></section>`).join('');}
async function init(){try{D=await(await fetch('annual-evolution-data.json',{cache:'no-store'})).json();for(let r of D.songs){let o=document.createElement('option');o.value=r.song_id;o.textContent=`${r.year} · ${r.title} · ${r.artist}`;$('case').appendChild(o)}await render();$('timeline').on('plotly_click',e=>{let d=e.points[0].customdata;if(typeof d==='string')showSong(d);else if(d?.ids?.length)showSong(d.ids[0])});for(let id of ['language','criterion','view'])$(id).onchange=render;$('case').onchange=()=>showSong($('case').value);showSong(D.songs.find(r=>r.title==='Malo').song_id)}catch(e){$('notice').textContent='No se pudo cargar el mapa anual';console.error(e)}}init();
</script></body></html>'''

if __name__=='__main__':main()
