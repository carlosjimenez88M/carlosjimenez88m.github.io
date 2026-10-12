import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import visualize as v
p=v.HERE/'pilot-comparison/results/arc-panel/fieldwise-posthoc'
annotations=json.loads((p/'song_annotations.json').read_text())
ontology=json.loads((p.parent/'ontology.json').read_text())['subtopics']
keys=[k for k in ontology if k not in ['none','uncertain','other']]
decades=sorted({r['decade'] for r in annotations})
fig,axes=plt.subplots(1,2,figsize=(13.2,6.3),sharey=True)
for ax,conditioned in zip(axes,[False,True]):
 counts=np.zeros((len(keys),len(decades)))
 for i,k in enumerate(keys):
  for j,d in enumerate(decades):
   group=[r for r in annotations if r['decade']==d]
   counts[i,j]=sum(k in r['consensus']['subtopics'] and (not conditioned or r['consensus']['fear_relevance'] in ['explicit','implied']) for r in group)
 im=ax.imshow(counts,cmap='YlGnBu',vmin=0,vmax=8,aspect='auto')
 ax.set(xticks=np.arange(len(decades)),xticklabels=['80s','90s','00s','10s','20–26*'],yticks=np.arange(len(keys)),yticklabels=[ontology[k] for k in keys],xlabel='Período de publicación')
 ax.set_title('Motivos del conjunto' if not conditioned else 'Motivos con miedo explícito o implícito',loc='left',weight='bold',fontsize=11)
 for i in range(len(keys)):
  for j in range(len(decades)):
   ax.text(j,i,f'{int(counts[i,j])}/8',ha='center',va='center',fontsize=10,color='white' if counts[i,j]>=5 else '#263341')
fig.subplots_adjust(left=.19,right=.91,bottom=.16,top=.86,wspace=.16)
cax=fig.add_axes([.935,.26,.018,.45]);fig.colorbar(im,cax=cax,label='Canciones con consenso por campo')
v.save(fig,'voted-subtopics-time','Piloto40: 8 canciones por período; 4 EN y 4 ES. Denominador íntegro; categorías solapadas. 7 casos sin acuerdo sobre relevancia del miedo.')
fig,ax=plt.subplots(figsize=(10.5,4.6))
fields=json.loads((p/'summary.json').read_text())['fields']
labels={'voice':'Voz','stance':'Postura','fear_relevance':'Relevancia del miedo','fear_polarity':'Polaridad','subtopics':'Subtemas','arc':'Arco con anclajes válidos'}
y=np.arange(len(fields));initial=[r['initial_agree_songs'] for r in fields];majority=[r['majority_resolved_songs'] for r in fields]
ax.barh(y-.15,initial,height=.28,color=v.COLORS[0],label='Acuerdo entre dos lectores iniciales')
ax.barh(y+.15,majority,height=.28,color=v.COLORS[1],label='Consenso tras hasta20 críticas ciegas')
ax.set(yticks=y,yticklabels=[labels[r['field']] for r in fields],xlim=(0,43),xlabel='Canciones de40 con este resultado',title='Reconocer una voz y reconstruir un arco son tareas distintas')
for i,n in enumerate(majority):ax.text(n+.4,i+.15,str(n),va='center')
ax.invert_yaxis();ax.legend(frameon=False,fontsize=9,loc='lower right')
fig.tight_layout(rect=(0,.13,1,.98))
v.save(fig,'agent-vote-agreement','Diagnóstico posterior por campo con los votos existentes. Arcos conservan validación estricta de orden y anclajes; consenso no mide exactitud.')
for name in ['voted-subtopics-time','agent-vote-agreement']:
 for ext in ['svg','png']:
  import shutil
  shutil.copy2(v.FIG/f'{name}.{ext}',v.SITE/f'{name}.{ext}')
print('Rendered: two voted-reading figures; no API calls')
