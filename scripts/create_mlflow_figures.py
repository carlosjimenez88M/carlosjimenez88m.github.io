"""Generate original diagrams and an explicitly synthetic comparison for the MLflow essay.
Run: MPLCONFIGDIR=/tmp/blog-matplotlib python3 scripts/create_mlflow_figures.py
No benchmark data or production measurements are used.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'static/img/mlflow-2026'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, INK, MUTED, BORDER = '#2465a4', '#242424', '#646464', '#d3dce6'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':14, 'text.color':INK,
    'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False})

def canvas(title, subtitle, height=7):
    fig, ax=plt.subplots(figsize=(10,height))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.03,right=.97,bottom=.04,top=.83)
    ax.set(xlim=(0,10),ylim=(0,7)); ax.axis('off')
    fig.text(.045,.95,title,fontsize=21,weight='bold',va='top')
    fig.text(.045,.89,subtitle,fontsize=12,color=MUTED,va='top')
    return fig,ax

def box(ax,x,y,w,h,title,detail):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.06',
        facecolor='#f4f8fc',edgecolor=BORDER,linewidth=1.1))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontsize=15,weight='bold',color=BLUE)
    ax.text(x+w/2,y+h*.30,detail,ha='center',va='center',fontsize=12.5,linespacing=1.5)

def arrow(ax,start,end,rad=0,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={rad}',
        arrowstyle='-|>',mutation_scale=17,linewidth=1.5,color=BLUE,
        linestyle='--' if dashed else '-'))

def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white')
    plt.close(fig)

fig,ax=canvas('A prompt change needs a release loop', 'Conceptual workflow · evaluation is the gate between a candidate and a release')
for x,y,title,detail in [(.5,5.2,'1. Observe a failure','Trace inputs, context, and actions'),(5.5,5.2,'2. Review the case','Label expectations; preserve evidence'),
    (5.5,2.9,'3. Register a candidate','Record the prompt diff and version'),(.5,2.9,'4. Evaluate the change','Inspect regressions and constraints'),
    (.5,.6,'5. Release deliberately','Pin configuration; retain rollback'),(5.5,.6,'6. Monitor behavior','Sample traces and review new failures')]:
    box(ax,x,y,4,1.3,title,detail)
arrow(ax,(4.6,5.85),(5.4,5.85)); arrow(ax,(7.5,5.1),(7.5,4.3))
arrow(ax,(5.4,3.55),(4.6,3.55)); arrow(ax,(2.5,2.8),(2.5,2.0))
arrow(ax,(4.6,1.25),(5.4,1.25))
# Dashed return path outside the process boxes.
ax.plot([9.6,9.85,9.85,.2,.2],[1.25,1.25,6.8,6.8,5.85],color=BLUE,lw=1.2,ls='--')
arrow(ax,(.2,5.85),(.4,5.85),dashed=True)
save(fig,'prompt-release-loop')

fig,ax=plt.subplots(figsize=(10,5.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(left=.24,right=.93,top=.73,bottom=.19)
fig.text(.045,.95,'An average can hide meaningful regressions',fontsize=21,weight='bold',va='top')
fig.text(.045,.87,'SYNTHETIC EXAMPLE · 100 paired cases · not an MLflow benchmark',fontsize=12,color=MUTED)
labels=['Correct in both','Fixed by candidate','Broken by candidate','Wrong in both']; vals=[72,10,8,10]
colors=['#aab8c7',BLUE,'#b45043','#d8dfe6']
ax.barh(labels,vals,color=colors,height=.6); ax.invert_yaxis()
ax.set_xlim(0,80); ax.set_xlabel('Number of evaluation cases',fontsize=13)
ax.spines[['left','bottom']].set_visible(False); ax.tick_params(axis='both',length=0,labelsize=13)
ax.xaxis.grid(True,color='#e6e9ed'); ax.set_axisbelow(True)
for i,v in enumerate(vals): ax.text(v+1.3,i,str(v),va='center',fontsize=14,weight='bold')
fig.text(.045,.06,'Baseline: 80/100 correct → Candidate: 82/100 correct. The +2-point gain hides 8 regressions.',fontsize=12,color=INK)
assert sum(vals)==100 and vals[0]+vals[2]==80 and vals[0]+vals[1]==82
save(fig,'paired-regressions')

fig,ax=canvas('Operating the workflow on Google Cloud', 'Reference design · direct model calls shown; AI Gateway routing is an optional alternative',height=7.5)
box(ax,.3,5.3,4,1.3,'Agent application','Cloud Run / GKE / managed runtime')
box(ax,5.7,5.3,4,1.3,'CI and evaluation','Reviewed cases + release checks')
box(ax,.3,2.9,3,1.3,'Gemini on Vertex AI','Generation and model calls')
box(ax,4.3,2.9,5.4,1.3,'MLflow server · Cloud Run or GKE','Registry, evaluation records, and traces')
box(ax,4.3,.4,2.5,1.3,'Cloud SQL','PostgreSQL\nMetadata')
box(ax,7.2,.4,2.5,1.3,'Cloud Storage','Private bucket\nArtifacts / span data')
arrow(ax,(1.8,5.2),(1.8,4.3))
ax.text(.3,4.65,'Inference',fontsize=11,color=MUTED)
arrow(ax,(3.65,5.2),(5.2,4.3))
ax.text(4.55,4.85,'SDK / OTLP',fontsize=11,color=MUTED,bbox={'facecolor':'white','edgecolor':'none','pad':2})
arrow(ax,(7.7,5.2),(7.7,4.3))
ax.text(7.95,4.65,'Evaluation',fontsize=11,color=MUTED)
arrow(ax,(5.55,2.8),(5.55,1.8)); arrow(ax,(8.45,2.8),(8.45,1.8))
ax.text(6.9,2.22,'Persistence',ha='center',fontsize=11,color=MUTED)
ax.text(.3,1.6,'Cross-cutting controls',fontsize=13,weight='bold',color=BLUE)
ax.text(.3,.94,'Service identities\nSecret Manager · retention\nAccess policy and network reachability',fontsize=11.5,linespacing=1.65,va='center')
save(fig,'gcp-architecture')
print('Generated three SVG figures and PNG previews in',OUT)
