"""Static publication figures from observed results, with editable SVG output."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from collect import ROOT, OUT, ALBUMS

DEST = ROOT / 'static/img/album-memory'
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'figure.facecolor': '#faf9f6',
                     'axes.facecolor': '#faf9f6'})
COLORS = dict(full='#293c50', recent3='#bc623d', summary='#6a7962', selective6='#6b5d91')
LABELS = dict(full='Full archive', recent3='Recent 3', summary='Rolling summary', selective6='Selective 6')


def save(fig, name):
    for ext in ['svg', 'png']:
        fig.savefig(DEST / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def architecture():
    fig,ax=plt.subplots(figsize=(11,6))
    ax.set(xlim=(0,11),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color='#e5e9e4'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color,edgecolor='#69766c'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#23332b')
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#69766c',linewidth=1.4))
    ax.text(.2,5.65,'Memory is stored state. Context is a deliberate selection.',fontsize=17,color='#23332b')
    box(.2,3.5,2,1,'Genius lyrics\nPrivate source cache')
    box(2.8,3.5,2.1,1,'Anonymous cards\nClaims + line references')
    box(5.5,3.5,2.1,1,'LangGraph archive\nOrdered track state')
    box(8.2,3.5,2.4,1,'Select memory\nFull / recent / summary\n/ selective retrieval')
    arrow((2.2,4),(2.8,4));arrow((4.9,4),(5.5,4));arrow((7.6,4),(8.2,4))
    box(8.2,1.8,2.4,1,'Generate interpretation\nCitations or abstention')
    box(5.5,1.8,2.1,1,'Validate + judge\nEvidence and coverage')
    arrow((9.4,3.5),(9.4,2.8));arrow((8.2,2.3),(7.6,2.3))
    box(.2,1.8,4.7,1,'MLflow\nPrompt versions · graph traces · provider tokens\nPer-case runs · evaluation artifacts','#e4e7ed')
    arrow((5.5,2.3),(4.9,2.3))
    ax.text(.2,.9,'Separate thread IDs isolate cases. Checkpoints do not automatically reduce prompt tokens.',fontsize=10,color='#56615a')
    ax.text(.2,.5,'Diagram of the implemented pilot. Lyrics stay private; published evidence uses paraphrases.',fontsize=10,color='#56615a')
    save(fig,'memory-architecture')


def main():
    architecture()
    data = pd.read_csv(OUT / 'policy_means.csv')
    albums = pd.read_csv(OUT / 'album_means.csv')
    fig, (ax, evidence_ax) = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained')
    for _, row in data.iterrows():
        policy = row['policy']
        sub = albums[albums.policy == policy]
        ax.scatter(sub.input_tokens, sub.coverage, s=35, alpha=.35, color=COLORS[policy])
        ax.scatter(row.input_tokens, row.coverage, s=140, color=COLORS[policy], label=LABELS[policy])
        evidence_ax.scatter(sub.input_tokens, sub.evidence_recall, s=35, alpha=.35, color=COLORS[policy])
        evidence_ax.scatter(row.input_tokens, row.evidence_recall, s=140, color=COLORS[policy])
    ax.set(xlabel='Generation input tokens per answer (provider reported)',
           ylabel='Model-judged coverage (0–4)', ylim=(-.15, 4.3))
    ax.set_title('A judge can reward a plausible answer', loc='left', pad=20, fontsize=12)
    evidence_ax.set(xlabel='Generation input tokens per answer',
                    ylabel='Designated evidence IDs cited (fraction)',ylim=(-.05,1.05))
    evidence_ax.set_title('Check the evidence separately',loc='left',pad=20,fontsize=12)
    evidence_ax.grid(alpha=.15)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=.15)
    fig.text(.08, -.06, 'Large dots: equal-album means. Small dots: individual albums. 12 questions × 3 repetitions per policy.\nBoth diagnostics are limited: silver references are fallible, and alternative evidence pairs may be defensible.', fontsize=9)
    save(fig, 'tokens-coverage')

    diagnostics = json.loads((OUT / 'arc_diagnostics.json').read_text())
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), layout='constrained')
    for ax, album, arc in zip(axes, ALBUMS, diagnostics):
        values = np.array(arc['profiles']).T
        im = ax.imshow(values, aspect='auto', vmin=0, vmax=3, cmap='YlGnBu', interpolation='nearest')
        ax.set_yticks(range(len(arc['themes'])), [t.replace('_', ' ') for t in arc['themes']], fontsize=9)
        ax.set_xticks(range(len(album['tracks'])), range(1, len(album['tracks']) + 1))
        ax.set_title(f"{album['artist']} · {album['album']}", loc='left', fontsize=12)
        ax.set_xlabel('Track position' + (' (Tightrope is unit 12)' if album['id']=='roach' else ''), fontsize=9)
    fig.colorbar(im, ax=list(axes), shrink=.5, label='Model annotation: 0 absent → 3 central', ticks=[0,1,2,3])
    fig.suptitle('Thematic recurrence in four lyric sequences', x=.02, ha='left', fontsize=17)
    save(fig, 'theme-profiles')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (album, arc) in enumerate(zip(ALBUMS, diagnostics)):
        ax.plot([arc['shuffled_p025'], arc['shuffled_p975']], [i, i], color='#b9beb9', linewidth=9, solid_capstyle='round')
        ax.scatter(arc['shuffled_mean'], i, color='#68736b', marker='|', s=150)
        ax.scatter(arc['canonical_adjacent_cosine'], i, color='#293c50', s=85, zorder=3)
    ax.set_yticks(range(4), [a['album'] for a in ALBUMS])
    ax.invert_yaxis()
    ax.set(xlabel='Mean cosine similarity between adjacent theme profiles', xlim=(0, 1))
    ax.set_title('Does the published order show thematic continuity?', loc='left', pad=20)
    fig.subplots_adjust(bottom=.26)
    fig.text(.12, .02, 'Dark dots: album order. Gray ranges: central 95% of 10,000 random orders.\nExploratory diagnostic of model annotations; not proof of a narrative or compositional intent.', fontsize=9)
    save(fig, 'order-continuity')


if __name__ == '__main__':
    main()
