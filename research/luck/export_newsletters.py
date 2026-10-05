"""Export complete essays for Buttondown; no API calls or secret access."""
import html
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE='https://carlosdanieljimenez.com'
INLINE={
    'A':'A', 'X':'X', 'U':'U', 'I_0':'I₀', 'I_1':'I₁', 'M':'M',
    r'\Pr(Y\ge y\mid A,I_0,M)':'Pr(Y ≥ y | A, I₀, M)',
    'p':'p',r'K\sim\operatorname{Binomial}(20,p)':'K ∼ Binomial(20, p)',
    '20p':'20p','20p(1-p)':'20p(1 − p)',
    r'p\sim\operatorname{Beta}(1,1)':'p ∼ Beta(1, 1)',
    r'p\mid\text{earlier data}\sim\operatorname{Beta}(13,9)':'p | earlier data ∼ Beta(13, 9)',
    '0.60':'0.60','13/22':'13/22',r'P\sim\operatorname{Beta}(2.4,1.6)':'P ∼ Beta(2.4, 1.6)',
    '0.20':'0.20',r'\theta_j':'θⱼ',
    r'\operatorname{Cov}(\theta,Y)=16':'Cov(θ, Y) = 16',
    r'\operatorname{Var}(Y)=52':'Var(Y) = 52',
    r'\Pr(\max_jY_j\le t)=F(t)^m':'Pr(maxⱼ Yⱼ ≤ t) = F(t)ᵐ',
    'm':'m','j^*':'j*','S':'S','S+U>2':'S + U > 2',
    'a':'a','b':'b','P':'P',r'P\sim\operatorname{Beta}(1,1)':'P ∼ Beta(1, 1)',
    '[0,1]':'[0, 1]',r'(1+499\times2/3)/500=0.6673':'(1 + 499 × 2/3)/500 = 0.6673',
    r'\mathbb E[Y(1)-Y(0)]':'E[Y(1) − Y(0)]',
    'Y_i(1)':'Yᵢ(1)','Y_i(0)':'Yᵢ(0)','i':'i',
    'n=40':'n = 40','k=3':'k = 3','r=4':'r = 4','D_{itr}':'Dᵢₜᵣ',
    r'\Delta=0.02':'Δ = 0.02',r'U_i\sim N(0,0.06^2)':'Uᵢ ∼ Normal(0, 0.06²)',
    r'V_{it}\sim N(0,0.04^2)':'Vᵢₜ ∼ Normal(0, 0.04²)',
    r'E_{itr}\sim N(0,0.04^2)':'Eᵢₜᵣ ∼ Normal(0, 0.04²)',
    '1/n':'1/n','1/(nk)':'1/(nk)','1/(nkr)':'1/(nkr)',
    't':'t','t_{39}':'t with 39 degrees of freedom','q':'q','Y':'Y','(q-Y)^2':'(q − Y)²','q=p':'q = p',
    'D':'D','p=0.50':'p = 0.50','m':'m','v':'v','n':'n',
    r'(1+499\times1/2)/500=0.501':'(1 + 499 × 1/2)/500 = 0.501',
}

def main():
    releases=json.loads((ROOT/'research/luck/releases.json').read_text())['releases']
    out=ROOT/'research/luck/newsletters';out.mkdir(exist_ok=True)
    for release in releases:
        source=(ROOT/'content/post'/f"{release['slug']}.md").read_text()
        body=source.split('---',2)[2].strip()
        # Buttondown supports display math, but not inline math. Preserve the
        # former in its native block and adapt the latter to readable notation.
        blocks=[]
        def block(m):
            blocks.append("<div class='buttondown-block-math'>[ "+html.escape(m[1])+" ]</div>")
            return f'LUCKMATHBLOCK{len(blocks)-1}END'
        body=re.sub(r'\$\$(.*?)\$\$',block,body,flags=re.S)
        body=re.sub(r'\$([^$\n]+)\$',lambda m:INLINE[m[1]],body)
        for index,value in enumerate(blocks):body=body.replace(f'LUCKMATHBLOCK{index}END',value)
        def figure(m):
            attrs=dict(re.findall(r'(\w+)="([^"]*)"',m[1]))
            src=BASE+attrs['src'].replace('.svg','.png')
            return f"![{attrs['alt']}]({src})\n\n*{attrs['caption']}*"
        body=re.sub(r'\{\{< figure (.*?) >\}\}',figure,body)
        series=['**A Statistical Account of Luck**']
        for r in releases:
            title=r['subject'].split(' — Part')[0]
            if r['part']<=release['part']:
                series.append(f"{r['part']}. [{title}]({BASE}/post/{r['slug']}/)")
            else:
                date=r['blog_at'].split('T')[0]
                series.append(f"{r['part']}. {title} — scheduled for {date}")
        body=body.replace('{{< luck-series >}}','\n\n'.join(series))
        body=re.sub(r'\]\((/[^)]+)\)',lambda m:']('+BASE+m[1]+')',body)
        body=("<!-- buttondown-editor-mode: plaintext -->\n\n"
              +f"[Read this essay on the blog]({BASE}/post/{release['slug']}/)\n\n"+body+'\n')
        assert '{{<' not in body and '$' not in body
        (out/f"part-{release['part']}.md").write_text(body)
        print(f"Exported complete newsletter part {release['part']}: {len(body.split())} words")

if __name__=='__main__':main()
