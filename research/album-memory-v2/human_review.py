"""Policy-blinded 72-item human review packet; no fabricated human labels."""
import random,html
from prepare import *
def main():
 rows=read(OUT/'runs.json')
 sample=[r for r in rows if r['policy']=='full' or (r['policy']=='embedding' and r['cap']==1600) or (r['policy']=='adaptive' and r['threshold']==3)]
 assert len(sample)==72
 random.Random(83).shuffle(sample);items=[];key=[]
 for i,r in enumerate(sample,1):
  ident=f'R{i:02}';key.append(dict(item_id=ident,case_id=r['case_id']))
  cards=cards_for(r['album'])
  items.append(dict(item_id=ident,album=next(a['album'] for a in ALBUMS if a['id']==r['album']),question=r['question']['question'],task=r['task'],answer=r['answer'],source_claims=[dict(id=f"{c['id']}:{v['claim_id']}",paraphrase=v['paraphrase']) for c in cards for v in c['claims']]))
 write(PRIVATE/'human-review-key.json',key)
 dest=Path(__file__).parent/'human-review';dest.mkdir(exist_ok=True)
 write(dest/'items.json',items)
 parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Album memory — blind review</title><style>body{max-width:850px;margin:40px auto;padding:0 20px;font:17px/1.65 system-ui;color:#23332b;background:#faf9f6}section{border-top:1px solid #ccc;padding:25px 0}textarea{width:100%;min-height:65px}label{display:inline-block;margin:8px 15px 8px 0}select,input,button{font:inherit;padding:6px}button{position:sticky;top:10px;background:#fff;border:1px solid #555}summary{cursor:pointer}pre{white-space:pre-wrap;font:14px/1.5 monospace}.notice{background:#e9eee9;padding:15px}</style><h1>Album memory: blind review</h1><p class="notice">72 responses. Policies and automated scores are hidden. Judge against the supplied analytical claims, which are model annotations rather than lyric ground truth. Do not infer author intent. Leave an item unrated if uncertain.</p><p>Score each dimension 0–2: 0 unsupported or fails the requirement; 1 mixed or materially incomplete; 2 supported within the supplied evidence. <strong>Support</strong>: are the substantive assertions supported by the supplied evidence? <strong>Attribution</strong>: do the cited claims support the assertions assigned to them? <strong>Requested relation</strong>: does the response establish the requested contrast, return or qualification and its temporal scope? <strong>Narrative defensibility</strong>: is the interpretation bounded without assuming a shared narrator? For an abstention, assess whether withholding the interpretation is warranted by the full cards, and explain in notes.</p><label>Reviewer ID <input id="reviewer" placeholder="A"></label><button onclick="download()">Download ratings JSON</button><p>Nothing is transmitted. Keep this page open while rating, then download your file. Two reviewers should work independently and exchange ratings only afterward.</p>''']
 for item in items:
  esc=lambda x:html.escape(str(x))
  parts.append(f'<section data-id="{item["item_id"]}"><h2>{item["item_id"]} · {esc(item["album"])}</h2><p>{esc(item["question"])}</p><h3>Response</h3><p>{esc(item["answer"]["answer"])}</p><p>Abstained: {item["answer"]["abstained"]}</p><pre>{esc(json.dumps(item["answer"]["claims"],ensure_ascii=False,indent=2))}</pre><details><summary>Source claims for the complete album</summary><pre>{esc(json.dumps(item["source_claims"],ensure_ascii=False,indent=2))}</pre></details>')
  for field in ['support','attribution','requested_relation','narrative_defensibility']:
   parts.append(f'<label>{field.replace("_"," ")} <select data-field="{field}"><option value="">Unrated</option>'+''.join(f'<option>{v}</option>' for v in range(3))+'</select></label>')
  parts.append('<label>Should have abstained <select data-field="should_have_abstained"><option value="">Unrated</option><option value="yes">Yes</option><option value="no">No</option></select></label>')
  parts.append('<textarea data-field="notes" placeholder="Reason, source addresses, uncertainty"></textarea></section>')
 parts.append('''<script>function download(){const items=[...document.querySelectorAll('section')].map(s=>({item_id:s.dataset.id,...Object.fromEntries([...s.querySelectorAll('[data-field]')].map(e=>[e.dataset.field,e.tagName==='SELECT'?(e.value===''?null:(e.dataset.field==='should_have_abstained'?e.value:Number(e.value))):e.value]))}));const data={reviewer_id:document.getElementById('reviewer').value,items};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='album-memory-human-ratings.json';a.click();URL.revokeObjectURL(a.href);}</script></html>''')
 (dest/'index.html').write_text(''.join(parts))
 (dest/'README.md').write_text('Open index.html locally. No ratings have been collected. Keep reviewer files independent. The policy key remains private until review is complete. This evaluates interpretations against source cards, not against a human transcription of the lyrics.\n')
 print('Prepared 72 blinded items; no human labels generated.')
if __name__=='__main__':main()
