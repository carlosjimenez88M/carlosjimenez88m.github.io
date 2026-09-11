"""Package already-written qualitative AI review; does not generate reviewer judgments."""
import json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
rows=json.loads((BASE.parent/'results/runs.json').read_text())
notes=json.loads((BASE/'notes.json').read_text())
items=[]
for album,tasks in notes.items():
 for task,comments in tasks.items():
  for policy,note in zip(['adaptive','embedding','full'],comments):
   r=next(r for r in rows if r['album']==album and r['task']==task and r['policy']==policy and (policy!='embedding' or r['cap']==1600) and r['threshold']==3)
   items.append(dict(case_id=r['case_id'],question=r['question']['question'],answer=r['answer'],review_note=note,online_accepted=r['metrics']['model_verified_supported'],online_verification=r['verification']))
assert len(items)==72 and len({r['case_id'] for r in items})==72
report=dict(review_type='Qualitative AI editorial audit',reviewer='Codex AI assistant; not a human researcher',date='2026-09-11',blinded=False,independent_validation=False,scope='All 72 final outputs for full, embedding 1600 and adaptive threshold 3; complete 168 source-card claims read by album. No raw-lyric reannotation. Original policy results were known before review. No new model experiment or human ratings.',criteria=['Support for each assertion in cited source claims','Correct entity and source attribution, including answer prose','Requested relation and temporal scope','Defensibility without assuming narrator identity or causal chronology','Whether unsupported premises warrant withholding an answer'],scoring='Qualitative notes only; no replacement accuracy, ordinal ratings, or inter-rater agreement inferred.',runs_sha256=hashlib.sha256((BASE.parent/'results/runs.json').read_bytes()).hexdigest(),items=items)
(BASE/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
parts=['# Qualitative AI editorial audit\n\n72 responses, reviewed against the complete source-card claims. This is a post-hoc, non-blind AI review, not human validation or a replacement benchmark. The author remains responsible for adjudication. No numerical quality ratings or inter-rater statistics were invented.\n\nThe review checks claim support, attribution in prose and structured fields, the requested relation, temporal scope, interpretive restraint and abstention. Source cards are fallible model annotations; this audit does not validate them against lyrics.\n']
for i in items:parts.append('## '+i['case_id']+'\n\n'+i['review_note']+'\n')
(BASE/'README.md').write_text('\n'.join(parts))
print('Packaged 72 qualitative AI review notes; original runs unchanged.')
