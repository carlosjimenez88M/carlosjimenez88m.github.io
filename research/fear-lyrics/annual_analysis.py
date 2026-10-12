"""Aggregate uniform paired readings and compute annual/period diagnostics offline."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT.parents[1] / '.research-cache/fear-lyrics/annual-review'
OUT = ROOT / 'results/annual-120'
TOPICS = ['surveillance_pursuit', 'collective_violence', 'loss_abandonment',
          'fragility_identity', 'uncertainty_change', 'relational_coercion', 'supernatural']
LABELS = ['Vigilancia y persecución', 'Violencia colectiva', 'Abandono y pérdida',
          'Fragilidad e identidad', 'Incertidumbre y cambio', 'Coerción relacional', 'Sobrenatural']
POSITIVE = {'explicit', 'implied'}
PERIODS = [1980, 1990, 2000, 2010, 2020]


def read(p): return json.loads(p.read_text())


def write(name, rows):
    p = OUT / name
    if p.suffix == '.json': p.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    elif rows:
        with p.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def validate(vote, case):
    problems = []; valid_ids = {b['id'] for b in case['blocks']}
    def supported(ids): return isinstance(ids, list) and bool(ids) and set(ids) <= valid_ids
    enums = {'fear_reference': {'explicit','implied','absent','uncertain'},
             'fear_polarity': {'affirmed','denied','hypothetical','mixed','absent','unclear'},
             'voice': {'first_singular','first_collective','third_person_story','dialogue_or_multiple','impersonal','unclear'},
             'stance': {'avoidance','resistance','care','ambivalence','resignation','other','unclear'},
             'arc_type': {'repetition','contrast','confrontation','turn','mixed','no_supported_arc','unresolved'}}
    for field, allowed in enums.items():
        if vote.get(field) not in allowed: problems.append('enum:' + field)
    if vote.get('fear_reference') in POSITIVE and not supported(vote.get('reference_support_ids')):
        problems.append('positive_reference_missing_support')
    topics = vote.get('subtopics', [])
    if not isinstance(topics,list) or len(topics)>2 or not set(topics)<=set(TOPICS+['other','none','uncertain']):
        problems.append('subtopic_schema')
    for topic in topics:
        if topic in TOPICS and not supported(vote.get('subtopic_support_ids',{}).get(topic)):
            problems.append('subtopic_missing_support:' + topic)
    steps = vote.get('arc_steps', []); links = vote.get('arc_links', [])
    arc_valid = True
    if vote.get('arc_type') not in {'no_supported_arc','unresolved'}:
        if not 2 <= len(steps) <= 3: arc_valid = False
        last = -1
        for step in steps:
            ids = step.get('ids', [])
            if not supported(ids): arc_valid = False; continue
            positions = [int(i[1:]) for i in ids]
            if min(positions) <= last: arc_valid = False
            last = max(positions)
            if len(step.get('reading','').split()) > 10: problems.append('step_paraphrase_too_long')
        if len(links) != max(0,len(steps)-1): arc_valid = False
        if any(link.get('from') != i or link.get('to') != i+1 for i,link in enumerate(links)): arc_valid = False
    if vote.get('value_support_ids') and not supported(vote['value_support_ids']): problems.append('value_ids')
    if len(vote.get('counterreading','').split())>12: problems.append('counterreading_too_long')
    return problems, arc_valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = read(PRIVATE/'mapping.json')
    cases = {r['case_id']:r for r in read(PRIVATE/'batch-a.json') + read(PRIVATE/'batch-b.json')}
    assert len(mapping)==len(cases)==120
    by_case = defaultdict(list); validation = []; public_votes = []
    for reader in ['a','b','c']:
        path=PRIVATE/f'reader-{reader}.json'
        records=read(path); expected={r['case_id'] for r in read(PRIVATE/f'batch-{reader}.json')}
        assert len(records)==len(expected)==80 and {r['case_id'] for r in records}==expected
        for vote in records:
            cid=vote['case_id']; meta=mapping[cid]; problems, arc_valid=validate(vote,cases[cid])
            validation.append(dict(case_id=cid,reader=reader,problems=problems,arc_chronology_valid=arc_valid))
            row={key:vote.get(key) for key in ['fear_reference','fear_polarity','voice','stance','subtopics','arc_type','value_reading','counterreading']}
            row.update(song_id=meta['song_id'],reader=reader,arc_chronology_valid=arc_valid,
                       category_schema_valid=not problems,
                       reference_support_ids=[meta['document_ids'][i] for i in vote.get('reference_support_ids',[]) if i in meta['document_ids']])
            row['subtopic_support_ids']={k:[meta['document_ids'][i] for i in ids if i in meta['document_ids']]
                                        for k,ids in vote.get('subtopic_support_ids',{}).items()}
            row['steps']=[dict(state=s.get('state'),reading=s.get('reading'),
                               source_document_ids=[meta['document_ids'][i] for i in s.get('ids',[]) if i in meta['document_ids']])
                          for s in vote.get('arc_steps',[])] if arc_valid else []
            row['links']=vote.get('arc_links',[]) if arc_valid else []
            row['value_support_ids']=[meta['document_ids'][i] for i in vote.get('value_support_ids',[]) if i in meta['document_ids']]
            public_votes.append(row);by_case[cid].append(row)
    annotations=[]
    for cid, pair in by_case.items():
        assert len(pair)==2
        m=mapping[cid]; positives=[p['fear_reference'] in POSITIVE and p['category_schema_valid'] for p in pair]
        absent=all(p['fear_reference']=='absent' and p['category_schema_valid'] for p in pair)
        fear=True if all(positives) else False if absent else None
        consensus={}
        for field in ['voice','stance','fear_polarity']:
            values=[p[field] for p in pair]
            consensus[field]=values[0] if values[0]==values[1] and all(p['category_schema_valid'] for p in pair) else 'unresolved'
        reference=[p['fear_reference'] for p in pair]
        consensus['fear_relevance']=reference[0] if reference[0]==reference[1] else 'explicit_or_implied' if fear else 'unresolved'
        common=set(pair[0]['subtopics']) & set(pair[1]['subtopics'])
        consensus['subtopics']=[k for k in TOPICS if k in common] if fear else []
        arcs=[p['arc_type'] for p in pair]
        consensus['arc']=arcs[0] if arcs[0]==arcs[1] and all(p['arc_chronology_valid'] and p['category_schema_valid'] for p in pair) else 'unresolved'
        if not fear: consensus['arc']='not_in_fear_subset'
        annotations.append({k:m[k] for k in ['song_id','title','artist','year','language']} |
                           dict(decade=m['year']//10*10,fear_reference_consensus=fear,
                                consensus=consensus,readers=[p['reader'] for p in pair],
                                one_reader_fear=any(positives),
                                one_reader_subtopics=[k for k in TOPICS if any(k in p['subtopics'] and p['fear_reference'] in POSITIVE for p in pair)]))
    annotations.sort(key=lambda r:(r['year'],r['song_id']))
    write('annotations.json',annotations);write('votes.json',public_votes);write('validation.json',validation)
    annual=[]
    for year in range(1980,2027):
        for language in ['balanced','en','es']:
            group=[r for r in annotations if r['year']==year and (language=='balanced' or r['language']==language)]
            for key,label in zip(TOPICS,LABELS):
                lower=sum(key in r['consensus']['subtopics'] for r in group)
                upper=sum(key in r['one_reader_subtopics'] for r in group)
                annual.append(dict(year=year,language=language,subtopic=key,label=label,songs=len(group),
                    fear_songs=sum(r['fear_reference_consensus'] is True for r in group),
                    reference_unresolved=sum(r['fear_reference_consensus'] is None for r in group),
                    count=lower,one_reader_count=upper,share=lower/len(group) if group else None,
                    one_reader_share=upper/len(group) if group else None,
                    languages_present='|'.join(sorted({r['language'] for r in group}))))
    write('annual_presence.csv',annual)
    periods=[]; styles=[]
    for d in PERIODS:
        group=[r for r in annotations if r['decade']==d]
        fear=[r for r in group if r['fear_reference_consensus'] is True]
        row=dict(decade=d,songs=len(group),fear_songs=len(fear),
                 reference_unresolved=sum(r['fear_reference_consensus'] is None for r in group),
                 fear_one_reader=sum(r['one_reader_fear'] for r in group))
        periods.append(row)
        for field in ['voice','stance','arc']:
            for value,n in Counter(r['consensus'][field] for r in fear).items():
                styles.append(dict(decade=d,field=field,value=value,count=n,fear_songs=len(fear),share=n/len(fear) if fear else None))
    write('periods.csv',periods);write('period_style.csv',styles)
    # Period diagnostics: every song weighs one; EN/ES equal. Artist removal keeps reads/fits fixed.
    sensitivity=[]
    for key,label in zip(TOPICS,LABELS):
        def rate(group,one=False):
            parts=[]
            for lang in ['en','es']:
                g=[r for r in group if r['language']==lang]
                if g:parts.append(sum(key in (r['one_reader_subtopics'] if one else r['consensus']['subtopics']) for r in g)/len(g))
            return float(np.mean(parts)) if parts else None
        first=[r for r in annotations if r['decade']==1980];last=[r for r in annotations if r['decade']==2020]
        difference=rate(last)-rate(first)
        loco=[]
        for artist in {r['artist'] for r in first+last}:
            lo=rate([r for r in first if r['artist']!=artist]);hi=rate([r for r in last if r['artist']!=artist])
            if lo is not None and hi is not None:loco.append(hi-lo)
        sensitivity.append(dict(subtopic=key,label=label,first_count=sum(key in r['consensus']['subtopics'] for r in first),
            last_count=sum(key in r['consensus']['subtopics'] for r in last),songs_each_period=24,
            observed_balanced_difference=difference,one_reader_difference=rate(last,True)-rate(first,True),
            leave_artist_min=min(loco),leave_artist_max=max(loco)))
    write('endpoint_sensitivity.csv',sensitivity)
    older=read(ROOT/'pilot-comparison/results/arc-panel/fieldwise-posthoc/song_annotations.json')
    current={r['song_id']:r for r in annotations};changes=[]
    for old in older:
        now=current[old['song_id']];old_ref=old['consensus']['fear_relevance']
        old_presence=True if old_ref in POSITIVE else False if old_ref=='absent' else None
        changes.append(dict(song_id=now['song_id'],title=now['title'],old_presence=old_presence,
                            new_presence=now['fear_reference_consensus'],
                            presence_agrees=old_presence==now['fear_reference_consensus']))
    write('previous_panel_sensitivity.csv',changes)
    summary=dict(songs=120,readings=240,readers=3,readings_each=2,
        fear_reference_counts={str(k):v for k,v in Counter(r['fear_reference_consensus'] for r in annotations).items()},
        fear_reference_subtypes=dict(Counter(r['consensus']['fear_relevance'] for r in annotations)),
        arcs_in_fear=dict(Counter(r['consensus']['arc'] for r in annotations if r['fear_reference_consensus'] is True)),
        observed_years=sorted({r['year'] for r in annotations}),empty_years=[y for y in range(1980,2027) if not any(r['year']==y for r in annotations)],
        periods=periods,schema_invalid_readings=sum(bool(r['problems']) for r in validation),
        arc_structure_invalid_readings=sum(not r['arc_chronology_valid'] for r in validation),
        prior_panel_presence_agreement=sum(r['presence_agrees'] for r in changes),prior_panel_songs=len(changes),
        project_api_calls=0,exact_model_revision='Not available; three processes of the same configured Codex assistant.',
        interpretation='Paired posthoc readings, not external gold or independent providers. Annual counts describe selected publications; missing years are absent observations.')
    write('summary.json',summary)
    write('receipt.json',dict(status='executed',protocol_sha256=hashlib.sha256((ROOT/'annual-reading-protocol.md').read_bytes()).hexdigest(),
        input_hashes={f'reader-{r}.json':hashlib.sha256((PRIVATE/f'reader-{r}.json').read_bytes()).hexdigest() for r in ['a','b','c']},
        mapping_sha256=hashlib.sha256((PRIVATE/'mapping.json').read_bytes()).hexdigest(),
        cohorts='120 uniformly read songs; older40-panel preserved as sensitivity, not pooled into new votes.',
        privacy='No lyrics; public categories, metadata, IDs and brief original paraphrases.',summary=summary))
    print(json.dumps({k:summary[k] for k in ['songs','readings','fear_reference_counts','arcs_in_fear','schema_invalid_readings','arc_structure_invalid_readings']},ensure_ascii=False))


if __name__=='__main__':main()
