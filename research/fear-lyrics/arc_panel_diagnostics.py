#!/usr/bin/env python3
"""Posthoc field-wise analysis of existing panel receipts, no new API calls.

Categorical type/enum validity is distinct from evidence-span chronology. This
extension preserves primary strict trajectory validation and all prior results.
It publishes categorical codes only from otherwise rejected raw proposals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import analyze
import arc_panel as panel
import llm_review as api

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=Path(__file__).resolve().parent/'pilot-comparison/results/arc-panel'
FIELDS=['voice','stance','fear_relevance','fear_polarity','subtopics']


def typed_vote(parsed,field):
    if not isinstance(parsed,dict):return None
    value=parsed.get(field)
    if field=='subtopics':
        if not isinstance(value,list) or not 1<=len(value)<=2 or len(set(value))!=len(value) or any(v not in panel.SUBTOPICS for v in value):return None
        if len(value)>1 and any(v in {'none','uncertain'} for v in value):return None
        return sorted(value)
    return value if isinstance(value,str) and value in panel.ENUMS[field] else None


def majority(votes,field):
    if field=='subtopics':
        count=Counter(v for vote in votes if vote is not None for v in vote)
        values=[v for v,n in count.items() if n>=2]
        return sorted(values) if 1<=len(values)<=2 else ['unresolved']
    count=Counter(v for v in votes if v is not None)
    value,n=count.most_common(1)[0] if count else ('unresolved',0)
    return value if n>=2 else 'unresolved'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir',type=Path,default=DEFAULT)
    parser.add_argument('--corpus',type=Path,default=panel.PRIVATE/'pilot-40/corpus.jsonl')
    parser.add_argument('--authorized-bundle',type=Path,default=panel.PRIVATE/'pilot-40/embedding-input.json')
    args=parser.parse_args();source=args.panel_dir
    output=source/'fieldwise-posthoc'
    protocol=json.loads((source/'protocol.json').read_text());rows=json.loads((source/'panel_votes.json').read_text())
    prepared,_=panel.prepare(args.corpus,args.authorized_bundle);lookup={r['song_id']:r for r in prepared}
    if len(prepared)!=protocol['songs'] or api.digest([(r['song_id'],r['lyrics_hash']) for r in prepared])!=protocol['corpus_identity']:
        raise ValueError('Diagnostic corpus does not match frozen primary panel.')
    metadata={r['song_id']:r for r in [json.loads(x) for x in args.corpus.read_text().splitlines() if x.strip()]}
    field_votes=[];raw=[];invalid_reasons=Counter()
    for row in rows:
        item=json.loads((panel.PRIVATE/'arc-panel-cache'/('openai-'+row['request_hash']+'.json')).read_text()) if row.get('request_hash') else {}
        parsed=item.get('parsed',{})
        vote={field:typed_vote(parsed,field) for field in FIELDS}
        strict_arc=row['reading']['arc'] if row['status']=='executed' else None
        if row['status']=='executed_invalid_panel_schema':
            try:panel.validate(parsed,lookup[row['song_id']])
            except ValueError as e:invalid_reasons[str(e)]+=1
        record={key:row[key] for key in ('song_id','anonymous_song_id','model_id','stage','year','decade','language','status')}
        record.update(primary_strict_status=row['status'],categorical_votes=vote,category_validity={f:vote[f] is not None for f in FIELDS},
                      strict_arc_vote=strict_arc,strict_arc_anchor_structure_valid=row['status']=='executed',request_hash=row.get('request_hash'))
        field_votes.append(record);raw.append(dict(row=row,parsed=parsed,votes=vote,strict_arc=strict_arc))
    annotations=[]
    for song in prepared:
        group=[r for r in raw if r['row']['song_id']==song['song_id']]
        first=[r for r in group if r['row']['stage']=='independent_reader']
        consensus={f:majority([r['votes'][f] for r in group],f) for f in FIELDS}
        consensus['arc']=majority([r['strict_arc'] for r in group],'arc')
        fear=consensus['fear_relevance'];subs=consensus['subtopics']
        fear_reference=True if fear in {'explicit','implied'} else False if fear=='absent' else None
        if fear_reference is True and subs!=['unresolved']:
            fear_conditioned=subs
        elif fear_reference is False:
            fear_conditioned=['no_supported_fear_reference']
        else:
            fear_conditioned=['unresolved']
        joint_votes=[]
        for item in group:
            f=item['votes']['fear_relevance'];s=item['votes']['subtopics']
            if f in {'explicit','implied'} and s is not None:joint_votes.append(s)
            elif f=='absent':joint_votes.append(['no_supported_fear_reference'])
            else:joint_votes.append(None)
        paired_joint=majority(joint_votes,'subtopics')
        unresolved=[f for f,v in consensus.items() if v=='unresolved' or v==['unresolved']]
        agreement={f:len(first)==2 and all(r['votes'][f] is not None for r in first) and first[0]['votes'][f]==first[1]['votes'][f] for f in FIELDS}
        agreement['arc']=len(first)==2 and all(r['strict_arc'] is not None for r in first) and first[0]['strict_arc']==first[1]['strict_arc']
        annotations.append(dict(song_id=song['song_id'],year=song['year'],decade=song['decade'],language=song['language'],
                                consensus=consensus,unresolved_fields=unresolved,initial_field_agreement=agreement,
                                fear_reference_consensus=fear_reference,fear_conditioned_subtopics=fear_conditioned,
                                paired_joint_vote_subtopics=paired_joint,
                                valid_votes_by_field={f:sum(r['votes'][f] is not None for r in group) for f in FIELDS},
                                valid_strict_arc_votes=sum(r['strict_arc'] is not None for r in group),third_review_attempted=len(group)>2))
    agreements=[]
    for field in [*FIELDS,'arc']:
        first_rows=[r for r in raw if r['row']['stage']=='independent_reader']
        valid_count=sum((r['strict_arc'] if field=='arc' else r['votes'][field]) is not None for r in first_rows)
        agreements.append(dict(field=field,songs=len(annotations),initial_votes=len(first_rows),valid_initial_enum_votes=valid_count,
                               initial_agree_songs=sum(r['initial_field_agreement'][field] for r in annotations),
                               majority_resolved_songs=sum(field not in r['unresolved_fields'] for r in annotations),
                               unresolved_songs=sum(field in r['unresolved_fields'] for r in annotations),
                               validity='Whole strict chronology/anchor schema' if field=='arc' else 'Field type/enum only;not semantic correctness'))
    temporal=[]
    for decade in sorted({r['decade'] for r in annotations}):
        for language in ('en','es'):
            group=[r for r in annotations if r['decade']==decade and r['language']==language]
            for field in [*FIELDS,'arc']:
                values=list(panel.SUBTOPICS) if field=='subtopics' else panel.ENUMS[field]
                counts=Counter(v for r in group for v in (r['consensus'][field] if field=='subtopics' else [r['consensus'][field]]))
                for value in [*values,'unresolved']:
                    temporal.append(dict(decade=decade,language=language,dimension=field,value=value,songs=len(group),count=counts[value],share=counts[value]/len(group) if group else None))
    cases=[]
    for annotation in annotations:
        song_id=annotation['song_id'];strict=[r for r in rows if r['song_id']==song_id and r['status']=='executed']
        if annotation['consensus']['arc'] in {'unresolved','no_supported_arc','unclear'}:continue
        supporting=[r for r in strict if r['reading']['arc']==annotation['consensus']['arc'] and r['reading']['steps'] and r['reading']['links']]
        if len(supporting)<2:continue
        song=metadata[song_id]
        cases.append(dict(song_id=song_id,title=song['title'],artist=song['artist'],year=song['year'],decade=annotation['decade'],language=annotation['language'],
                          consensus=annotation['consensus'],supporting_vote_count=len(supporting),
                          models=[r['model_id'] for r in supporting],stages=[r['stage'] for r in supporting],
                          readings=[r['reading'] for r in supporting],
                          selection='All cases with>=2strict-valid anchored votes for same arc category;illustrative support list,not representative temporal sample.'))
    analyze.write_json(output/'categorical_votes.json',field_votes)
    analyze.write_json(output/'song_annotations.json',annotations)
    analyze.write_csv(output/'field_agreement.csv',agreements)
    analyze.write_csv(output/'decade_readings.csv',temporal)
    concern_temporal=[]
    concern_summary=[]
    for decade in sorted({r['decade'] for r in annotations}):
        for language in ('en','es','balanced'):
            group=[r for r in annotations if r['decade']==decade and (language=='balanced' or r['language']==language)]
            for view,key in (('all_concerns','subtopics'),('fear_consensus_cross','fear_conditioned_subtopics'),('fear_paired_joint_votes','paired_joint_vote_subtopics')):
                counts=Counter(v for r in group for v in (r['consensus']['subtopics'] if key=='subtopics' else r[key]))
                for value in [*panel.SUBTOPICS,'no_supported_fear_reference','unresolved']:
                    concern_temporal.append(dict(decade=decade,language=language,view=view,value=value,
                                                label=panel.SUBTOPICS.get(value,'Sin referencia al miedo' if value=='no_supported_fear_reference' else 'Sin resolución'),
                                                songs=len(group),count=counts[value],share=counts[value]/len(group) if group else None,
                                                denominator='All selected songs in period/language;balanced pools equally-sizedEN/EScells',
                                                fear_reference_definition='Consensusexplicit/implied,includingdenied/hypothetical polarity;not experienced/affirmed fear'))
            concern_summary.append(dict(decade=decade,language=language,songs=len(group),
                                        fear_relevant_consensus=sum(r['fear_reference_consensus'] is True for r in group),
                                        fear_absent_consensus=sum(r['fear_reference_consensus'] is False for r in group),
                                        fear_reference_unresolved=sum(r['fear_reference_consensus'] is None for r in group),
                                        subtopics_unresolved=sum(r['consensus']['subtopics']==['unresolved'] for r in group)))
    analyze.write_csv(output/'concern_presence_by_period.csv',concern_temporal)
    analyze.write_json(output/'concern_presence_summary.json',dict(periods=concern_summary,
                        consensus_cross='Presence of each voted concern gated by separate consensusfear_relevance explicit/implied;doesnotrequire samevoters jointly endorse both fields.',
                        paired_joint_votes='Sensitivity:eachlabel requires>=2individualvotes that bothname thelabel and classifyfearreference explicit/implied.',
                        all_concerns='Includes motifs in controls with fearabsence,so cannot be read as fearprevalence.',
                        denominator='All40selectedsongs:8perperiod,4EN+4ES;unresolved/missing always retained. Multilabel totals can exceed100%.',
                        scope='Descriptive automated interpretations;fearreference includes negation/aspiration and isnot subjective experienced fear. Thirdcall is correlatedmini,not independent model.'))
    analyze.write_json(output/'anchored_cases.json',cases)
    diagnostic=dict(status='executed_offline_posthoc',primary_panel_identity=api.digest(protocol),primary_status_unchanged=Counter(r['status'] for r in rows),
                    interpretation='Categorical enum/type validity is measured separately from trajectory chronology. Existing votes only, no additional API or relaxed arc validator.',
                    categories_origin='Frozen ontology,model readings;enum validity doesnot establish semantic correctness.',
                    temporal_scope='All selected40songs remain denominator;unresolved/missing retained;subtopics multilabel may sum above100%.',
                    third_review_scope='Uses the existing20blind critic calls selected by the original strict/core disagreement protocol,not newly selected from posthoc semantic fields.',
                    invalid_primary_reasons=dict(invalid_reasons),fields=agreements,anchored_cases=len(cases),new_api_calls=0,new_cost_usd=0)
    analyze.write_json(output/'summary.json',diagnostic)
    print(json.dumps(dict(status='fieldwise_diagnostic_finished',songs=len(annotations),fields=agreements,anchored_cases=len(cases),primary_invalid_reasons=dict(invalid_reasons))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='stopped',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
