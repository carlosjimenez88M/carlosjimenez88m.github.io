#!/usr/bin/env python3
"""Freeze first; then optionally execute twelve NEW fictional scope controls.

The benchmark compares four models on identical data and expected behavior.
Expected labels are assistant-authored, not independently validated human gold.
Never sends lyrics, real medical facts or copyrighted derivations to Anthropic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from dotenv import dotenv_values

import analyze
import llm_comparison as comparison
import llm_review as api
import scope_controls

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/".research-cache/fear-lyrics"
PUBLIC=Path(__file__).resolve().parent/"results/llm-comparison-detail-controls-v2"
VERSION="fictional_requested_detail_controls_v2_20261011"


def specification(models):
    return dict(version=VERSION,controls=scope_controls.controls(),models=models,
                prompt_sha256=api.digest(scope_controls.SYSTEM),input_identity=api.digest(scope_controls.controls()),
                expected_behavior="Voice-report questions supported by explicit wording; verified-performer questions insufficient without source/linkage; hypothetical-authoritative records support their stated fictional subject/detail.",
                authorship="New assistant-authored fictional cases and expectations, specified before these requests. No human adjudication or external benchmark.",
                scope="Four slot types times three provenance scopes; 12 English cases, dependent within slot. This tests a predeclared mechanism, not bilingual generalization or global provider performance.",
                privacy="All input text is original synthetic fiction and may be public; no real lyrics, copyrighted derivation, diagnosis, medication or medical advice.",
                inference_controls="Same textual prompts/payloads for all four models; temperature0, maximum400 output tokens; OpenAI strict JSON schema, Anthropic prompt-constrained JSON; no tools or thinking.",
                revision="V1 did not enumerate JSON fields in the textual prompt; only OpenAI received the API schema, so Anthropic valid JSON had different field names. V2 specifies the same five fields textually for ALL models; cases and expected labels are unchanged. V1 retained as adapter-format failure, not semantic model failure. This is a transparent post-execution interface repair, not a new independent test.")


def freeze(output,models):
    expected=specification(models);path=output/"controls.json"
    if path.exists():
        if api.digest(json.loads(path.read_text()))!=api.digest(expected):raise ValueError("Frozen control specification differs; do not revise after seeing results.")
    else:analyze.write_json(path,expected)
    return expected


def self_test():
    rows=scope_controls.controls()
    assert len(rows)==12 and sum(r['expected_support_decision']=='supported' for r in rows)==8
    for row in rows:
        payload=row['payload']
        _,body=api.api_body('anthropic',scope_controls.SYSTEM,payload,synthetic=True,model_id='claude-haiku-4-5-20251001')
        assert body['model']=='claude-haiku-4-5-20251001'
        if row['expected_evidence_id']:assert row['expected_evidence_id'] in {b['document_id'] for b in payload['evidence_blocks']}
    try:
        api.api_body('anthropic',scope_controls.SYSTEM,{'original_question':'Arbitrary','evidence_blocks':[]},synthetic=True)
        raise AssertionError('Original-control scope guard failed')
    except ValueError:pass
    print('New-scope control self-test passed:12 original cases,8supported/4insufficient,source IDs and fixed Anthropic payload guard.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--models',nargs='+',default=list(comparison.REGISTRY));parser.add_argument('--output',type=Path,default=PUBLIC)
    parser.add_argument('--max-usd',type=float,default=.80)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not set(args.models)<=set(comparison.REGISTRY):raise ValueError('Unknown or unauthorized provider/model.')
    if args.freeze:
        spec=freeze(args.output,args.models)
        print(json.dumps(dict(status='controls_frozen_before_API',input_identity=spec['input_identity'],controls=12,expected_supported=8,expected_insufficient=4)),flush=True)
    if not args.run:
        print(json.dumps(dict(mode='plan_only',cases=12,models=args.models,planned_requests=12*len(args.models))),flush=True);return
    path=args.output/'controls.json'
    if not path.exists():raise ValueError('Run --freeze before executing; expected behavior must be saved first.')
    spec=freeze(args.output,args.models)
    ledger=api.Ledger(PRIVATE/'cost-ledger-llm.json',args.max_usd);before=ledger.report()
    s=dotenv_values(ROOT/'.env');keys={'openai':s.get('OPENAI_API_KEY'),'anthropic':s.get('ANTHROPIC_API_KEY') or s.get('ANTROPIC_API_KEY')}
    cache=PRIVATE/'llm-review';rows=[]
    for model_id in args.models:
        config=comparison.REGISTRY[model_id];provider=config['provider'];blocked=None
        for index,control in enumerate(spec['controls']):
            row={k:control[k] for k in ('control_id','slot','scope','language','expected_support_decision','expected_evidence_id')}
            row.update(model_id=model_id,provider=provider,input_identity=api.digest(control['payload']),prompt_sha256=spec['prompt_sha256'])
            if blocked or not keys[provider]:row['status']='not_attempted_after_'+blocked if blocked else 'key_not_configured'
            else:
                item=api.execute(provider,scope_controls.SYSTEM,control['payload'],keys[provider],ledger,cache,synthetic=True,
                                 schema=api.REVIEW_SCHEMA,model_id=config['api_id'],rates=config['rates'])
                row.update({k:v for k,v in item.items() if k!='parsed'})
                if item['status']=='executed':
                    ids=[b['document_id'] for b in control['payload']['evidence_blocks']]
                    try:
                        row.update(api.public_review(item['parsed'],ids))
                        row['decision_matches_expected']=row['support_decision']==control['expected_support_decision']
                        row['expected_support_cited']=(control['expected_evidence_id'] in row['cited_ids']) if control['expected_evidence_id'] else None
                    except ValueError:row['status']='executed_invalid_schema'
                if item['status'] in {'http_401','http_402','http_403','http_404','http_429','network_unknown'}:blocked=item['status']
            rows.append(row);analyze.write_json(args.output/'runs.json',rows)
            print(json.dumps(dict(model=model_id,progress=index+1,total=12,status=row['status'])),flush=True)
    summaries=[]
    for model_id in args.models:
        group=[r for r in rows if r['model_id']==model_id];valid=[r for r in group if r['status']=='executed']
        summaries.append(dict(model_id=model_id,requested=len(group),valid=len(valid),status_counts=dict(Counter(r['status'] for r in group)),
                              assistant_expected_matches=sum(r['decision_matches_expected'] for r in valid),expected_constraints=len(valid),
                              verified_missing_safe_abstentions=sum(r['abstained'] for r in valid if r['scope']=='verified_missing_source'),
                              voice_report_supported=sum(r['support_decision']=='supported' for r in valid if r['scope']=='voice_report'),
                              authoritative_supported=sum(r['support_decision']=='supported' for r in valid if r['scope']=='verified_authoritative_source'),
                              supported_expected_id_cited=sum(r['expected_support_cited'] is True for r in valid),
                              invalid_citation_reviews=sum(not r['citation_ids_valid'] for r in valid),
                              accounted_usd=sum(r.get('accounted_usd') or 0 for r in group),
                              limitation='12 fictional English cases grouped in4slot-types; assistant expectations,not human/externalgold; no real factual or medical accuracy claim.'))
    analyze.write_json(args.output/'summary.json',dict(models=summaries,budget_before=before,budget_after=ledger.report(),protocol=spec))
    analyze.write_csv(args.output/'summary.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in summaries])
    print(json.dumps(dict(status='new_scope_controls_finished',budget=ledger.report())),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status='stopped',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
