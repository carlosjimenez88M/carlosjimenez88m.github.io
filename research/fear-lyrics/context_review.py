#!/usr/bin/env python3
"""Matched OpenAI readers on frozen contexts, with an actual bounded LangGraph.

Plan-only by default. Neither reader retrieves, rewrites nor chooses tools. Both
receive identical ORIGINAL test queries and audited private blocks. Anthropic
and Gemini are not permitted in this lyric-context phase.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values
from langgraph.graph import END, START, StateGraph

import analyze
import llm_comparison as comparison
import llm_review as api
import rag

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / ".research-cache/fear-lyrics"
PUBLIC = Path(__file__).resolve().parent / "results/llm-comparison-context"
MODELS = ["openai/gpt-5.4-mini", "openai/gpt-5.4-nano"]


def prepare(retrieval_dir, bundle_path, method, arm, authorized_bundle=None):
    questions = json.loads((retrieval_dir/"questions.json").read_text())
    runs = json.loads((retrieval_dir/"runs.json").read_text())
    protocol = json.loads((retrieval_dir/"protocol.json").read_text())
    bundle = json.loads(bundle_path.read_text())
    docs = {d["input_id"]: d for d in bundle["documents"]}
    approved = None
    if authorized_bundle is not None:
        approved = {d["text_sha256"] for d in json.loads(authorized_bundle.read_text())["documents"]}
    for doc in docs.values():
        if hashlib.sha256(doc["text"].encode()).hexdigest() != doc["text_sha256"]:
            raise ValueError("Private text does not match its declared audited hash.")
    input_hash = rag.identity([(d["input_id"],d["text_sha256"]) for d in bundle["documents"]])
    if protocol["private_inputs_hash"] != input_hash:
        raise ValueError("Frozen retrieval outputs do not match current private corpus.")
    selected = {r["query_id"]: r for r in runs if r["method"]==method and r["arm"]==arm}
    records=[]
    for q in questions:
        if q["split"] != "test":continue
        retrieved = selected[q["query_id"]]
        ids = retrieved["context_ids"]
        if any(doc_id not in docs for doc_id in ids):raise ValueError("Unknown frozen context ID.")
        blocks=[dict(document_id=doc_id,text=docs[doc_id]["text"]) for doc_id in ids]
        if approved is not None and any(docs[doc_id]["text_sha256"] not in approved for doc_id in ids):
            raise ValueError("Context contains text outside the original explicitly authorized block set.")
        words=sum(len(b["text"].split()) for b in blocks)
        if words>320 or len(ids)>5:raise ValueError("Frozen contexts exceed matched evidence budget.")
        records.append(dict(query=q,payload=dict(original_question=q["query"],evidence_blocks=blocks),
                            context_ids=ids,context_words=words,context_identity=api.digest([(i,docs[i]["text_sha256"]) for i in ids]),
                            rule_abstained=retrieved["abstained"],
                            assistant_context_support_recall=retrieved.get("context_support_recall"),
                            assistant_all_support_in_context=retrieved.get("all_annotated_support_in_context")))
    if len(records)!=24:raise ValueError("Expected the frozen 24 test queries.")
    return records,input_hash


class State(TypedDict,total=False):
    record:dict
    model_id:str
    private_response:dict
    public_response:dict
    trace:list[str]
    abstained:bool


def reader_graph(call):
    def attach(state):
        return dict(trace=["attach_frozen_context:original_query_no_retrieval"])
    def read(state):
        return dict(private_response=call(state["model_id"],state["record"]["payload"]),
                    trace=[*state["trace"],"support_reader:bounded_provider_JSON"])
    def validate(state):
        item=state["private_response"]
        exported={k:v for k,v in item.items() if k!="parsed"}
        if item["status"]=="executed":
            try:
                exported.update(api.public_review(item["parsed"],state["record"]["context_ids"]))
                if set(item["parsed"]) != set(api.REVIEW_SCHEMA["required"]) or not isinstance(item["parsed"]["uncertainty"],str):
                    raise ValueError("Invalid requested-review JSON keys or uncertainty type.")
                exported["supported_has_citation"] = exported["support_decision"]!="supported" or bool(exported["cited_ids"])
                exported["supported_has_nonempty_claim"] = exported["support_decision"]!="supported" or any(x.strip() for x in item["parsed"]["claims"])
                exported["supported_has_no_missing_slots"] = exported["support_decision"]!="supported" or exported["missing_slot_count"] == 0
            except ValueError:exported["status"]="executed_invalid_schema"
        abstain=(exported["status"]!="executed" or exported.get("abstained",True) or
                 not exported.get("citation_ids_valid",False) or not exported.get("supported_has_citation",False) or
                 not exported.get("supported_has_nonempty_claim",False) or not exported.get("supported_has_no_missing_slots",False))
        return dict(public_response=exported,abstained=abstain,
                    trace=[*state["trace"],"citation_validation:membership_only_not_entailment"])
    def branch(state):return "abstain" if state["abstained"] else "answer"
    def answer(state):return dict(trace=[*state["trace"],"answer:private_short_claims_public_counts_only"])
    def abstain(state):return dict(trace=[*state["trace"],"abstain:missing_support_or_invalid_citation"])
    g=StateGraph(State)
    for name,node in (("attach_frozen_context",attach),("support_reader",read),("citation_validation",validate),("answer",answer),("abstain",abstain)):g.add_node(name,node)
    g.add_edge(START,"attach_frozen_context");g.add_edge("attach_frozen_context","support_reader");g.add_edge("support_reader","citation_validation")
    g.add_conditional_edges("citation_validation",branch,{"answer":"answer","abstain":"abstain"})
    g.add_edge("answer",END);g.add_edge("abstain",END)
    return g.compile()


def self_test():
    record=dict(payload=dict(original_question="Who is afraid?",evidence_blocks=[]),context_ids=["d1"])
    def fake(model,payload):
        return dict(status="executed",parsed=dict(support_decision="supported",missing_requested_slots=[],claims=["Synthetic private statement"],cited_ids=["foreign"],uncertainty=""))
    graph=reader_graph(fake);r=graph.invoke(dict(record=record,model_id=MODELS[0]),{"recursion_limit":10})
    assert r["abstained"] and not r["public_response"]["citation_ids_valid"]
    assert "Synthetic private statement" not in json.dumps(r["public_response"])
    assert len(r["trace"])==4
    for overrides in ({"claims":[]},{"claims":[" "]},{"missing_requested_slots":["requested name"]}):
        def incomplete(model,payload):
            parsed=dict(support_decision="supported",missing_requested_slots=[],claims=["Original synthetic claim"],cited_ids=["d1"],uncertainty="")
            parsed.update(overrides)
            return dict(status="executed",parsed=parsed)
        checked=reader_graph(incomplete).invoke(dict(record=record,model_id=MODELS[0]),{"recursion_limit":10})
        assert checked["abstained"]
    print("Context-reader self-test passed: bounded LangGraph, citation membership, missing-slot and empty-claim guards; private claims remain private.")


def paired_summary(rows):
    results=[]
    for a,b in combinations(sorted({r['model'] for r in rows}),2):
        left={r['query_id']:r for r in rows if r['model']==a and r['status']=='executed'}
        right={r['query_id']:r for r in rows if r['model']==b and r['status']=='executed'}
        shared=sorted(set(left)&set(right))
        results.append(dict(model_a=a,model_b=b,paired_valid_queries=len(shared),
                            support_decision_agreement=sum(left[q]['support_decision']==right[q]['support_decision'] for q in shared),
                            final_abstention_agreement=sum(left[q]['final_abstained']==right[q]['final_abstained'] for q in shared),
                            decision_disagreements=[dict(query_id=q,model_a_decision=left[q]['support_decision'],model_b_decision=right[q]['support_decision'],
                                                        model_a_abstained=left[q]['final_abstained'],model_b_abstained=right[q]['final_abstained'])
                                                    for q in shared if left[q]['support_decision']!=right[q]['support_decision']],
                            interpretation='Agreement measures matching decisions, not correctness; equal aggregate counts can conceal different accepted questions.'))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true");parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--models",nargs="+",default=MODELS)
    parser.add_argument("--retrieval-dir",type=Path,default=Path(__file__).resolve().parent/"results/rag-ablation")
    parser.add_argument("--bundle",type=Path,default=PRIVATE/"embedding-input.json")
    parser.add_argument("--authorized-bundle",type=Path,default=PRIVATE/"pilot-40/embedding-input.json")
    parser.add_argument("--output",type=Path,default=PUBLIC)
    parser.add_argument("--method",choices=["bm25","dense","hybrid"],default="hybrid")
    parser.add_argument("--arm",default="rewrite_1_neighbors_0")
    parser.add_argument("--max-usd",type=float,default=.80)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not set(args.models)<=set(MODELS):raise ValueError("Lyric contexts authorized for registered OpenAI models only.")
    if not args.bundle.resolve().is_relative_to(PRIVATE.resolve()):raise ValueError("Corpus must remain private.")
    if args.output.resolve().name=="llm-review":raise ValueError("Preserve the pilot LLM outputs.")
    records,corpus_hash=prepare(args.retrieval_dir,args.bundle,args.method,args.arm,args.authorized_bundle)
    plan=dict(mode="execute" if args.run else "plan_only",models=args.models,test_queries=len(records),test_families=len({r['query']['family'] for r in records}),
              private_inputs_hash=corpus_hash,context_pool_identity=api.digest([(r['query']['query_id'],r['context_identity']) for r in records]),
              retrieval_source=str(args.retrieval_dir),method=args.method,arm=args.arm,prompt_version=api.VERSION,prompt_sha256=api.digest(api.REVIEW_SYSTEM),
              graph="attach_frozen_context -> support_reader -> citation_validation -> answer/abstain; no model tool selection or retrieval/rewrite changes",
              maximum_evidence=dict(words=320,blocks=5),max_output_tokens=api.MAX_OUTPUT,temperature=0,
              authorship="Questions/qrels authored by assistant; matched context review, no independent human gold, no new held-out test.",
              annotation_limit="All-support designation means original provisional support IDs, not exhaustive relevance. Expanded corpus can contain additional valid evidence.",
              final_answer_guard="Supported requires at least one nonempty claim, a supplied citation ID and no missing requested slots. Citation membership is not entailment validation.",
              authorization_scope="Each transmitted block SHA256 must match the original explicitly authorized 386-block set; no expanded-corpus API call.",
              privacy="OpenAI receives explicitly authorized private blocks. Claims/uncertainty text remain private. Public whitelist only decisions, IDs, counts, usage and traces.")
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if not args.run:return
    key=dotenv_values(ROOT/".env").get("OPENAI_API_KEY")
    ledger=api.Ledger(PRIVATE/"cost-ledger-llm.json",args.max_usd);before=ledger.report();cache=PRIVATE/"llm-review"
    def call(model_id,payload):
        spec=comparison.REGISTRY[model_id]
        if not key:return dict(status="key_not_configured")
        return api.execute("openai",api.REVIEW_SYSTEM,payload,key,ledger,cache,model_id=spec["api_id"],rates=spec["rates"])
    graph=reader_graph(call);rows=[]
    analyze.write_json(args.output/"protocol.json",plan)
    for model_id in args.models:
        blocked=None
        for index,record in enumerate(records):
            q=record["query"]
            row=dict(provider="openai",model=model_id,query_id=q["query_id"],family=q["family"],language=q["language"],split=q["split"],assistant_answerable=q["answerable"],
                     **{k:record[k] for k in ("context_ids","context_words","context_identity","rule_abstained","assistant_context_support_recall","assistant_all_support_in_context")})
            if blocked:row["status"]="not_attempted_after_"+blocked
            else:
                state=graph.invoke(dict(record=record,model_id=model_id),{"recursion_limit":10})
                row.update(state["public_response"]);row.update(final_abstained=state["abstained"],trace=state["trace"])
                if row["status"] in {"http_401","http_402","http_403","http_404","http_429","network_unknown"}:blocked=row["status"]
            rows.append(row);analyze.write_json(args.output/"runs.json",rows)
            print(json.dumps(dict(model=model_id,progress=index+1,total=len(records),status=row["status"])),flush=True)
    summaries=[]
    for model_id in args.models:
        group=[r for r in rows if r["model"]==model_id];valid=[r for r in group if r["status"]=="executed"]
        positives=[r for r in valid if r["assistant_answerable"]];negatives=[r for r in valid if not r["assistant_answerable"]]
        complete=[r for r in positives if r["assistant_all_support_in_context"]];incomplete=[r for r in positives if not r["assistant_all_support_in_context"]]
        summaries.append(dict(model_id=model_id,requested=len(group),valid_reviews=len(valid),status_counts=dict(Counter(r['status'] for r in group)),
                              positive_queries=len(positives),negative_queries=len(negatives),positive_abstentions=sum(r['final_abstained'] for r in positives),
                              negative_safe_abstentions=sum(r['final_abstained'] for r in negatives),positive_complete_contexts=len(complete),
                              complete_context_abstentions=sum(r['final_abstained'] for r in complete),positive_incomplete_contexts=len(incomplete),
                              incomplete_context_abstentions=sum(r['final_abstained'] for r in incomplete),invalid_citation_reviews=sum(not r['citation_ids_valid'] for r in valid),
                              accounted_usd=sum(r.get('accounted_usd') or 0 for r in group),newly_accounted_usd=sum(r.get('accounted_usd') or 0 for r in group if not r.get('cache_reused')),
                              limitation="Same frozen contexts; assistant qrels are provisional and incomplete, no human entailment adjudication or provider-level generalization."))
    analyze.write_json(args.output/"summary.json",dict(models=summaries,pairwise=paired_summary(rows),
                        requested_review_rows=len(rows),new_requests=sum(not r.get('cache_reused',False) for r in rows),
                        cache_reused_rows=sum(r.get('cache_reused',False) for r in rows),budget_before=before,budget_after=ledger.report(),protocol=plan))
    print(json.dumps(dict(status="context_comparison_finished",budget=ledger.report())),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        print(json.dumps(dict(status="stopped",error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
