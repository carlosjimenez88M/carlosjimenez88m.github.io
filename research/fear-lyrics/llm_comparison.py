#!/usr/bin/env python3
"""Plan-only by default: matched synthetic relationship probes across four models.

Anthropic is restricted to the fixed ORIGINAL analyze.PROBES. No lyrics or
corpus-derived paraphrases are sent there. Provider/model outputs are judgments,
not embedding vectors, literary gold or population-level performance estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import tempfile

from dotenv import dotenv_values

import analyze
import llm_review as api

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / ".research-cache/fear-lyrics"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/llm-comparison-synthetic"
REGISTRY = {
    "openai/gpt-5.4-mini": dict(provider="openai", api_id="gpt-5.4-mini", rates=(.75, 4.50), source="https://developers.openai.com/api/docs/models/gpt-5.4-mini"),
    "openai/gpt-5.4-nano": dict(provider="openai", api_id="gpt-5.4-nano", rates=(.20, 1.25), source="https://developers.openai.com/api/docs/models/gpt-5.4-nano", lifecycle="Official docs mark deprecated; metadata GET returned 200 on 2026-10-11."),
    "anthropic/claude-sonnet-4-6": dict(provider="anthropic", api_id="claude-sonnet-4-6", rates=(3., 15.), source="https://platform.claude.com/docs/en/models/sonnet-4-6/overview"),
    "anthropic/claude-haiku-4-5-20251001": dict(provider="anthropic", api_id="claude-haiku-4-5-20251001", rates=(1., 5.), source="https://platform.claude.com/docs/en/models/haiku-4-5/overview", lifecycle="Active legacy; metadata GET returned 200 on 2026-10-11."),
}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"relationship_ab": {"type": "string", "enum": api.RELATIONS},
                         "relationship_ac": {"type": "string", "enum": api.RELATIONS},
                         "uncertainty": {"type": "string"}},
          "required": ["relationship_ab", "relationship_ac", "uncertainty"]}
# Narrow logical controls, not a blanket accuracy gold for all 40 relationships.
# Other pairs can admit multiple readings (entailment vs equivalence, events at
# different times). This audit is fixed before the new runs, but informed by the
# previous Sonnet pilot; it is not independent held-out validation.
CORE = {
    ("negation_en", "relationship_ab"): {"contradiction"},
    ("negation_es", "relationship_ab"): {"contradiction"},
    ("role_en", "relationship_ab"): {"role_reversal"},
    ("role_es", "relationship_ab"): {"role_reversal"},
    ("translation_en_es", "relationship_ab"): {"equivalent"},
    ("translation_es_en", "relationship_ab"): {"equivalent"},
    ("translation_en_es", "relationship_ac"): {"unrelated", "related_not_equivalent"},
    ("translation_es_en", "relationship_ac"): {"unrelated", "related_not_equivalent"},
    ("order_en", "relationship_ac"): {"order_reversal"},
}


def summarize(rows):
    summaries = []
    for model_id in dict.fromkeys(r["model_id"] for r in rows):
        group = [r for r in rows if r["model_id"] == model_id]
        valid = [r for r in group if r["status"] == "executed"]
        core_results = [(r, field, expected) for r in valid for (probe_id, field), expected in CORE.items() if r["probe_id"] == probe_id]
        summaries.append(dict(model_id=model_id, requested=len(group), valid=len(valid),
                              status_counts=dict(Counter(r["status"] for r in group)),
                              relationship_ab_counts=dict(Counter(r["relationship_ab"] for r in valid)),
                              relationship_ac_counts=dict(Counter(r["relationship_ac"] for r in valid)),
                              core_assistant_constraints=len(core_results),
                              core_constraint_matches=sum(r[field] in expected for r, field, expected in core_results),
                              coexistence_false_contradictions=sum(r["probe_id"].startswith("translation_") and r["relationship_ac"] == "contradiction" for r in valid),
                              accounted_usd_including_reused_receipts=sum(r.get("accounted_usd") or 0 for r in group),
                              newly_accounted_usd=sum(r.get("accounted_usd") or 0 for r in group if not r.get("cache_reused")),
                              reused_requests=sum(bool(r.get("cache_reused")) for r in group),
                              limitation="Nine assistant-defined logical constraints, informed by pilot; not accuracy gold for all pairs. Twenty correlated synthetic triplets, no independent human validation."))
    pairwise = []
    for left, right in combinations([r["model_id"] for r in summaries], 2):
        l = {r["probe_id"]: r for r in rows if r["model_id"] == left and r["status"] == "executed"}
        rr = {r["probe_id"]: r for r in rows if r["model_id"] == right and r["status"] == "executed"}
        shared = sorted(l.keys() & rr.keys())
        total = len(shared)*2
        matched = sum(l[p][f] == rr[p][f] for p in shared for f in ("relationship_ab", "relationship_ac"))
        pairwise.append(dict(left=left, right=right, common_triplets=len(shared), pair_judgments=total,
                             agreement_count=matched, agreement_fraction=matched/total if total else None,
                             limitation="Agreement is not correctness; paired language/category controls are not iid samples."))
    return summaries, pairwise


def self_test():
    p = analyze.PROBES[0]
    payload = {"A": p["query"], "B": p["related"], "C": p["contrast"]}
    _, body = api.api_body("openai", api.SYNTHETIC_SYSTEM, payload, schema=SCHEMA, model_id="gpt-5.4-nano")
    assert body["model"] == "gpt-5.4-nano" and body["response_format"]["json_schema"]["schema"] == SCHEMA
    _, body = api.api_body("anthropic", api.SYNTHETIC_SYSTEM, payload, synthetic=True, model_id="claude-haiku-4-5-20251001")
    assert body["model"] == "claude-haiku-4-5-20251001"
    with tempfile.TemporaryDirectory() as directory:
        ledger = api.Ledger(Path(directory)/"ledger.json", .80)
        ledger.reserve("a", "openai", .01); ledger.settle("a", .001, "ok")
        assert abs(ledger.report()["settled_usd"]-.001)<1e-12
    assert len(CORE)==9
    print("Matched-model self-test passed: model/schema selection, Anthropic original-only scope, shared ledger and fixed constraints.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(REGISTRY))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-usd", type=float, default=.80)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not set(args.models)<=set(REGISTRY):
        raise ValueError("Only explicitly registered OpenAI/Anthropic models are authorized in this phase.")
    forbidden = Path(__file__).resolve().parent/"results/llm-review"
    if args.output.resolve()==forbidden.resolve():
        raise ValueError("Do not overwrite the frozen pilot LLM outputs.")
    plan = dict(mode="execute" if args.run else "plan_only", task="matched_original_synthetic_relationship_judgments",
                models={k:REGISTRY[k] for k in args.models}, triplets=len(analyze.PROBES), planned_rows=len(analyze.PROBES)*len(args.models),
                prompt_sha256=api.digest(api.SYNTHETIC_SYSTEM), prompt_version=api.VERSION,
                input_identity=api.digest(analyze.PROBES), max_output_tokens=api.MAX_OUTPUT, temperature=0,
                provider_controls="Identical textual system/user content; OpenAI adds strict output schema, Anthropic JSON follows prompt. No web/tools or extended thinking.",
                core_constraints=[dict(probe_id=p, field=f, allowed_labels=sorted(v)) for (p,f),v in CORE.items()],
                constraint_provenance="Assistant-defined diagnostic subset informed by earlier Sonnet pilot, frozen before new model execution; not independent human gold.",
                budget="Shared existing .80 LLM ledger; all previous pilot costs retained; other phases reserved separately under user total USD1.",
                privacy="Original synthetic sentences only, no lyrics/corpus-derived text to any provider. Raw model text private; labels/IDs/usage public.",
                scope="Compare only the same task/input across models. Do not average this with retrieval, lyric sufficiency or embedding tests.")
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if not args.run:return
    s=dotenv_values(ROOT/".env")
    keys={"openai":s.get("OPENAI_API_KEY"), "anthropic":s.get("ANTHROPIC_API_KEY") or s.get("ANTROPIC_API_KEY")}
    ledger=api.Ledger(PRIVATE/"cost-ledger-llm.json", args.max_usd)
    before=ledger.report()
    cache=PRIVATE/"llm-review";cache.mkdir(exist_ok=True)
    analyze.write_json(args.output/"protocol.json",plan)
    rows=[]
    for model_id in args.models:
        spec=REGISTRY[model_id];provider=spec["provider"];blocked=None
        for index,probe in enumerate(analyze.PROBES):
            row=dict(model_id=model_id,provider=provider,api_model_requested=spec["api_id"],probe_id=probe["id"],category=probe["category"],
                     input_identity=api.digest({k:probe[k] for k in ("query","related","contrast")}), prompt_sha256=plan["prompt_sha256"])
            if blocked or not keys[provider]:
                row["status"]="not_attempted_after_"+blocked if blocked else "key_not_configured"
            else:
                item=api.execute(provider,api.SYNTHETIC_SYSTEM,{"A":probe["query"],"B":probe["related"],"C":probe["contrast"]},
                                 keys[provider],ledger,cache,synthetic=True,schema=SCHEMA,model_id=spec["api_id"],rates=spec["rates"])
                row.update({k:v for k,v in item.items() if k!="parsed"})
                if item["status"]=="executed":
                    parsed=item["parsed"]
                    if parsed.get("relationship_ab") in api.RELATIONS and parsed.get("relationship_ac") in api.RELATIONS:
                        row.update(relationship_ab=parsed["relationship_ab"],relationship_ac=parsed["relationship_ac"],has_uncertainty=bool(parsed.get("uncertainty")))
                    else:row["status"]="executed_invalid_schema"
                if item["status"] in {"http_401","http_402","http_403","http_404","http_429","network_unknown"}:blocked=item["status"]
            rows.append(row);analyze.write_json(args.output/"runs.json",rows)
            print(json.dumps(dict(model_id=model_id,progress=index+1,total=len(analyze.PROBES),status=row["status"],cache_reused=bool(row.get("cache_reused")))),flush=True)
    models,pairwise=summarize(rows)
    analyze.write_json(args.output/"summary.json",dict(models=models,pairwise=pairwise,budget_before=before,budget_after=ledger.report(),protocol=plan))
    analyze.write_csv(args.output/"summary.csv",[{k:v for k,v in m.items() if not isinstance(v,dict)} for m in models])
    print(json.dumps(dict(status="comparison_finished",budget=ledger.report())),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        print(json.dumps(dict(status="stopped",error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
