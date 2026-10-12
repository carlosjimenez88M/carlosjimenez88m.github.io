"""Optional hosted-embedding adapter. Default mode is a local cost plan only.

No network request runs without --execute and an explicit positive cost ceiling.
The primary reported study uses local models. This adapter is NOT an executed
provider comparison. It never logs credentials or lyric text to public outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from dotenv import dotenv_values
import requests

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / ".research-cache/fear-lyrics"

def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", required=True, choices=["openai", "gemini"])
    p.add_argument("--model", required=True)
    p.add_argument("--input", type=Path, default=PRIVATE / "embedding-input.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-usd", type=float, required=True)
    p.add_argument("--usd-per-million-input-tokens", type=float, required=True,
                   help="Verified current price for the selected billing plan, including zero for a verified free tier.")
    p.add_argument("--dimensions", type=int, default=768)
    args = p.parse_args()
    if not args.input.resolve().is_relative_to(PRIVATE.resolve()) or not args.output.resolve().is_relative_to(PRIVATE.resolve()):
        raise ValueError("Inputs, outputs, and receipts must stay in the private cache.")
    if args.max_usd <= 0 or args.usd_per_million_input_tokens < 0:
        raise ValueError("Explicit positive ceiling and nonnegative verified unit price required.")
    if args.provider == "gemini" and args.model not in {"gemini-embedding-001", "gemini-embedding-2"}:
        raise ValueError("This adapter implements only the documented embedding-001/2 contracts.")
    data = json.loads(args.input.read_text())
    records = [(kind, entry) for kind in ("documents", "probes", "prototypes") for entry in data[kind]]
    def rendered(entry):
        # Embedding-2 has no task_type: make task intent explicit in its text.
        if args.provider == "gemini" and args.model == "gemini-embedding-2":
            task = "Represent this query for retrieving relevant lyric passages: " if entry["role"] == "query" else "Represent the meaning of this lyric passage for semantic analysis and retrieval: "
            return task + entry["text"]
        return entry["text"]
    upper_tokens = sum(len(rendered(e).encode("utf-8")) for _, e in records)
    projected = upper_tokens * args.usd_per_million_input_tokens / 1_000_000
    print(json.dumps({"mode":"execute" if args.execute else "plan_only", "provider":args.provider,
                      "model":args.model, "inputs":len(records), "conservative_byte_token_bound":upper_tokens,
                      "estimated_upper_input_usd":projected, "ceiling_usd":args.max_usd}))
    if not args.execute: return
    if projected > args.max_usd: raise ValueError("Conservative cost projection exceeds ceiling.")
    secrets = dotenv_values(ROOT / ".env")
    key = secrets.get("OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY")
    if not key: raise ValueError("Provider key is not configured.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / (args.output.stem + "-requests")
    cache.mkdir(exist_ok=True)
    session = requests.Session()
    arrays = {k: [] for k in ("documents", "probes", "prototypes")}
    receipts = []
    charged = 0.0
    for kind, entry in records:
        text = rendered(entry)
        request_identity = hashlib.sha256(json.dumps([args.provider,args.model,args.dimensions,entry["role"],text],ensure_ascii=False).encode()).hexdigest()
        dest = cache / (request_identity + ".json")
        if dest.exists():
            item = json.loads(dest.read_text()); reused = True
        else:
            upper = len(text.encode()) * args.usd_per_million_input_tokens / 1_000_000
            if charged + upper > args.max_usd: raise ValueError("Request would exceed remaining ceiling.")
            started = time.monotonic()
            if args.provider == "openai":
                url = "https://api.openai.com/v1/embeddings"
                body = {"model":args.model,"input":text,"dimensions":args.dimensions}
                headers = {"Authorization":"Bearer " + key}
            else:
                url = "https://generativelanguage.googleapis.com/v1beta/models/" + args.model + ":embedContent"
                body = {"content":{"parts":[{"text":text}]}, "outputDimensionality":args.dimensions}
                if args.model == "gemini-embedding-001":
                    body["taskType"] = "RETRIEVAL_QUERY" if entry["role"] == "query" else "RETRIEVAL_DOCUMENT"
                headers = {"x-goog-api-key":key}
            response = session.post(url, headers=headers, json=body, timeout=60)
            # No automatic retries: unknown billing outcomes must be reconciled.
            if response.status_code != 200:
                raise RuntimeError(f"Embedding provider HTTP status {response.status_code}; response text withheld.")
            payload = response.json()
            vector = payload["data"][0]["embedding"] if args.provider == "openai" else payload["embedding"]["values"]
            tokens = payload.get("usage", {}).get("prompt_tokens")
            accounted_tokens = tokens if tokens is not None else len(text.encode())
            item = {"vector":vector,"tokens":tokens,"accounting_basis":"provider_usage" if tokens is not None else "conservative_utf8_bytes",
                    "accounted_usd":accounted_tokens * args.usd_per_million_input_tokens / 1_000_000,
                    "latency_s":time.monotonic()-started,"request_id":response.headers.get("x-request-id"),
                    "model_returned":payload.get("model",args.model)}
            dest.write_text(json.dumps(item)); reused = False
            charged += item["accounted_usd"]
            time.sleep(.15)
        arrays[kind].append(item["vector"])
        receipts.append({"input_id":entry["input_id"],"hash":entry["text_sha256"],"role":entry["role"],
                         **{k:v for k,v in item.items() if k != "vector"},"cache_reused":reused})
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    for a in arrays.values(): a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(args.output, **arrays)
    sidecar = {"model_id":args.model, "provider":args.provider, "revision":None,
               "document_ids":[e["input_id"] for e in data["documents"]],
               "document_hashes":[e["text_sha256"] for e in data["documents"]],
               "probe_hashes":[e["text_sha256"] for e in data["probes"]],
               "prototype_hashes":[e["text_sha256"] for e in data["prototypes"]],
               "usage":{"newly_accounted_usd":charged,"cumulative_accounted_usd":sum(r["accounted_usd"] for r in receipts),"total_input_tokens":sum(r["tokens"] or 0 for r in receipts),"distinct_embedding_requests":len(receipts),"input_token_unit_price":args.usd_per_million_input_tokens},
               "prefixes":({"query":"", "passage":""} if args.provider=="openai" else {"query":"retrieval intent", "passage":"document/semantic intent"}),
               "token_counts":{kind:[r["tokens"] for r in receipts if r["input_id"] in {x["input_id"] for x in data[kind]}] for kind in ("documents","probes","prototypes")},
               "max_seq_length":8192 if args.provider=="openai" else None,
               "receipts":receipts,"note":"Original probes use passage encoding to diagnose symmetric similarities; not retrieval-query performance."}
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(json.dumps({"status":"executed","model":args.model,"requests":len(receipts),"newly_accounted_usd":charged}))

if __name__ == "__main__": run()
