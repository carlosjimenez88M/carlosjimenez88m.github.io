"""Private, cached OpenAI embedding batches with bounded cost and exact order.

Default is a local plan. A pending receipt is written before each network call;
unknown outcomes stop execution and are never automatically retried. Array
batches are ordinary synchronous embedding requests, not the Batch API.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / '.research-cache/fear-lyrics'
PRICES = {'text-embedding-3-large': .13, 'text-embedding-3-small': .02}

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()

def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=list(PRICES))
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dimensions', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-usd', type=float, default=.19)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    for path in [args.input, args.queries, args.output]:
        if not path.resolve().is_relative_to(PRIVATE.resolve()):
            raise ValueError('All inputs and outputs must stay in the private cache.')
    if args.dimensions != 1024 or not 1 <= args.batch_size <= 64 or not 0 < args.max_usd <= .19:
        raise ValueError('This experiment fixes 1024 dimensions and a <= USD .19 phase ceiling.')
    data = json.loads(args.input.read_text())
    data['queries'] = json.loads(args.queries.read_text())['queries']
    groups = ['documents', 'probes', 'prototypes', 'queries']
    rows = [(kind, row) for kind in groups for row in data[kind]]
    for kind, row in rows:
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Input hash mismatch.')
        if not row['text'].strip() or len(row['text'].encode()) > 8192:
            raise ValueError('Input violates the conservative per-input byte/token bound.')
    price = PRICES[args.model]
    unique = {}
    for kind, row in rows:
        signature = digest(['openai', args.model, args.dimensions, row['role'], row['text']])
        unique.setdefault(signature, row)
    cache = PRIVATE / 'embedding-batches' / args.model
    cache.mkdir(parents=True, exist_ok=True)
    legacy = PRIVATE / 'openai-large-requests'
    vectors, receipts, missing = {}, {}, []
    for signature, row in unique.items():
        dest = cache / (signature + '.json')
        old = legacy / (signature + '.json')
        if dest.exists():
            receipt = json.loads(dest.read_text())
        elif args.model == 'text-embedding-3-large' and old.exists():
            receipt = {**json.loads(old.read_text()), 'billing_origin': 'pilot_single_input'}
        else:
            missing.append(signature); continue
        vector = np.asarray(receipt['vector'], dtype=np.float32)
        if vector.shape != (args.dimensions,) or not np.isfinite(vector).all():
            raise ValueError('Cached vector invalid.')
        vectors[signature] = vector
        receipts[signature] = {k: v for k, v in receipt.items() if k != 'vector'}
    upper = sum(len(unique[s]['text'].encode()) for s in missing) * price / 1e6
    print(json.dumps({'mode': 'execute' if args.execute else 'plan', 'model': args.model,
                      'inputs': len(rows), 'unique_inputs': len(unique), 'reused': len(vectors),
                      'new_inputs': len(missing), 'conservative_cost_usd': upper}), flush=True)
    if not args.execute:
        return
    if upper > args.max_usd:
        raise ValueError('Conservative cost exceeds the phase cap.')
    ledger_path = PRIVATE / 'cost-ledger-embeddings-expanded.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'phase_cap_usd': .19, 'requests': []}
    if any(r['status'] == 'pending' for r in ledger['requests']):
        raise RuntimeError('Unknown billing outcome requires reconciliation; no automatic retry.')
    key = dotenv_values(ROOT / '.env').get('OPENAI_API_KEY')
    if not key:
        raise ValueError('OpenAI key is not configured.')
    session = requests.Session()
    new_cost = 0.
    for start in range(0, len(missing), args.batch_size):
        identities = missing[start:start + args.batch_size]
        texts = [unique[s]['text'] for s in identities]
        bound = sum(len(t.encode()) for t in texts) * price / 1e6
        committed = sum(r.get('actual_usd', r['reserved_usd']) for r in ledger['requests'])
        if committed + bound > min(args.max_usd, ledger['phase_cap_usd']):
            raise ValueError('Batch would exceed the accumulated embedding phase cap.')
        record = {'signature': digest([args.model, identities]), 'model': args.model,
                  'input_hashes': identities, 'reserved_usd': bound, 'status': 'pending'}
        ledger['requests'].append(record)
        ledger_path.write_text(json.dumps(ledger, indent=2))
        started = time.monotonic()
        response = session.post('https://api.openai.com/v1/embeddings',
                                headers={'Authorization': 'Bearer ' + key},
                                json={'model': args.model, 'dimensions': args.dimensions, 'input': texts}, timeout=90)
        if response.status_code != 200:
            record.update(http_status=response.status_code)
            ledger_path.write_text(json.dumps(ledger, indent=2))
            raise RuntimeError(f'Embedding HTTP {response.status_code}; payload withheld.')
        payload = response.json()
        ordered = sorted(payload['data'], key=lambda r: r['index'])
        if [r['index'] for r in ordered] != list(range(len(identities))):
            raise ValueError('Batch indices are not complete.')
        matrix = np.asarray([r['embedding'] for r in ordered], dtype=np.float32)
        if matrix.shape != (len(identities), args.dimensions) or not np.isfinite(matrix).all():
            raise ValueError('Provider vector shape or values invalid.')
        tokens = int(payload['usage']['prompt_tokens'])
        cost = tokens * price / 1e6
        if cost > bound + 1e-12:
            raise ValueError('Observed billing exceeds reserved byte bound.')
        record.update(status='settled', actual_usd=cost, input_tokens=tokens,
                      request_id=response.headers.get('x-request-id'), latency_s=time.monotonic()-started)
        for signature, vector in zip(identities, matrix):
            item = {'vector': vector.tolist(), 'batch_receipt': record['signature'],
                    'model_returned': payload.get('model', args.model),
                    'accounting_basis': 'batch_provider_usage', 'per_input_tokens': None}
            (cache / (signature + '.json')).write_text(json.dumps(item))
            vectors[signature] = vector
            receipts[signature] = {k: v for k, v in item.items() if k != 'vector'}
        ledger_path.write_text(json.dumps(ledger, indent=2))
        new_cost += cost
        print(json.dumps({'model': args.model, 'completed_new': start+len(identities),
                          'new_total': len(missing), 'new_usd': new_cost}), flush=True)
    arrays = {}
    for kind in groups:
        matrix = np.asarray([vectors[digest(['openai', args.model, args.dimensions, r['role'], r['text']])]
                             for r in data[kind]], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError('Zero vector.')
        arrays[kind] = matrix / norms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    ledger_for_model = [r for r in ledger['requests'] if r['model'] == args.model and r['status'] == 'settled']
    metadata = {'model_id': args.model, 'provider': 'openai', 'revision': None, 'dimensions': args.dimensions,
                'document_ids': [r['input_id'] for r in data['documents']],
                'document_hashes': [r['text_sha256'] for r in data['documents']],
                'probe_hashes': [r['text_sha256'] for r in data['probes']],
                'prototype_hashes': [r['text_sha256'] for r in data['prototypes']],
                'query_hashes': [r['text_sha256'] for r in data['queries']],
                'max_seq_length': 8192, 'prefixes': {'query': '', 'passage': ''},
                'input_token_upper_bounds': {k: [len(r['text'].encode()) for r in data[k]] for k in groups},
                'usage': {'newly_accounted_usd': new_cost,
                          'expanded_phase_actual_usd': sum(r['actual_usd'] for r in ledger_for_model),
                          'expanded_phase_input_tokens': sum(r['input_tokens'] for r in ledger_for_model),
                          'batch_requests': len(ledger_for_model), 'input_token_unit_price': price,
                          'pilot_receipts_reused': sum(r.get('billing_origin') == 'pilot_single_input' for r in receipts.values())},
                'tokenization_note': 'Per-input byte bounds guarantee <=8192 tokens; actual usage is only known at batch level.',
                'receipts': [{'input_signature': s, **r} for s, r in receipts.items()]}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'status': 'executed', 'model': args.model, 'rows': len(rows), 'new_usd': new_cost}))

if __name__ == '__main__':
    run()
