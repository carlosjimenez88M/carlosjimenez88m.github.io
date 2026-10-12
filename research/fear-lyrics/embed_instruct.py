"""Embeddings locales para la lectura argumental del miedo.

Calcula, sin APIs, los vectores de los 1.137 bloques únicos y de las anclas semánticas con:
- intfloat/multilingual-e5-large-instruct, sin instrucción y con una instrucción orientada al miedo;
- los dos encoders locales ya usados (MiniLM y E5 small), solo para las anclas: sus bloques ya están en caché.

Las letras se leen de la caché privada y los vectores se guardan allí; nada se publica.
Uso: python embed_instruct.py [--device mps]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
PRIVATE = STUDY.parents[1] / '.research-cache' / 'fear-lyrics'
OUT = PRIVATE / 'embeddings-v3'
INSTRUCT_MODEL = 'intfloat/multilingual-e5-large-instruct'
FEAR_TASK = ('Given a passage from a song lyric, represent what the speaker fears, '
             'where the threat comes from, and how the speaker responds to it')
LOCAL_MODELS = {
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': '',
    'intfloat/multilingual-e5-small': 'passage: ',
}


def anchor_texts(anchors: dict) -> list[tuple[str, str]]:
    rows = []
    for axis, spec in anchors['axes'].items():
        for pole in ('negative', 'positive'):
            rows += [(f'axis::{axis}::{pole}::{i:02d}', t) for i, t in enumerate(spec[pole])]
    for concept, texts in anchors['concepts'].items():
        rows += [(f'concept::{concept}::{i:02d}', t) for i, t in enumerate(texts)]
    return rows


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    matrix = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(matrix, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from sentence_transformers import SentenceTransformer

    documents = json.loads((PRIVATE / 'expanded' / 'embedding-input.json').read_text())['documents']
    anchors = json.loads((STUDY / 'semantic-anchors.json').read_text())
    anchor_rows = anchor_texts(anchors)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = PRIVATE / 'model-cache'
    receipt = {'documents': len(documents), 'anchors': len(anchor_rows), 'fear_task': FEAR_TASK,
               'anchors_version': anchors['version'], 'models': {}}

    model = SentenceTransformer(INSTRUCT_MODEL, device=args.device, cache_folder=str(cache))
    model.max_seq_length = 512
    texts = [d['text'] for d in documents]
    prefix = f'Instruct: {FEAR_TASK}\nQuery: '
    variants = {
        'plain': (texts, [t for _, t in anchor_rows]),
        'fear': ([prefix + t for t in texts], [prefix + t for _, t in anchor_rows]),
    }
    for name, (doc_inputs, anchor_inputs) in variants.items():
        np.save(OUT / f'e5-large-instruct-{name}-blocks.npy', encode(model, doc_inputs, args.batch_size))
        np.save(OUT / f'e5-large-instruct-{name}-anchors.npy', encode(model, anchor_inputs, args.batch_size))
    tokens = model.tokenizer(texts, truncation=False)['input_ids']
    receipt['models'][INSTRUCT_MODEL] = {
        'variants': list(variants), 'dimensions': int(model.get_sentence_embedding_dimension()),
        'max_seq_length': 512, 'blocks_over_512_tokens': int(sum(len(t) > 512 for t in tokens)),
        'commit': getattr(model[0].auto_model.config, '_commit_hash', None)}
    del model

    for model_id, local_prefix in LOCAL_MODELS.items():
        local = SentenceTransformer(model_id, device=args.device, cache_folder=str(cache))
        slug = model_id.replace('/', '--')
        np.save(OUT / f'{slug}-anchors.npy', encode(local, [local_prefix + t for _, t in anchor_rows], args.batch_size))
        receipt['models'][model_id] = {'variants': ['anchors_only'], 'anchor_prefix': local_prefix}

    index = {'document_ids': [d['input_id'] for d in documents], 'anchor_ids': [i for i, _ in anchor_rows],
             'document_text_sha256': hashlib.sha256('\n'.join(texts).encode()).hexdigest()}
    (OUT / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=1))
    (OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1))
    print(json.dumps(receipt, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
