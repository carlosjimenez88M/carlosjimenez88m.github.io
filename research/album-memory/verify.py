"""Audit corpus provenance, factorial coverage, receipts, and local tracking."""
import hashlib
import itertools
import json
import sqlite3
from collect import ALBUMS, PRIVATE, OUT


def read(path):
    return json.loads(path.read_text())


def main():
    manifest=read(OUT/'corpus_manifest.json')
    assert len(manifest)==56
    for item in manifest:
        raw=read(PRIVATE/f"{item['id']}.json")
        digest=hashlib.sha256('\n'.join(raw['lines']).encode()).hexdigest()
        assert digest==item['sha256']
        assert item['source_url'].startswith('https://genius.com/')
        assert item['line_count']==len(raw['lines'])
    rows=read(OUT/'runs.json')
    expected={(a['id'],p,q,r) for a,p,q,r in itertools.product(
        ALBUMS,['full','recent3','summary','selective6'],
        ['return','change','counterevidence'],range(3))}
    observed={(r['album'],r['policy'],r['probe_kind'],r['repetition']) for r in rows}
    assert len(rows)==144 and observed==expected
    with sqlite3.connect(PRIVATE/'mlflow.db') as db:
        for row in rows:
            run=db.execute('SELECT status FROM runs WHERE run_uuid=?',(row['mlflow_run_id'],)).fetchone()
            assert run==('FINISHED',)
            assert row['metrics']['input_tokens']==row['generation']['usage']['prompt_tokens']>0
            assert row['metrics']['output_tokens']==row['generation']['usage']['completion_tokens']>0
            receipt=read(PRIVATE/'calls'/f"{row['generation']['request_sha256']}.json")
            assert receipt['output']==row['answer']
        graph_count=db.execute("SELECT count(*) FROM spans WHERE name='LangGraph'").fetchone()[0]
        assert graph_count>=144
        trace_count=db.execute('SELECT count(*) FROM trace_info').fetchone()[0]
    for album in ALBUMS:
        cards=read(OUT/f"cards-{album['id']}.json")
        assert len(cards)==len(album['tracks'])
        for c in cards:
            source=read(PRIVATE/f"{album['id']}-{c['position']:02}.json")
            assert all(1<=line<=len(source['lines']) for claim in c['claims'] for line in claim['evidence_lines'])
        probes=read(OUT/f"probes-{album['id']}.json")
        for q in probes:
            i,j=sorted(int(s[1:]) for s in q['target_ids'])
            assert 1<=i<=len(cards)//2<j<=len(cards) and j-i>=4
    report=dict(corpus_tracks=56,completed_cases=144,graph_spans=graph_count,
                traces=trace_count,provider_receipts_verified=True,
                source_hashes_verified=True,
                scope='Structural and provenance audit, not semantic validation of interpretations or judge scores.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
