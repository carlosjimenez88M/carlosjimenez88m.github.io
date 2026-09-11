"""Recompute descriptive results from completed runs; never export lyric text."""
from collections import defaultdict
import json
import re
import numpy as np
import pandas as pd
from collect import ALBUMS, PRIVATE, OUT


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    rows = json.loads((OUT / 'runs.json').read_text())
    assert len(rows) == 144 and len({r['case_id'] for r in rows}) == 144
    counts={a['id']:len(a['tracks']) for a in ALBUMS}
    diagnostic_rows=[]
    for r in rows:
        ids=r['answer']['evidence_ids']
        positions=[int(i[1:]) for i in ids if re.fullmatch(r'T\d{2}',i)]
        n=counts[r['album']]
        # Added after inspecting an attribution failure: diagnostic only,
        # not a predeclared primary metric or a guarantee of semantic support.
        cross_half=any(1<=i<=n//2 for i in positions) and any(n//2<i<=n for i in positions)
        diagnostic_rows.append(dict(case_id=r['case_id'],
                                    citations_span_halves=cross_half,
                                    answer_within_90_words=len(r['answer']['answer'].split())<=90))
    save('posthoc_diagnostics.json',diagnostic_rows)
    frame = pd.DataFrame([dict(album=r['album'], policy=r['policy'],
                              probe=r['probe_kind'], repetition=r['repetition'],
                              **r['metrics'],
                              **{k:v for k,v in d.items() if k!='case_id'})
                          for r,d in zip(rows,diagnostic_rows)])
    metrics = list(rows[0]['metrics'])+['citations_span_halves','answer_within_90_words']
    # Collapse stochastic repetitions before comparing the 12 paired questions.
    cases = frame.groupby(['album', 'policy', 'probe'])[metrics].mean().reset_index()
    album_means = cases.groupby(['album', 'policy'])[metrics].mean().reset_index()
    overall = album_means.groupby('policy')[metrics].mean().reset_index()
    full_tokens = float(overall.loc[overall.policy == 'full', 'input_tokens'].iloc[0])
    overall['input_token_reduction_vs_full'] = 1 - overall.input_tokens / full_tokens
    cases.to_csv(OUT / 'case_means.csv', index=False)
    album_means.to_csv(OUT / 'album_means.csv', index=False)
    overall.to_csv(OUT / 'policy_means.csv', index=False)

    usage = defaultdict(lambda: dict(calls=0, input_tokens=0, output_tokens=0))
    for path in (PRIVATE / 'calls').glob('*.json'):
        result = json.loads(path.read_text())
        group = usage[result['kind']]
        group['calls'] += 1
        group['input_tokens'] += result['usage']['prompt_tokens']
        group['output_tokens'] += result['usage']['completion_tokens']
    if usage:
        save('usage_by_stage.json', dict(usage))
    else:
        assert (OUT/'usage_by_stage.json').exists(), 'Provider usage export required for offline analysis'

    analyze_arcs()
    print(overall.to_string(index=False))
    print('Exported 144 runs, 12 paired questions, 4 album summaries.')


def analyze_arcs():
    arcs = []
    for album in ALBUMS:
        private_cards=PRIVATE / f"cards-{album['id']}.json"
        cards = json.loads((private_cards if private_cards.exists() else OUT/private_cards.name).read_text())
        # Eight ordinal theme annotations; exploratory distance, not acoustics.
        themes = ['attachment','rupture','agency','escape','identity',
                  'social_critique','mortality','renewal']
        x = np.array([[c['themes'][t] for t in themes] for c in cards], dtype=float)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        unit = x / np.maximum(norm, 1e-12)
        similarity = unit @ unit.T
        canonical = float(np.mean(np.diag(similarity, 1)))
        rng = np.random.default_rng(41)
        shuffled = []
        for _ in range(10000):
            order = rng.permutation(len(cards))
            shuffled.append(float(np.mean(similarity[order[:-1], order[1:]])))
        # Descriptive location in an order randomization reference, not a
        # population inference or evidence of compositional intent.
        arcs.append(dict(album=album['id'], canonical_adjacent_cosine=canonical,
                         shuffled_mean=float(np.mean(shuffled)),
                         shuffled_p025=float(np.quantile(shuffled, .025)),
                         shuffled_p975=float(np.quantile(shuffled, .975)),
                         fraction_shuffles_at_least_canonical=float(np.mean(np.array(shuffled)>=canonical)),
                         themes=themes, profiles=x.tolist(),
                         tension=[c['tension'] for c in cards]))
        save(f"cards-{album['id']}.json", cards)
        probe_file=PRIVATE/f"probes-{album['id']}.json"
        if probe_file.exists():
            save(probe_file.name,json.loads(probe_file.read_text()))
        summary_file=PRIVATE/f"summaries-{album['id']}.json"
        if summary_file.exists():
            save(f"final-memory-{album['id']}.json",json.loads(summary_file.read_text())[-1])
    save('arc_diagnostics.json', arcs)


if __name__ == '__main__':
    import sys
    analyze_arcs() if '--arcs-only' in sys.argv else main()
