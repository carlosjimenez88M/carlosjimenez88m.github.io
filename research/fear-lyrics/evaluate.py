"""Post-fit diagnostics: language association and paired question-family bootstrap.

These summaries do not change model parameters or the test queries. Resampling
is conditional on analyst-created cases, not population or independent gold.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

def main():
    meta = pd.DataFrame(json.loads((HERE/'corpus-manifest.json').read_text()))[['song_id','language']]
    language = []
    for path in sorted(RESULTS.glob('*/section_assignments.csv')):
        data = pd.read_csv(path)
        data = data[~data.exact_repeat].merge(meta,on='song_id',validate='many_to_one')
        language.append(dict(model=path.parent.name, diagnostic='posthoc_language_association',
                             nmi_topic_language=normalized_mutual_info_score(data.language,data.topic_id),
                             documents=len(data), interpretation='Confound diagnostic, not thematic validity.'))
    pd.DataFrame(language).to_csv(RESULTS/'topic_language_association.csv',index=False)
    from analyze import agreement
    assignments=[]
    for path in sorted(RESULTS.glob('*/section_assignments.csv')):
        table=pd.read_csv(path)
        table=table[~table.exact_repeat].sort_values('document_id')
        assignments.append((path.parent.name,table.document_id.tolist(),table.topic_id.tolist()))
    comparisons=[]
    for i,(left,ids,labels) in enumerate(assignments):
        for right,other_ids,other_labels in assignments[i+1:]:
            if ids!=other_ids: raise ValueError('Cross-model comparison must use the same blocks.')
            comparisons.append(dict(left_model=left,right_model=right,**agreement(labels,other_labels)))
    pd.DataFrame(comparisons).to_csv(RESULTS/'embedding_model_cluster_agreement.csv',index=False)
    rng = np.random.default_rng(20261011)
    rows=[]
    for name in ('rag','rag-e5','rag-openai-large','rag-openai-small'):
        file=RESULTS/name/'runs.json'
        if not file.exists(): continue
        data=pd.DataFrame(json.loads(file.read_text()))
        data=data[data.split=='test']
        for method in ('bm25','dense','hybrid'):
            group=data[data.method==method]
            for metric in ('recall_at_5','context_support_recall','negative_safe_abstention'):
                selected=group[group[metric].notna()]
                # EN/ES queries in a family are paired, never independent samples.
                paired=selected.groupby(['family','arm'])[metric].mean().unstack()
                difference=(paired.bounded_contextual-paired.one_pass).to_numpy(dtype=float)
                draws=rng.integers(0,len(difference),size=(5000,len(difference)))
                boot=difference[draws].mean(axis=1)
                rows.append(dict(model=name,method=method,metric=metric,families=len(difference),
                                 difference_contextual_minus_onepass=float(difference.mean()),
                                 ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                                 resamples=5000,seed=20261011,
                                 scope='Paired conditional question-family bootstrap; no population/generalization claim.'))
    pd.DataFrame(rows).to_csv(RESULTS/'rag_family_sensitivity.csv',index=False)
    print(json.dumps(dict(language_models=len(language),family_diagnostics=len(rows),status='executed')))

if __name__=='__main__': main()
