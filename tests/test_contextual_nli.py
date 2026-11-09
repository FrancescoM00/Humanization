import math

import numpy as np
import pytest

from semantics.contextual_nli import build_plan, contiguous_contexts, score_plan, summarize_units


class Tokenizer:
    def encode(self, a, b=None, **kwargs):
        return list(range(len(a.split())+(len(b.split()) if b else 0)+3))

    def num_special_tokens_to_add(self, pair=True):
        return 3


def aligned_pair():
    import hashlib
    pair={'id':'x','original':'A one. B two. C three.','modified':'A one and B two.'}
    item={k+'_sha256':hashlib.sha256(pair[k].encode()).hexdigest() for k in ('original','modified')}
    item.update(original_chunks=['A one.','B two.','C three.'],modified_chunks=['A one and B two.'])
    return pair,{'x':item}


def test_contexts_cover_native_contiguous_windows_without_reordering():
    assert contiguous_contexts(['A.','B.','C.'],2)==['A.','B.','C.','A. B.','B. C.']
    assert contiguous_contexts(['A.','B.','C.'],3)[-1]=='A. B. C.'
    with pytest.raises(ValueError):
        contiguous_contexts([],2)


def test_plan_records_over_budget_document_without_truncating_it():
    pair,alignment=aligned_pair()
    plan=build_plan(pair,alignment,Tokenizer(),max_pair_tokens=11)
    forward=[x for x in plan if x['direction']=='forward']
    document=next(x for x in forward if x['mode']=='document')
    assert document['premises']==[]
    assert document['n_candidates_over_budget']==1
    assert next(x for x in forward if x['mode']=='window2')['premises']==alignment['x']['original_chunks']


def test_alignment_hash_mismatch_is_rejected():
    pair,alignment=aligned_pair()
    pair['original']+=' changed'
    with pytest.raises(ValueError,match='hash'):
        build_plan(pair,alignment,Tokenizer())


def test_incomplete_coverage_does_not_selectively_average_easy_hypotheses():
    units=[dict(id='x',direction='forward',mode='document',eligible=True,entailment=.9,
                n_candidates_eligible=1,n_candidates_over_budget=0),
           dict(id='x',direction='forward',mode='document',eligible=False,entailment=None,
                n_candidates_eligible=0,n_candidates_over_budget=1)]
    summary=summarize_units(units)[0]
    assert summary['coverage_fraction']==.5 and not summary['complete']
    assert math.isnan(summary['mean']) and math.isnan(summary['min'])


def test_nested_evidence_maximum_is_monotonic_not_an_accuracy_measure():
    pair,alignment=aligned_pair()
    plan=build_plan(pair,alignment,Tokenizer(),max_pair_tokens=30)
    def predict(pairs):
        return np.array([[min(.99,len(p.split())/10),.005,1-min(.99,len(p.split())/10)-.005]
                         for p,h in pairs])
    scores,units=score_plan(plan,predict,{'entailment':0,'neutral':1,'contradiction':2})
    f={x['mode']:x['mean'] for x in scores if x['direction']=='forward'}
    assert f['single_all']<=f['window2']<=f['window3']
    assert units and all(x['eligible'] for x in units)
