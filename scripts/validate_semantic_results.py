#!/usr/bin/env python3
"""Check coverage, numerical integrity and saved evidence of the semantic run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate(data_dir, output):
    pairs_path = data_dir / 'pairs.jsonl'
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    ids = {pair['id'] for pair in pairs}
    assert len(ids) == len(pairs), 'Duplicate prepared IDs'
    digest = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    manifest = json.loads((output / 'preparation_manifest.json').read_text())
    assert digest == manifest['pairs_sha256']
    columns = {
        'lexical': ['lexical_rouge_l_f1', 'lexical_bleu'],
        'embedding': ['embedding_cosine'],
        'bertscore': ['bert_chunk_precision', 'bert_chunk_recall', 'bert_chunk_f1'],
        'nli': ['nli_forward_mean', 'nli_reverse_mean'],
    }
    stages = {}
    for stage, required in columns.items():
        frame = pd.read_csv(output / f'{stage}.csv')
        run = json.loads((output / f'{stage}_run.json').read_text())
        assert run['status'] == 'executed' and run['pairs_sha256'] == digest
        assert len(frame) == len(ids) and not frame.id.duplicated().any() and set(frame.id) == ids
        assert np.isfinite(frame[required].to_numpy()).all(), f'Missing {stage} metric'
        lower = -1 if stage == 'embedding' else 0
        assert ((frame[required] >= lower - 1e-6) & (frame[required] <= 1 + 1e-6)).all().all()
        stages[stage] = {'rows': len(frame), 'sha256': hashlib.sha256((output / f'{stage}.csv').read_bytes()).hexdigest()}
        if stage == 'nli':
            for prefix in ['nli_', 'local_nli_']:
                for direction in ['forward_', 'reverse_']:
                    names = [prefix + direction + suffix for suffix in ['mean', 'neutral_mean', 'contradiction_mean']]
                    valid = frame[names].dropna()
                    assert np.allclose(valid.sum(axis=1), 1, atol=1e-6, rtol=0)
            if run.get('cache_only'):
                assert run['new_native_predictions_during_aggregation'] == 0
        if stage == 'bertscore':
            local_protocols = frame.local_bert_protocol.value_counts().to_dict()
    alignment = json.loads((data_dir / 'alignment.json').read_text())
    assert set(alignment) == ids
    for pair in pairs:
        item = alignment[pair['id']]
        for side in ['original', 'modified']:
            text = pair[side]
            assert hashlib.sha256(text.encode()).hexdigest() == item[side + '_sha256']
            # Segmentation may normalize whitespace, but must retain every other character.
            assert ''.join(text.split()) == ''.join(''.join(item[side + '_chunks']).split())
        assert len(item['forward_candidates']) == len(item['modified_chunks'])
        assert len(item['reverse_candidates']) == len(item['original_chunks'])
    keys = set()
    with (data_dir / 'nli_probability_cache.jsonl').open() as handle:
        header = json.loads(next(handle))
        assert header['type'] == 'nli_probability_cache_v1'
        for line in handle:
            value = json.loads(line)
            assert value['key'] not in keys
            keys.add(value['key'])
            prob = np.array(value['probabilities'])
            assert len(prob) == 3 and np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            assert np.isclose(prob.sum(), 1, atol=1e-6, rtol=0)
    plan = json.loads((output / 'nli_plan.json').read_text())
    assert len(keys) == plan['unique_model_pairs']
    parity = json.loads((output / 'nli_onnx_real_pair_parity.json').read_text())
    assert parity['max_probability_abs_difference'] <= parity['tolerance']
    assert parity['argmax_agreement'] == 1
    bert_reference = json.loads((output / 'bertscore_reference_validation.json').read_text())
    assert bert_reference['passed']
    merged = pd.read_csv(output / 'scores.csv')
    assert len(merged) == len(ids) and set(merged.id) == ids and not merged.id.duplicated().any()
    result = {'passed': True, 'pairs_sha256': digest, 'n_pairs': len(pairs),
              'datasets': dict(Counter(pair['dataset'] for pair in pairs)),
              'controlled_conditions': dict(Counter(pair['perturbation'] for pair in pairs if pair['dataset'] == 'controlled')),
              'stages': stages, 'nli_unique_cached_predictions': len(keys),
              'embedding_segmentation_retains_all_nonwhitespace_characters': True,
              'local_bert_protocol_counts': local_protocols,
              'nli_onnx_real_pair_parity': parity, 'bertscore_standard_reference': bert_reference}
    (output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/semantic_full')
    parser.add_argument('--output', type=Path, default=ROOT / 'results_semantic_full')
    args = parser.parse_args()
    validate(args.data_dir, args.output)
