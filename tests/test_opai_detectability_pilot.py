"""Controlli su soglie, orientamento AUROC e integrità del pannello OpAI."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.opai_detectability_pilot import sha, summarize, read_panel

ROOT=Path(__file__).resolve().parents[1]


def fixture_panel(tmp_path):
    config=json.loads((ROOT/'results_opai_detectability_pilot/config_pilot.json').read_text())
    config=copy.deepcopy(config)
    config['n_documents_per_domain']=1
    (tmp_path/'config_pilot.json').write_text(json.dumps(config))
    records=[]
    for domain in config['domains']:
        for version in ['v0','v5','v8']:
            records.append(dict(doc_id=domain,group_id=domain,domain=domain,version=version,
                editor='test_editor',trajectory_id=domain+'|test_editor',record_id=domain+version,
                n_words=2,text='test text',text_sha256='fixture'))
    (tmp_path/'panel.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    header=dict(config_sha256=sha(tmp_path/'config_pilot.json'),panel_sha256=sha(tmp_path/'panel.jsonl'))
    (tmp_path/'selection.json').write_text(json.dumps(header))
    for name in config['frozen_thresholds']:
        threshold=config['frozen_thresholds'][name]['raw_threshold']
        rows=[]
        for r in records:
            direction=-1 if name.startswith('binoculars') else 1
            delta={'v0':0,'v5':.01,'v8':-.01}[r['version']]
            rows.append({k:v for k,v in r.items() if k!='text'} | dict(
                detector=name,raw_score=threshold+delta*direction,detected=delta>0))
        pd.DataFrame(rows).to_csv(tmp_path/f'scores_{name}.csv',index=False)
        (tmp_path/f'run_{name}.json').write_text(json.dumps(header))
    return config


def test_ties_wilson_and_auc_direction(tmp_path):
    config=fixture_panel(tmp_path)
    summarize(config,tmp_path)
    rates=pd.read_csv(tmp_path/'summary_rates.csv')
    opai=rates[rates.dataset.eq('OpAI-Bench') & rates.domain.eq('overall')]
    assert opai[opai.version.eq('v0')].rate.eq(0).all()
    assert opai[opai.version.eq('v0')].wilson95_high.gt(0).all()
    assert opai[opai.version.eq('v5')].rate.eq(1).all()
    auc=pd.read_csv(tmp_path/'auroc.csv')
    assert auc[auc.comparison.eq('v0_vs_v5')].auroc.eq(1).all()
    assert auc[auc.comparison.eq('v0_vs_v8')].auroc.eq(0).all()
    assert json.loads((tmp_path/'experiment.json').read_text())['status']=='completed'
    feasibility=pd.read_csv(tmp_path/'feasibility.csv')
    assert feasibility[feasibility.version.eq('v5')].proceed.all()
    assert not feasibility[feasibility.version.eq('v8')].proceed.any()


def test_different_document_cannot_disappear_in_join(tmp_path):
    config=fixture_panel(tmp_path)
    path=tmp_path/'scores_openai_roberta_base.csv'
    f=pd.read_csv(path);f.loc[0,'doc_id']='unrelated';f.to_csv(path,index=False)
    with pytest.raises(ValueError,match='stessi documenti'):
        summarize(config,tmp_path)


def test_changed_texts_and_wrong_frozen_decision_rejected(tmp_path):
    config=fixture_panel(tmp_path)
    path=tmp_path/'scores_openai_roberta_base.csv'
    f=pd.read_csv(path);f.loc[0,'detected']=True;f.to_csv(path,index=False)
    with pytest.raises(ValueError,match='soglia congelata'):
        summarize(config,tmp_path)
    with (tmp_path/'panel.jsonl').open('a') as handle:handle.write('\n')
    with pytest.raises(ValueError,match='testi selezionati'):
        read_panel(tmp_path,config)


def test_missing_detector_is_pending_not_zero_detection(tmp_path):
    config=fixture_panel(tmp_path)
    (tmp_path/'scores_binoculars_falcon7b.csv').unlink()
    summarize(config,tmp_path)
    status=json.loads((tmp_path/'experiment.json').read_text())
    assert status['pending_detectors']==['binoculars_falcon7b']
    rates=pd.read_csv(tmp_path/'summary_rates.csv')
    assert not ((rates.dataset=='OpAI-Bench') & (rates.detector=='binoculars_falcon7b')).any()
