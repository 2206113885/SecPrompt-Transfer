"""Run the unchanged RQ3 statistics using the compact archived predictions.

Only the original raw-response loader is replaced. This adapter does not
recreate responses; the stored labels and vote counts are checked first.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import numpy as np
import analyze_original as engine

ROOT=Path(__file__).resolve().parents[2]


def audit_compact(cfg,out):
    data=ROOT/'data/rq3'
    a=np.load(data/'predictions.npz',allow_pickle=False)
    with (ROOT/'data/datasets/gitreq_enriched_v3.csv').open() as f: target=list(csv.DictReader(f))
    keys=a['route_ids'].tolist();pred=a['predictions'].astype(bool);gold=a['gold'][0].astype(bool)
    assert pred.shape==(189,6302) and a['votes'].shape==(189,6302,1)
    assert np.array_equal(pred,a['votes'][:,:,0])
    assert all(np.array_equal(row,a['gold'][0]) for row in a['gold'])
    assert np.array_equal(a['item_ids'][0],[r['gitreq_id'] for r in target])
    routes=[json.loads(line) for line in (data/'route_manifest.jsonl').read_text().splitlines()]
    assert keys==[r['route_id'] for r in routes]
    logical=[json.loads(line) for line in (data/'logical_matrix_manifest.jsonl').read_text().splitlines()]
    assert len(logical)==217 and sum(r['is_baseline'] for r in logical)==7
    names,repo=np.unique([r['repository'] for r in target],return_inverse=True)
    masks=np.array([np.ones(len(gold),bool),[r['original_label']!='Functional' for r in target]])
    assert len(names)==4080 and masks.sum(1).tolist()==[6302,5771]
    metadata={f'analysis/{name}':(ROOT/'paper/data/rq3'/name).read_bytes()
              for name in ['route_scores.csv','group_summary.csv']}
    prov=json.loads((data/'prediction_provenance.json').read_text())
    sources=Counter()
    for r in prov:sources.update(r['prediction_sources'])
    checks={'routes':189,'logical_winner_cells':210,'baselines':7,'items':6302,
            'repositories':4080,'prediction_sources':dict(sources),
            'input_kind':'verified compact prediction and vote arrays',
            'raw_response_loader_replaced':True,'new_model_calls':0}
    return target,keys,pred,gold,repo,masks,logical,metadata,checks


if __name__=='__main__':
    cfg=json.loads((ROOT/'paper/data/rq3-systematic/analysis_protocol.json').read_text())
    cfg['output']=str(ROOT/'reproduced/rq3')
    cfg['target']=str(ROOT/'data/datasets/gitreq_enriched_v3.csv')
    cfg['archive']='not required: compact prediction arrays'
    engine.self_test()
    engine.audit=audit_compact
    complete=Path(cfg['output'])/'analysis_checks.json'
    if complete.exists():
        print('RQ3 analysis already completed; remove only reproduced/rq3 to repeat.')
    else:
        engine.analyze(cfg)
