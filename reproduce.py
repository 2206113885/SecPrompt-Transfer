"""Offline replication entry point; never imports inference code or calls a model.

python reproduce.py --check
python reproduce.py --full
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduced'
TOL = 1e-12


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


math = load_module('rq2_math', ROOT / 'analysis/rq2/code/analyze.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL, equal_nan=True)


def basic_check():
    OUT.mkdir(exist_ok=True)
    manifest = ROOT / 'MANIFEST.sha256'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            expected, path = line.split('  ', 1)
            assert sha(ROOT / path) == expected, path
    target = pd.read_csv(ROOT / 'data/datasets/gitreq_enriched_v3.csv', keep_default_na=False)
    y = (target.binary_label.to_numpy() == 'Security').astype(np.uint8)
    masks = {'T_All': np.ones(len(y), bool), 'T_NFR': target.original_label.to_numpy() != 'Functional'}
    no_word = ~target.requirement_text.str.contains(r'\bsecurity\b', case=False, regex=True).to_numpy()
    masks.update(T_NoSecurityWord_All=no_word, T_NoSecurityWord_NFR=no_word & masks['T_NFR'])
    assert len(y) == 6302 and int(y.sum()) == 1646 and int(masks['T_NFR'].sum()) == 5771
    assert sha(ROOT / 'data/datasets/gitreq_enriched_v3.csv') == 'd765fb052feccb0fc0d031595b855834637d03662d798c090c309e32b6b772e7'
    checks = {'new_model_calls': 0, 'metrics_tolerance': TOL, 'stages': {}}
    arrays, manifests = {}, {}
    for stage, nr, ni, nv in [('rq2',350,6302,3),('rq3',189,6302,1),
                              ('source_test',380,204,3),('concise',3,6302,3)]:
        with np.load(ROOT / 'data' / stage / 'predictions.npz', allow_pickle=False) as archive:
            a = {key: archive[key] for key in archive.files}
        assert a['predictions'].shape == (nr,ni)
        assert a['votes'].shape == (nr,ni,nv)
        assert np.isin(a['votes'], [0,1]).all() and np.isin(a['predictions'], [0,1]).all()
        assert np.array_equal(a['predictions'], a['votes'].sum(axis=2) > nv/2)
        assert len(set(a['route_ids'])) == nr
        if stage != 'source_test':
            assert np.all(a['item_ids'] == target.gitreq_id.to_numpy()[None,:])
            assert np.all(a['gold'] == y[None,:])
        routes = (json.loads((ROOT/'data/concise/route_manifest.json').read_text())['routes'] if stage=='concise'
                  else read_jl(ROOT / 'data' / stage / 'route_manifest.jsonl'))
        assert len(routes) == nr
        arrays[stage], manifests[stage] = a, routes
        checks['stages'][stage] = {'routes':nr,'prediction_rows':nr*ni,'vote_labels':nr*ni*nv}

    a = arrays['rq2']; routes = manifests['rq2']
    lookup = {(r['model'],r['prompt_sha256']):i for i,r in enumerate(routes)}
    target_scores = {v:math.scores(y[m], a['predictions'][:,m]) for v,m in masks.items()}
    expected = pd.read_csv(ROOT/'analysis/rq2/data/original_tables/candidate_scores.csv')
    for v, rows in expected.groupby('view'):
        idx = [lookup[(r.model,r.prompt_sha256)] for r in rows.itertuples()]
        for metric, values in target_scores[v].items():
            close(values[idx], rows['target_'+metric])
    checks['rq2_candidate_score_rows_recomputed'] = len(expected)

    # Recompute original source-validation votes and scores; never select a new p*.
    z = zipfile.ZipFile(ROOT/'data/source/source_30cells_analysis_slim_v1.zip')
    jl = lambda n: [json.loads(s) for s in z.read(n).splitlines()]
    candidates,cells,vp = jl('candidates.jsonl'),jl('cells.jsonl'),jl('validation_predictions.jsonl')
    assert (len(candidates),len(cells),len(vp)) == (630,30,96390)
    byprompt = {}
    for row in vp:
        assert len(row['vote_labels']) == 3
        assert all(x in ('Security','Non-security') for x in row['vote_labels'])
        pred = 'Security' if row['vote_labels'].count('Security') >= 2 else 'Non-security'
        assert row['prediction'] == pred
        byprompt.setdefault(row['prompt_id'], []).append(row)
    for c in candidates:
        assert hashlib.sha256(c['prompt_text'].encode()).hexdigest() == c['prompt_sha256']
        rows = byprompt[c['prompt_id']]; assert len(rows)==153
        scores = math.scores([r['gold']=='Security' for r in rows], [r['prediction']=='Security' for r in rows])
        for k in ('weighted_f1','macro_f1','accuracy','balanced_accuracy'):
            close(scores[k][0],c['validation_metrics'][k])
    winner={}; p0={}
    for c in cells:
        hist=[p for p in candidates if p['cell_id']==c['cell_id']]
        best=max(hist,key=lambda p:(p['validation_metrics']['weighted_f1'],p['iteration']))
        assert best['is_selected_best'] and best['iteration']==c['best_iteration']
        winner[(c['model'],c['optimizer'],c['seed'])]=best['prompt_sha256']
        p0[c['model']]=next(p['prompt_sha256'] for p in hist if p['iteration']==0)
    # Verify that RQ1 endpoint predictions are precisely the matching RQ2 routes.
    item_lookup={x:i for i,x in enumerate(target.gitreq_id)}
    paired=0
    with gzip.open(ROOT/'analysis/rq1/data/target_paired_predictions_slim.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            i=item_lookup[r['gitreq_id']]; model=r['model']
            key=(model,r['optimizer'],int(r['seed']))
            for field,h in [('p0_prediction',p0[model]),('pstar_prediction',winner[key])]:
                assert a['predictions'][lookup[model,h],i] == (r[field]=='Security')
            paired+=1
    assert paired == 30*6302
    checks['rq1_paired_rows_matched_to_rq2'] = paired

    # Source-test supplement: verify ALL 380 routes, including the reused endpoints.
    st=arrays['source_test']; scores=math.scores(st['gold'][0],st['predictions'][0])
    source_csv=pd.read_csv(ROOT/'data/datasets/SecReq_510_BTAPE.csv')
    smap={}
    for i,r in enumerate(manifests['source_test']):
        ids=np.asarray(r['test_ids']); assert np.array_equal(ids,st['item_ids'][i])
        close(st['gold'][i],source_csv.iloc[ids].label.to_numpy()=='Security')
        smap[r['cell_id'],r['prompt_sha256']]=math.scores(st['gold'][i],st['predictions'][i])
    src_expected=pd.read_csv(ROOT/'paper/data/source-test-supplement/analysis/candidate_scores.csv')
    for r in src_expected.itertuples():
        for k in ('macro_f1','weighted_f1','accuracy'):
            close(smap[r.cell_id,r.prompt_sha256][k][0],getattr(r,'source_test_'+k))
    checks['source_test_score_rows_recomputed']=len(src_expected)
    # Reconstruct the supplement's ranks and gaps from the verified candidate scores.
    rank_ref=pd.read_csv(ROOT/'paper/data/source-test-supplement/analysis/cell_rank_reference.csv')
    gap_ref=pd.read_csv(ROOT/'paper/data/source-test-supplement/analysis/cell_selection_gaps.csv')
    groups={(cid,v):df.reset_index(drop=True) for (cid,v),df in src_expected.groupby(['cell_id','view'])}
    rebuilt_rank=[]
    for r in rank_ref.to_dict('records'):
        df=groups[r['cell_id'],r['view']]
        xk,yk=r['metric_pair'].split(' -> ')
        tau,rho=math.corr(df[xk].to_numpy(),df[yk].to_numpy())
        close([tau,rho],[r['tau_b'],r['rho']])
        rebuilt_rank.append({**r,'tau_b':tau,'rho':rho})
    rebuilt_gap=[]
    for r in gap_ref.to_dict('records'):
        df=groups[r['cell_id'],r['view']];metric=r['metric']
        idx=int(np.flatnonzero(df.is_selected_best)[0]);i0=int(np.flatnonzero(df.is_p0)[0])
        values=df[metric].to_numpy();winner=values[idx];span=float(np.ptp(values))
        actual={'p0_score':values[i0],'p_source_score':winner,'pool_best_score':max(values),
                'gain_over_p0_pp':100*(winner-values[i0]),'selection_gap_pp':100*(max(values)-winner),
                'target_or_test_range_pp':100*span,
                'normalized_gap':(max(values)-winner)/span if span>TOL else np.nan,
                'uniform_unique_score':float(values.mean()),'p_source_average_rank':math.ranks(values)[idx]}
        for key,value in actual.items():close(value,r[key])
        rebuilt_gap.append({**r,**actual})
    pd.DataFrame(rebuilt_rank).to_csv(OUT/'source_test_rank_reference.csv',index=False)
    pd.DataFrame(rebuilt_gap).to_csv(OUT/'source_test_selection_gaps.csv',index=False)
    checks['source_test_rank_rows_recomputed']=len(rebuilt_rank)
    checks['source_test_gap_rows_recomputed']=len(rebuilt_gap)

    for stage, path, model_field in [('rq3','paper/data/rq3/route_scores.csv','deployer'),
                                     ('concise','paper/data/concise-definition-supplement/analysis/concise_scores.csv','model')]:
        arr=arrays[stage]; rs=manifests[stage]
        lm={r['model']:i for i,r in enumerate(rs)}
        lr={rid:i for i,rid in enumerate(arr['route_ids'])}
        ex=pd.read_csv(ROOT/path)
        for v, df in ex.groupby('view'):
            m=masks[v]; met=math.scores(y[m],arr['predictions'][:,m])
            idx=[lr[r.route_id] for r in df.itertuples()] if stage=='rq3' else [lm[r.model] for r in df.itertuples()]
            for k in ('macro_f1','weighted_f1','accuracy','security_precision','security_recall','security_f1'):
                close(met[k][idx],df[k])
        checks[stage+'_score_rows_recomputed']=len(ex)
    checks['status']='PASS'
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


def full_analysis():
    for rq in ('rq1','rq2'):
        dest=OUT/rq
        if not dest.exists():
            shutil.copytree(ROOT/'analysis'/rq,dest)
        # Keep one canonical copy of the shared datasets in the repository.
        shared = {'source_30cells_analysis_slim_v1.zip': ROOT/'data/source/source_30cells_analysis_slim_v1.zip',
                  'gitreq_enriched_v3.csv': ROOT/'data/datasets/gitreq_enriched_v3.csv'}
        for name, origin in shared.items():
            shutil.copy2(origin, dest/'data'/name)
        (dest/'tables').mkdir(exist_ok=True)
        subprocess.run([sys.executable,str(dest/'code/analyze.py')],check=True)
        subprocess.run([sys.executable,str(dest/'code/test_analysis.py')],check=True)
    # Reuse the original RQ3 statistical engine with a compact, checked input adapter.
    subprocess.run([sys.executable,str(ROOT/'analysis/rq3/from_compact.py')],check=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--full',action='store_true')
    args=ap.parse_args()
    basic_check()
    if args.full: full_analysis()
