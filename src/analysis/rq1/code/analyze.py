"""Offline RQ1 reanalysis. No model calls or network access.

Run from any directory: uv run --isolated --no-project --with numpy --with pandas
  --with scikit-learn python code/analyze.py
"""
from __future__ import annotations
import collections, hashlib, io, json, platform, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
D, T = ROOT / 'data', ROOT / 'tables'
MODELS = ['llama3_8b', 'qwen2_7b', 'granite3_2_8b']
METHODS = ['pe2', 'btape']
NAMES = {'llama3_8b': 'Llama', 'qwen2_7b': 'Qwen2', 'granite3_2_8b': 'Granite'}
GROUPS = [(m, o) for m in MODELS for o in METHODS]
LABELS = ['Non-security', 'Security']
B, RNG_SEED, TOL = 2000, 20261021, 1e-12

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(path, x): path.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def csv(name, rows):
    f = pd.DataFrame(rows)
    f.to_csv(T/name, index=False, encoding='utf-8-sig', float_format='%.17g')
    return f
def metric(c):
    """Confusion order TN,FP,FN,TP; supports vectorized bootstrap arrays."""
    c = np.asarray(c, dtype=float)
    tn, fp, fn, tp = np.moveaxis(c, -1, 0)
    def div(a,b): return np.divide(a,b,out=np.zeros_like(np.broadcast_arrays(a,b)[0]),where=b!=0)
    pos, neg = tp+fn, tn+fp
    sf, nf = div(2*tp,2*tp+fp+fn), div(2*tn,2*tn+fp+fn)
    return {'macro_f1': (sf+nf)/2, 'weighted_f1':div(sf*pos+nf*neg,pos+neg),
            'accuracy':div(tp+tn,pos+neg), 'security_precision':div(tp,tp+fp),
            'security_recall':div(tp,pos), 'security_f1':sf,
            'nonsecurity_f1':nf, 'nonsecurity_fpr':div(fp,neg),
            'balanced_accuracy':(div(tp,pos)+div(tn,neg))/2,
            'mcc':div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))}
def conf(y,p): return np.bincount(2*np.asarray(y,dtype=int)+np.asarray(p,dtype=int),minlength=4)
def score(y,p): return {k:float(v) for k,v in metric(conf(y,p)).items()}
def check_metrics(records, expected):
    y=np.array([r['gold']=='Security' for r in records])
    p=np.array([r['prediction']=='Security' for r in records])
    got=score(y,p)
    for k in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']:
        assert abs(got[k]-expected[k]) <= TOL, (k,got[k],expected[k])
    assert abs(got['mcc']-expected['matthews_correlation']) <= TOL
    for k in ['precision','recall','f1']:
        assert abs(got['security_'+k]-expected['per_class']['Security'][k]) <= TOL
    assert expected['invalid_count']==0
    return got
def assert_votes(r):
    vs=r['vote_labels']
    assert len(vs)==3 and all(v in LABELS for v in vs)
    assert r['prediction']==collections.Counter(vs).most_common(1)[0][0]
    assert r['gold'] in LABELS
def interval(v): return np.quantile(v,[.025,.975]).tolist()

def main():
    config={'analysis_version':'rq1-sqj-2026-10-21','bootstrap_replicates':B,'analysis_seed':RNG_SEED,
      'main_estimand':'mean of five paired per-cell Macro-F1 gains, conditional on frozen source runs',
      'bootstrap':'shared repository multiplicities across all routes and target views; fixed source runs',
      'primary_views':['T_All','T_NFR'],'sensitivity_views':['T_All_no_security','T_NFR_no_security'],
      'lexical_regex':r'(?i)\bsecurity\b','lexical_field':'requirement_text',
      'source_selection':'historical validation weighted-F1 maximum; equal scores prefer newer iteration',
      'labels':LABELS,'zero_division':0,'audit_tolerance':TOL,
      'post_result_reanalysis':True,'models':MODELS,'optimizers':METHODS,'source_split_seeds':list(range(40,45)),
      'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}
    dump(ROOT/'analysis_config.json',config)
    expected_hashes={'source_30cells_analysis_slim_v1.zip':'edeb08b3bc2556fe69f3e9a935e86d6d3f61b21e2b95123509a3f97baf5b7c11',
      'gitreq_enriched_v3.csv':'d765fb052feccb0fc0d031595b855834637d03662d798c090c309e32b6b772e7'}
    hashes={p.name:sha(p.read_bytes()) for p in D.iterdir() if p.is_file() and p.name!='selected_prompts.jsonl'}
    for k,v in expected_hashes.items(): assert hashes[k]==v,(k,hashes[k])
    dump(ROOT/'input_hashes.json',hashes)
    with zipfile.ZipFile(D/'source_30cells_analysis_slim_v1.zip') as z:
        def jl(n): return [json.loads(s) for s in z.read(n).decode().splitlines()]
        cells=jl('cells.jsonl'); cand=jl('candidates.jsonl')
        vp=jl('validation_predictions.jsonl'); sp=jl('source_test_predictions.jsonl')
        events=jl('iteration_events.jsonl')
        src_bytes=z.read('SecReq_510_BTAPE.csv')
        assert sha(src_bytes)=='d6d1535bacb754f3390a60638f9de38bacd72499f694f34b14808b01b5c77083'
        src=pd.read_csv(io.BytesIO(src_bytes))
    assert [len(cells),len(cand),len(vp),len(sp),len(events)]==[30,630,96390,12240,600]
    assert len({c['cell_id'] for c in cells})==30 and len({c['prompt_id'] for c in cand})==630
    byid={c['prompt_id']:c for c in cand}; bycell=collections.defaultdict(list)
    for c in cand:
        assert sha(c['prompt_text'].encode())==c['prompt_sha256']
        assert c['demo_ids']==[]
        bycell[c['cell_id']].append(c)
    vpred=collections.defaultdict(list); spred=collections.defaultdict(list)
    for r in vp:
        assert_votes(r); assert r['gold']==src.iloc[r['item_id']]['label']
        vpred[r['prompt_id']].append(r)
    for r in sp:
        assert_votes(r); assert r['gold']==src.iloc[r['item_id']]['label']
        assert r['stage'] in ['p0','p_star']
        spred[(r['cell_id'],{'p0':'p0','p_star':'pstar'}[r['stage']])].append(r)
    split_map={}; source=[]; selected={}; p0s={}; duplicates=0
    for c in cells:
        cid=c['cell_id']; cs=sorted(bycell[cid],key=lambda x:x['iteration'])
        assert [x['iteration'] for x in cs]==list(range(21))
        sets=[set(c['split'][k]) for k in ['pool_ids','val_ids','test_ids']]
        assert list(map(len,sets))==[153,153,204] and len(set.union(*sets))==510
        assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        ss=[sorted(s) for s in sets]
        if c['seed'] in split_map: assert ss==split_map[c['seed']]
        split_map[c['seed']]=ss
        earlier={}
        for p in cs:
            rr=sorted(vpred[p['prompt_id']],key=lambda x:x['item_id'])
            assert len(rr)==153 and {r['item_id'] for r in rr}==sets[1]
            check_metrics(rr,p['validation_metrics'])
            vec=tuple((r['prediction'],tuple(r['vote_labels'])) for r in rr)
            if p['prompt_sha256'] in earlier:
                assert earlier[p['prompt_sha256']]==vec
                duplicates+=1
            earlier[p['prompt_sha256']]=vec
        best=max(cs,key=lambda x:(x['validation_metrics']['weighted_f1'],x['iteration']))
        assert [x['iteration'] for x in cs if x['is_selected_best']]==[best['iteration']]
        assert c['best_iteration']==best['iteration']
        assert abs(c['best_validation_weighted_f1']-best['validation_metrics']['weighted_f1'])<=TOL
        selected[cid]=best; p0s[cid]=cs[0]
        row={'cell_id':cid,'model':c['model'],'optimizer':c['optimizer'],'seed':c['seed'],
            'best_iteration':best['iteration'],'p0_hash':cs[0]['prompt_sha256'],
            'pstar_hash':best['prompt_sha256'],'same_prompt_as_p0':cs[0]['prompt_text']==best['prompt_text'],
            'unique_candidates':len(earlier),'candidate_occurrences':len(cs)}
        for stage, p in [('p0',cs[0]),('pstar',best)]:
            rr=spred[(cid,stage)]
            assert len(rr)==204 and {r['item_id'] for r in rr}==sets[2]
            assert all(r['prompt_id']==p['prompt_id'] for r in rr)
            got=check_metrics(rr,c['source_test_'+stage+'_metrics'])
            for k,v in got.items(): row['test_'+stage+'_'+k]=v
            for k in ['macro_f1','weighted_f1','accuracy']:
                row['val_'+stage+'_'+k]=p['validation_metrics'][k]
        for domain in ['test','val']:
            for k in ['macro_f1','weighted_f1','accuracy']:
                row[domain+'_delta_'+k]=row[domain+'_pstar_'+k]-row[domain+'_p0_'+k]
        source.append(row)
    sdf=csv('source_cells.csv',source).set_index('cell_id',drop=False)
    assert len({p['prompt_sha256'] for p in p0s.values()})==1
    rich=pd.read_csv(D/'source_cells_original.csv').set_index('cell_id')
    for cid,r in sdf.iterrows():
        for stage in ['p0','pstar']:
            for k in ['macro_f1','weighted_f1','accuracy']:
                assert abs(r[f'test_{stage}_{k}']-rich.loc[cid,f'{stage}_test_{k}'])<=TOL
    print('Source: all 630 validation and 60 endpoint metric records independently verified.',flush=True)

    git=pd.read_csv(D/'gitreq_enriched_v3.csv').set_index('gitreq_id',drop=False)
    pred=pd.read_csv(D/'target_paired_predictions_slim.csv.gz')
    assert len(git)==6302 and git.index.is_unique and git.repository.nunique()==4080
    assert len(pred)==189060 and not pred.duplicated(['model','optimizer','seed','gitreq_id']).any()
    assert (git.binary_label==np.where(git.original_label=='Security','Security','Non-security')).all()
    ids=git.index.to_list(); y=(git.binary_label=='Security').to_numpy()
    lexical=git.requirement_text.str.contains(r'\bsecurity\b',case=False,regex=True).to_numpy()
    masks={'T_All':np.ones(len(git),bool),'T_NFR':(git.original_label!='Functional').to_numpy(),
       'T_All_no_security':~lexical,'T_NFR_no_security':~lexical&(git.original_label!='Functional').to_numpy()}
    assert y.sum()==1646 and masks['T_NFR'].sum()==5771
    views=list(masks); repos, ridx=np.unique(git.repository.to_numpy(),return_inverse=True)
    routes={}; route_vectors=[]; cellroutes={}; outcomes=[]; cats=[]
    for cid,c in sdf.iterrows():
        sub=pred[(pred.model==c.model)&(pred.optimizer==c.optimizer)&(pred.seed==c.seed)].set_index('gitreq_id').loc[ids]
        assert len(sub)==6302
        for a,b in [('gold_binary','binary_label'),('original_label','original_label'),('repository','repository')]:
            assert (sub[a].to_numpy()==git[b].to_numpy()).all(),(cid,a)
        assert np.array_equal(sub.has_security_word.to_numpy(),lexical)
        pp=[]
        for stage in ['p0','pstar']:
            raw=sub[stage+'_prediction']; assert set(raw)<=set(LABELS)
            p=(raw=='Security').to_numpy(); pp.append(p)
            key=(c.model,c[stage+'_hash'])
            if key in routes: assert np.array_equal(p,route_vectors[routes[key]])
            else: routes[key]=len(route_vectors); route_vectors.append(p)
        cellroutes[cid]=[routes[(c.model,c.p0_hash)],routes[(c.model,c.pstar_hash)]]
        for view,mask in masks.items():
            rr={'cell_id':cid,'model':c.model,'optimizer':c.optimizer,'seed':int(c.seed),'view':view,
                'n':int(mask.sum()),'security_n':int(y[mask].sum()),'nonsecurity_n':int((~y[mask]).sum()),
                'same_prompt_as_p0':bool(c.same_prompt_as_p0),'p0_hash':c.p0_hash,'pstar_hash':c.pstar_hash}
            for stage,p in zip(['p0','pstar'],pp):
                rr.update({stage+'_'+k:v for k,v in score(y[mask],p[mask]).items()})
                rr.update({stage+'_'+k:int(v) for k,v in zip(['tn','fp','fn','tp'],conf(y[mask],p[mask]))})
            for k in metric([1,1,1,1]): rr['delta_'+k]=rr['pstar_'+k]-rr['p0_'+k]
            a=pp[0]==y; b=pp[1]==y
            for tag,tmask in [('fixed',~a&b),('broken',a&~b),('preserved_correct',a&b),('preserved_wrong',~a&~b),
                            ('security_fixed',~a&b&y),('security_broken',a&~b&y),
                            ('nonsecurity_fixed',~a&b&~y),('nonsecurity_broken',a&~b&~y)]:
                rr[tag]=int((tmask&mask).sum())
            assert rr['fixed']-rr['broken']==(rr['pstar_tp']+rr['pstar_tn'])-(rr['p0_tp']+rr['p0_tn'])
            outcomes.append(rr)
        for category in git.original_label.unique():
            mask=(git.original_label==category).to_numpy(); rate=[float(p[mask].mean()) for p in pp]
            cats.append({'cell_id':cid,'model':c.model,'optimizer':c.optimizer,'seed':int(c.seed),
               'category':category,'n':int(mask.sum()),'metric':'Security recall' if category=='Security' else 'FPR',
               'p0_rate':rate[0],'pstar_rate':rate[1],'delta_pp':100*(rate[1]-rate[0])})
    out=csv('target_cells.csv',outcomes); catdf=csv('category_cells.csv',cats)
    assert len(routes)==29
    predmat=np.asarray(route_vectors).T
    # Verify independent formulas against sklearn for every endpoint on T_All.
    for p in predmat.T:
        a=score(y,p)
        assert abs(a['macro_f1']-f1_score(y,p,average='macro',labels=[0,1],zero_division=0))<TOL
        assert abs(a['weighted_f1']-f1_score(y,p,average='weighted',labels=[0,1],zero_division=0))<TOL
        assert abs(a['mcc']-matthews_corrcoef(y,p))<TOL
    old=pd.read_csv(D/'rq1_group_summary.csv').set_index(['model','optimizer'])
    old_routes=pd.read_csv(D/'target_routes_original.csv')
    for _,r in out[out.view.isin(['T_All','T_NFR'])].iterrows():
        prefix='all' if r['view']=='T_All' else 'nfr'
        for stage in ['p0','pstar']:
            route=old_routes[(old_routes.model==r.model)&(old_routes.prompt_id==('p0' if stage=='p0' else r.cell_id))]
            assert len(route)==1
            rr=route.iloc[0]
            assert rr.prompt_sha256==r[stage+'_hash'] and rr.rows==6302 and rr.votes_per_item==3
            for k in ['macro_f1','weighted_f1','security_f1','security_recall','security_precision','tn','fp','fn','tp']:
                assert abs(r[stage+'_'+k]-rr[prefix+'_'+k])<=TOL,(r.cell_id,stage,k)
    for (m,o),block in out.groupby(['model','optimizer']):
        for view,prefix in [('T_All','all'),('T_NFR','nfr')]:
            v=block[block.view==view]
            assert abs(v.delta_macro_f1.mean()-old.loc[(m,o),prefix+'_mean_delta_macro_f1'])<TOL
    # Raw source votes are rechecked above. Target slim has majority labels only;
    # original target vote check is retained and explicitly distinguished.
    prior=json.loads((D/'rq1_analysis_checks.json').read_text())
    manifest=json.loads((D/'route_manifest_original.json').read_text())
    assert len(manifest['routes'])==33 and manifest['votes_per_item']==3
    assert manifest['gitreq_enriched_sha256']==hashes['gitreq_enriched_v3.csv']
    assert manifest['prior_target_pilots_disclosed']
    assert 'RQ1_COMPLETE {"routes":33,"target_rows":6302}' in (D/'rq1_run.log').read_text()
    for route in manifest['routes']:
        cid=route['prompt_id']
        if cid=='p0': expected=next(iter(p0s.values()))['prompt_sha256']
        else: expected=selected[cid]['prompt_sha256']
        assert route['semantic_sha256']==expected
    print('Target: 189,060 paired rows; 29 unique prediction vectors; prior group means reproduced.',flush=True)

    # Repository bootstrap. Each draw resamples 4,080 repository IDs with replacement;
    # multiplicity is retained and identical for every route/view in the replicate.
    nrepo,nroute=len(repos),len(routes)
    counts=np.zeros((len(views),nrepo,nroute,4),dtype=np.float64)
    for vi,view in enumerate(views):
        mask=masks[view]
        for j in range(nroute):
            code=2*y.astype(int)+predmat[:,j].astype(int)
            np.add.at(counts[vi,:,j,:],(ridx[mask],code[mask]),1)
    rng=np.random.default_rng(RNG_SEED)
    boot={v:{k:np.empty((B,nroute)) for k in metric([1,1,1,1])} for v in views}
    for start in range(0,B,100):
        draws=rng.integers(0,nrepo,size=(min(100,B-start),nrepo))
        mult=np.array([np.bincount(d,minlength=nrepo) for d in draws],dtype=float)
        assert np.all(mult.sum(axis=1)==nrepo)
        for vi,view in enumerate(views):
            bc=(mult@counts[vi].reshape(nrepo,-1)).reshape(len(mult),nroute,4)
            assert np.all((bc[:,:,2]+bc[:,:,3]>0)&(bc[:,:,0]+bc[:,:,1]>0))
            for k,val in metric(bc).items(): boot[view][k][start:start+len(mult)]=val
        if (start+100)%500==0: print('Bootstrap',start+100,'/',B,flush=True)
    cirows=[]; grouped=[]; seedrows=[]
    for m,o in GROUPS:
        srcgroup=sdf[(sdf.model==m)&(sdf.optimizer==o)]
        cids=sorted(srcgroup.index)
        r0=[cellroutes[c][0] for c in cids]; rs=[cellroutes[c][1] for c in cids]
        for view in views:
            block=out[(out.model==m)&(out.optimizer==o)&(out.view==view)]
            row={'model':m,'optimizer':o,'view':view,'n_seeds':5,
              'p0_macro_f1':block.p0_macro_f1.mean(),'pstar_macro_f1_mean':block.pstar_macro_f1.mean(),
              'gain_pp':100*block.delta_macro_f1.mean(),'gain_median_pp':100*block.delta_macro_f1.median(),
              'gain_min_pp':100*block.delta_macro_f1.min(),'gain_max_pp':100*block.delta_macro_f1.max(),
              'positive_seeds':int((block.delta_macro_f1>TOL).sum()),
              'tied_seeds':int((block.delta_macro_f1.abs()<=TOL).sum()),
              'negative_seeds':int((block.delta_macro_f1 < -TOL).sum()),
              'same_prompt_as_p0':int(srcgroup.same_prompt_as_p0.sum()),
              'unique_pstar_texts':srcgroup.pstar_hash.nunique(),
              'source_test_gain_macro_pp':100*srcgroup.test_delta_macro_f1.mean(),
              'source_test_gain_weighted_pp':100*srcgroup.test_delta_weighted_f1.mean(),
              'source_val_gain_weighted_pp':100*srcgroup.val_delta_weighted_f1.mean(),
              'fixed_mean':block.fixed.mean(),'broken_mean':block.broken.mean(),
              'net_mean':(block.fixed-block.broken).mean()}
            for k in metric([1,1,1,1]):
                delta=100*(boot[view][k][:,rs]-boot[view][k][:,r0]).mean(axis=1)
                lo,hi=interval(delta)
                row[k+'_gain_pp']=100*block['delta_'+k].mean()
                row[k+'_ci_low_pp'],row[k+'_ci_high_pp']=lo,hi
                row['p0_'+k]=block['p0_'+k].mean();row['pstar_'+k+'_mean']=block['pstar_'+k].mean()
            grouped.append(row)
        for cid in cids:
            for view in views:
                i,j=cellroutes[cid]; a=100*(boot[view]['macro_f1'][:,j]-boot[view]['macro_f1'][:,i])
                lo,hi=interval(a)
                cirows.append({'cell_id':cid,'view':view,'ci_low_pp':lo,'ci_high_pp':hi})
    gdf=csv('group_summary.csv',grouped)
    out=out.merge(pd.DataFrame(cirows),on=['cell_id','view'],validate='one_to_one')
    csv('target_cells.csv',out)
    # Group contrasts use the same repository draw and fixed matched source seeds.
    contrasts=[]
    for m in MODELS:
        pe=sorted(sdf[(sdf.model==m)&(sdf.optimizer=='pe2')].index)
        bt=sorted(sdf[(sdf.model==m)&(sdf.optimizer=='btape')].index)
        for v in ['T_All','T_NFR']:
            a=[cellroutes[c][1] for c in pe];b=[cellroutes[c][1] for c in bt]
            delta=100*(boot[v]['macro_f1'][:,b]-boot[v]['macro_f1'][:,a]).mean(axis=1)
            q=gdf[(gdf.model==m)&(gdf.view==v)].set_index('optimizer')
            contrasts.append({'model':m,'view':v,'btape_minus_pe2_pp':q.loc['btape','gain_pp']-q.loc['pe2','gain_pp'],
                              'ci_low_pp':interval(delta)[0],'ci_high_pp':interval(delta)[1],
                              'interpretation':'conditional comparison of these frozen outputs; not optimizer causal effect'})
    csv('optimizer_contrasts.csv',contrasts)
    # Counts are per deployment, averaged over five seeds; never unique people/items across runs.
    cg=catdf.groupby(['model','optimizer','category','metric','n'],sort=False).agg(
       p0_rate=('p0_rate','mean'),pstar_rate_mean=('pstar_rate','mean'),delta_mean_pp=('delta_pp','mean'),
       delta_min_pp=('delta_pp','min'),delta_max_pp=('delta_pp','max')).reset_index()
    csv('category_summary.csv',cg)
    source_summary=[]
    for m,o in GROUPS:
        b=sdf[(sdf.model==m)&(sdf.optimizer==o)]
        r={'model':m,'optimizer':o,'n_seeds':5,'p0_selected_seeds':int(b.same_prompt_as_p0.sum()),
           'unique_candidates_min':int(b.unique_candidates.min()),'unique_candidates_max':int(b.unique_candidates.max())}
        for domain in ['val','test']:
            for k in ['macro_f1','weighted_f1']:
                for stage in ['p0','pstar']:r[f'{domain}_{stage}_{k}_mean']=b[f'{domain}_{stage}_{k}'].mean()
                r[f'{domain}_gain_{k}_pp']=100*b[f'{domain}_delta_{k}'].mean()
                r[f'{domain}_gain_{k}_min_pp']=100*b[f'{domain}_delta_{k}'].min()
                r[f'{domain}_gain_{k}_max_pp']=100*b[f'{domain}_delta_{k}'].max()
        source_summary.append(r)
    csv('source_summary.csv',source_summary)
    joined=out[out.view.isin(['T_All','T_NFR'])].merge(sdf[['cell_id','test_delta_macro_f1','val_delta_weighted_f1']].reset_index(drop=True),on='cell_id')
    csv('source_target_pairs.csv',joined)
    signs=[]
    for v in ['T_All','T_NFR']:
        b=joined[joined.view==v]
        sign=lambda a:np.where(a>TOL,'positive',np.where(a < -TOL,'negative','tie'))
        z=pd.DataFrame({'source':sign(b.test_delta_macro_f1),'target':sign(b.delta_macro_f1)})
        for (a,t),n in z.value_counts().items():signs.append({'view':v,'source_sign':a,'target_sign':t,'cells':int(n)})
    csv('gain_direction_counts.csv',signs)
    # Exact text dedup is an alternative estimand, not an independent-sample expansion.
    sens=[]
    for m,o in GROUPS:
        for v in views:
            b=out[(out.model==m)&(out.optimizer==o)&(out.view==v)]
            unique=b.drop_duplicates('pstar_hash')
            sens.append({'model':m,'optimizer':o,'view':v,'seed_weighted_gain_pp':100*b.delta_macro_f1.mean(),
                'unique_prompt_weighted_gain_pp':100*unique.delta_macro_f1.mean(),'n_unique':len(unique),
                'changed_prompt_only_gain_pp':100*b[~b.same_prompt_as_p0].delta_macro_f1.mean()})
    csv('prompt_dedup_sensitivity.csv',sens)
    dist=[]
    for v,mask in masks.items():
        dist.append({'view':v,'items':int(mask.sum()),'security':int(y[mask].sum()),
                     'nonsecurity':int((~y[mask]).sum()),'repositories':git[mask].repository.nunique()})
    csv('dataset_views.csv',dist)
    csv('category_distribution.csv',git.groupby('original_label').agg(n=('gitreq_id','size'),repositories=('repository','nunique')).reset_index())
    # Save all frozen endpoint texts for audit and later human interpretation.
    promptrows=[]
    for cid in sdf.index:
        for stage,p in [('p0',p0s[cid]),('pstar',selected[cid])]:
            promptrows.append({'cell_id':cid,'stage':stage,**{k:p[k] for k in ['prompt_id','iteration','prompt_sha256','prompt_text']}})
    (D/'selected_prompts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in promptrows))
    env=[]
    for m in MODELS:
        f=manifest['model_fingerprints'][m]
        for _,sr in rich[rich.model_key==m].iterrows():
            assert sr.model_revision==f['local_artifact']['expected_revision']
            assert sr.weight_index_sha256==f['local_artifact']['weight_index_sha256']
            assert sr.tokenizer_template_sha256==f['tokenizer_chat_template_sha256']
            assert sr.classification_interface==('package_minimal' if m=='qwen2_7b' else 'appendix_f')
            assert sr.voting_runs==3 and sr.max_batch_requests==1 and sr['dtype']==f['dtype']
        env.append({'model':m,'model_name':f['model_name'],'revision':f['local_artifact']['expected_revision'],
           'weight_index_sha256':f['local_artifact']['weight_index_sha256'],
           'chat_template_sha256':f['tokenizer_chat_template_sha256'],
           'wrapper':'package_minimal' if m=='qwen2_7b' else 'appendix_f',
           **{k:f[k] for k in ['vllm_version','torch_version','transformers_version','dtype','logical_gpu_name','max_batch_requests','max_model_len','classification_max_model_len']}})
    csv('environment.csv',env)
    audit={'passed':True,'source_cells':30,'source_candidates':630,'source_validation_rows_rechecked':96390,
       'source_test_rows_rechecked':12240,'source_votes_rechecked':3*(96390+12240),
       'source_metric_records_recomputed':690,'source_duplicate_occurrences_with_identical_votes':duplicates,
       'target_paired_rows_recomputed':189060,'target_logical_routes':33,'unique_model_prompt_routes':29,
       'target_majority_labels_valid':True,'target_raw_votes_reaudited_in_this_revision':False,
       'target_vote_audit_basis':'retained 2026-10-13 audit and original completed-run log; current slim lacks raw votes',
       'prior_target_rows_audited':prior['target_rows_audited'],'prior_target_votes_audited':prior['votes_audited'],
       'selection_rule_reproduced':True,'same_seed_splits_equal':True,'split_sizes':[153,153,204],
       'source_target_model_template_fingerprints_matched':True,
       'reproduced_prior_group_means_tolerance':TOL,'cross_model_cache_sharing':False,
       'all_current_checks_passed':True,'bootstrap_draws':B,'bootstrap_all_views_have_both_classes':True,
       'input_source':'final v4 slim archive, not desktop source-optimization-v1',
       'model_calls':0,'server_writes':0}
    dump(ROOT/'analysis_checks.json',audit)
    outputhash={p.name:sha(p.read_bytes()) for p in sorted(T.glob('*.csv'))}
    oldhashpath=ROOT/'table_hashes.json'
    if oldhashpath.exists():
        oldhash=json.loads(oldhashpath.read_text()); assert oldhash==outputhash,'Non-deterministic tables'
        audit['rerun_table_hashes_identical']=True;dump(ROOT/'analysis_checks.json',audit)
    dump(oldhashpath,outputhash)
    print(gdf[gdf.view=='T_All'][['model','optimizer','p0_macro_f1','pstar_macro_f1_mean','gain_pp','macro_f1_ci_low_pp','macro_f1_ci_high_pp','positive_seeds','tied_seeds','negative_seeds']].to_string(index=False))
    print('All analysis checks passed. Output:',ROOT,flush=True)

if __name__=='__main__': main()
