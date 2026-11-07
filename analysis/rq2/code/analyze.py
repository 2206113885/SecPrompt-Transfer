"""Independent offline RQ2 reanalysis. Uses only archived predictions.

Main analysis preserves all frozen rules. source-macro selection is explicitly
a new, post-hoc diagnostic, never a replacement for the historical p_source.
"""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json, zipfile, re, importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

R=Path(__file__).resolve().parents[1]; D=R/'data'; T=R/'tables'
TOL=1e-12; B=2000; SEED=20261013
VIEWS=['T_All','T_NFR']; MODELS=['llama3_8b','qwen2_7b','granite3_2_8b']
PAIRS=[('primary_weighted_to_macro','source_validation_weighted_f1','target_macro_f1'),
       ('aligned_macro_to_macro','source_validation_macro_f1','target_macro_f1'),
       ('weighted_to_weighted','source_validation_weighted_f1','target_weighted_f1')]

def sha(b):return hashlib.sha256(b).hexdigest()
def read(name):return pd.read_csv(D/'original_tables'/name)
def save(name,rows):pd.DataFrame(rows).to_csv(T/name,index=False,float_format='%.17g')
def close(a,b):
    assert np.allclose(a,b,atol=TOL,rtol=0,equal_nan=True),(np.asarray(a),np.asarray(b))
def ranks(x):
    x=np.asarray(x);order=np.argsort(-x,kind='stable');out=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and abs(x[order[i]]-x[order[j]])<=TOL:j+=1
        out[order[i:j]]=(i+1+j)/2;i=j
    return out
def corr(x,y):
    rx,ry=ranks(x),ranks(y)
    if len(x)<2 or np.ptp(rx)==0 or np.ptp(ry)==0:return (np.nan,np.nan)
    return float(kendalltau(rx,ry).statistic),float(np.corrcoef(rx,ry)[0,1])
def order(c,x):
    rr=ranks(x)
    return sorted(range(len(c)),key=lambda i:(rr[i],-int(c.iloc[i].last_iteration),c.iloc[i].prompt_sha256))
def scores(y,p,weights=None):
    p=np.atleast_2d(np.asarray(p,dtype=np.int64));y=np.asarray(y,dtype=np.int64)
    if not (np.isin(y,[0,1]).all() and np.isin(p,[0,1]).all()):raise ValueError('Invalid binary prediction or gold')
    w=np.ones(len(y),dtype=np.int64) if weights is None else np.asarray(weights)
    tp=(p*y)@w;fp=(p*(1-y))@w;fn=np.sum(y*w)-tp;tn=np.sum((1-y)*w)-fp
    return counts(tn,fp,fn,tp)
def counts(tn,fp,fn,tp):
    def div(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=b!=0)
    f1n=div(2*tn,2*tn+fp+fn);f1p=div(2*tp,2*tp+fp+fn);n=tn+fp+fn+tp
    return {'macro_f1':(f1n+f1p)/2,'weighted_f1':div(f1n*(tn+fp)+f1p*(tp+fn),n),
        'accuracy':div(tp+tn,n),'security_recall':div(tp,tp+fn),
        'security_precision':div(tp,tp+fp),'security_f1':f1p,
        'balanced_accuracy':(div(tp,tp+fn)+div(tn,tn+fp))/2}
def label(values):
    assert set(values)<= {'Security','Non-security'}
    return np.asarray(values)=='Security'
def rank_boot(x,ys):
    # Pairwise tau-b and average-rank rho, on the same bootstrap rows.
    x=np.asarray(x);ys=np.asarray(ys);n=len(x);a,b=np.triu_indices(n,1)
    dx=x[a]-x[b];dy=ys[:,a]-ys[:,b];tx=np.abs(dx)<=TOL;ty=np.abs(dy)<=TOL
    c=((dx*dy>0)&~tx&~ty).sum(1);d=((dx*dy<0)&~tx&~ty).sum(1)
    xt=(tx&~ty).sum(1);yt=(~tx&ty).sum(1);den=np.sqrt((c+d+xt)*(c+d+yt))
    tau=np.divide(c-d,den,out=np.full(len(ys),np.nan),where=den>0)
    rx=ranks(x);ry=rankdata(-ys,axis=1,method='average');xc=rx-rx.mean();yc=ry-ry.mean(1)[:,None]
    den2=np.sqrt((xc**2).sum()*(yc**2).sum(1));rho=np.divide(yc@xc,den2,out=np.full(len(ys),np.nan),where=den2>0)
    return tau,rho,ry
def interval(x):return np.nanpercentile(x,[2.5,97.5])

def main():
    T.mkdir(exist_ok=True);audit={'new_model_calls':0,'original_files_modified':False,'independent_reanalysis':True}
    c=read('candidate_scores.csv');cr=read('cell_rank_transfer.csv');cg=read('cell_selection_gap.csv')
    target=pd.read_csv(D/'gitreq_enriched_v3.csv',keep_default_na=False);y=label(target.binary_label)
    assert np.array_equal(y,target.original_label.to_numpy()=='Security')
    assert sha((D/'gitreq_enriched_v3.csv').read_bytes())=='d765fb052feccb0fc0d031595b855834637d03662d798c090c309e32b6b772e7'
    masks={'T_All':np.ones(len(y),bool),'T_NFR':target.original_label.to_numpy()!='Functional'}
    no_word=~target.requirement_text.str.contains(r'\bsecurity\b',case=False,regex=True).to_numpy()
    masks.update({'T_NoSecurityWord_All':no_word,'T_NoSecurityWord_NFR':no_word&masks['T_NFR']})
    with gzip.open(D/'candidate_prediction_vectors.jsonl.gz','rt') as f:
        header=json.loads(next(f));routes=[json.loads(s) for s in f]
    assert header['gitreq_ids']==target.gitreq_id.tolist()
    assert len(routes)==350
    lookup={};P=[];sources=Counter();vote_disagreements=0
    for i,r in enumerate(routes):
        pred=np.frombuffer(r['prediction'].encode(),np.uint8)-48
        vs=np.array([np.frombuffer(v.encode(),np.uint8)-48 for v in r['votes']])
        assert len(pred)==6302 and np.array_equal(pred,vs.sum(0)>=2)
        lookup[(r['model'],r['prompt_sha256'])]=i;P.append(pred);sources.update(r['sources'])
        vote_disagreements+=int(np.any(vs!=vs[0],axis=0).sum())
    P=np.stack(P).astype(np.int64);assert len(lookup)==350
    audit.update(routes=350,route_item_predictions=int(P.size),logical_votes=int(P.size*3),
                 prediction_sources=dict(sources),vote_disagreement_items=vote_disagreements)
    met={v:scores(y[m],P[:,m]) for v,m in masks.items()}
    for v,d in c.groupby('view'):
        ix=[lookup[(r.model,r.prompt_sha256)] for r in d.itertuples()]
        for k,vals in met[v].items():close(d['target_'+k],vals[ix])
    audit['target_metric_rows_recomputed']=len(c)
    # Exact source text lineage and votes, not JSON score summaries alone.
    z=zipfile.ZipFile(D/'source_30cells_analysis_slim_v1.zip')
    assert sha(z.read('SecReq_510_BTAPE.csv'))=='d6d1535bacb754f3390a60638f9de38bacd72499f694f34b14808b01b5c77083'
    source_csv=pd.read_csv(z.open('SecReq_510_BTAPE.csv'))
    jl=lambda name:[json.loads(s) for s in z.read(name).splitlines()]
    cells=jl('cells.jsonl');occ=jl('candidates.jsonl');vp=jl('validation_predictions.jsonl')
    cellmap={r['cell_id']:r for r in cells};op={r['prompt_id']:r for r in occ};vmap=defaultdict(dict)
    assert (len(cells),len(occ),len(vp))==(30,630,96390)
    for r in vp:
        assert source_csv.iloc[r['item_id']]['label']==r['gold']
        assert len(r['vote_labels'])==3 and Counter(r['vote_labels']).most_common(1)[0][0]==r['prediction']
        vmap[r['prompt_id']][r['item_id']]=(r['gold'],r['prediction'])
    for r in occ:
        assert sha(r['prompt_text'].encode())==r['prompt_sha256'] and not r['demo_ids']
        v=vmap[r['prompt_id']];assert len(v)==153
        sc=scores(label([v[i][0] for i in sorted(v)]),label([v[i][1] for i in sorted(v)]))
        for k in ['macro_f1','weighted_f1','accuracy','balanced_accuracy']:close(sc[k][0],r['validation_metrics'][k])
    for cell in cells:
        sets=[set(cell['split'][k]) for k in ['pool_ids','val_ids','test_ids']]
        assert [len(s) for s in sets]==[153,153,204] and len(set.union(*sets))==510
        hist=[r for r in occ if r['cell_id']==cell['cell_id']]
        assert sorted(r['iteration'] for r in hist)==list(range(21))
        winner=max(hist,key=lambda r:(r['validation_metrics']['weighted_f1'],r['iteration']))
        assert winner['is_selected_best'] and winner['iteration']==cell['best_iteration']
        peers=[r for r in cells if r['seed']==cell['seed']]
        assert all(r['split']==cell['split'] for r in peers)
    for row in c[c.view=='T_All'].itertuples():
        ids=json.loads(row.occurrence_ids);its=json.loads(row.occurrence_iterations)
        assert all(op[i]['prompt_sha256']==row.prompt_sha256 for i in ids)
        assert [op[i]['iteration'] for i in ids]==its
        assert all(vmap[i]==vmap[ids[0]] for i in ids)
        close(row.source_validation_weighted_f1,op[ids[-1]]['validation_metrics']['weighted_f1'])
    audit.update(source_occurrences_verified=630,source_validation_vote_rows_verified=96390,
        cell_unique_candidates=380,source_repeated_prediction_disagreements=0,source_selection_verified=30)
    # Restore explicitly named average ranks; old *_rank columns are tie-broken.
    cc=[];cells_out=[];top=[];sens=[]
    for (cid,v),d0 in c.groupby(['cell_id','view'],sort=True):
        d=d0.reset_index(drop=True);n=len(d);x=d.source_validation_weighted_f1.to_numpy();t=d.target_macro_f1.to_numpy()
        ir=int(np.flatnonzero(d.is_p_source)[0]);i0=int(np.flatnonzero(d.is_p0)[0]);isrc=order(d,x)[0]
        assert isrc==ir
        ir_m=order(d,d.source_validation_macro_f1.to_numpy())[0]
        targetorder=order(d,t);gap=100*(t.max()-t[ir]);span=100*np.ptp(t)
        for j,row in d.iterrows():
            cc.append({**row.to_dict(),'source_average_rank':ranks(x)[j],
                       'target_average_rank':ranks(t)[j],'target_tie_broken_rank':targetorder.index(j)+1})
        tau,rho=corr(x,t)
        info={'cell_id':cid,'model':d.model[0],'optimizer':d.optimizer[0],'seed':int(d.seed[0]),'view':v,
          'n_unique':n,'tau':tau,'rho':rho,'low_resolution':n<5,'p0_macro':t[i0],'p_source_macro':t[ir],
          'pool_best_macro':t.max(),'uniform_expectation_macro':t.mean(),'selection_gap_pp':gap,'target_range_pp':span,
          'normalized_gap':gap/span if span>TOL else np.nan,'p_source_minus_p0_pp':100*(t[ir]-t[i0]),
          'p_source_minus_uniform_pp':100*(t[ir]-t.mean()),'p_source_target_rank':ranks(t)[ir],
          'p_source_is_target_top_tie':t.max()-t[ir]<=TOL,'p_source_is_p0':ir==i0,
          'posthoc_source_macro_selected_hash':d.prompt_sha256[ir_m],
          'posthoc_macro_selector_changed':ir_m!=ir,
          'posthoc_macro_selector_gain_pp':100*(t[ir_m]-t[ir]),
          'posthoc_macro_selector_gap_pp':100*(t.max()-t[ir_m])}
        cells_out.append(info)
        if v in VIEWS:
            old=cg[(cg.cell_id==cid)&(cg.view==v)].iloc[0]
            for k in ['selection_gap_pp','target_range_pp','normalized_gap']:close(info[k],old[k])
            close(info['p_source_minus_uniform_pp'],old.p_source_minus_uniform_mean_pp)
            for pair,scol,tcol in PAIRS:
                a,b=corr(d[scol],d[tcol]);oldr=cr[(cr.cell_id==cid)&(cr.view==v)&(cr.metric_pair==pair)].iloc[0]
                close([a,b],[oldr.kendall_tau_b,oldr.spearman_rho])
            for k in [1,3,5]:
                valid=k<=n;src=order(d,x)[:k];dst=targetorder[:k]
                si=set(np.flatnonzero(x>=x[src[-1]]-TOL)) if valid else set()
                ti=set(np.flatnonzero(t>=t[dst[-1]]-TOL)) if valid else set()
                top.append({**{key:info[key] for key in ['cell_id','model','optimizer','seed','view','n_unique']},
                   'k':k,'eligible':valid,'trivial_full_pool':k==n,
                   'overlap':len(set(src)&set(dst))/k if valid else np.nan,
                   'tie_inclusive_jaccard':len(si&ti)/len(si|ti) if valid else np.nan})
    cs=pd.DataFrame(cells_out);save('candidate_scores_audited.csv',cc);save('topk_audited.csv',top)
    ot=read('topk_overlap.csv').set_index(['cell_id','view','k'])
    for row in top:
        if row['eligible']:
            old=ot.loc[(row['cell_id'],row['view'],row['k'])]
            close([row['overlap'],row['tie_inclusive_jaccard']],[old.deterministic_overlap,old.tie_inclusive_jaccard])
    audit['topk_rows_recomputed']=len(top)
    # Independently replay repository-grouped calibration and held-out evaluation.
    folds=read('target_group_folds.csv');assert folds.gitreq_id.tolist()==target.gitreq_id.tolist()
    assert folds.groupby('repository').fold.nunique().max()==1
    ff=folds.fold.to_numpy();comparator=read('target_selected_comparator.csv');comp_summary=read('target_selected_comparator_summary.csv')
    oof_saved=pd.read_csv(D/'original_tables/target_selected_oof_predictions.csv.gz')
    folds_out=[];oof_out=[]
    for (cid,v),d0 in c[c.view.isin(VIEWS)].groupby(['cell_id','view'],sort=True):
        d=d0.reset_index(drop=True);ix=[lookup[(r.model,r.prompt_sha256)] for r in d.itertuples()]
        pm=P[ix];ir=int(np.flatnonzero(d.is_p_source)[0]);i0=int(np.flatnonzero(d.is_p0)[0]);mask=masks[v];oof=np.full(len(y),-1)
        for f in range(5):
            held=mask&(ff==f);cal=mask&(ff!=f)
            assert not set(target.repository[held])&set(target.repository[cal])
            score=scores(y[cal],pm[:,cal])['macro_f1'];chosen=order(d,score)[0];oof[held]=pm[chosen,held]
            old=comparator[(comparator.cell_id==cid)&(comparator.view==v)&(comparator.fold==f)].iloc[0]
            assert old.selected_prompt_sha256==d.prompt_sha256[chosen];close(score[chosen],old.calibration_macro_f1)
            hs=scores(y[held],pm[[chosen,ir,i0]][:,held])['macro_f1']
            close(hs,[old.target_selected_heldout_macro_f1,old.p_source_heldout_macro_f1,old.p0_heldout_macro_f1])
            saved=oof_saved[(oof_saved.cell_id==cid)&(oof_saved.view==v)&(oof_saved.fold==f)]
            assert saved.gitreq_id.tolist()==target.gitreq_id[held].tolist()
            for name,pi in [('target_selected_prediction',chosen),('p_source_prediction',ir),('p0_prediction',i0)]:
                assert np.array_equal(label(saved[name]),pm[pi,held])
            folds_out.append(old.to_dict())
        pooled=scores(y[mask],np.stack([oof[mask],pm[ir,mask],pm[i0,mask]]))['macro_f1']
        old=comp_summary[(comp_summary.cell_id==cid)&(comp_summary.view==v)].iloc[0]
        close(pooled,[old.pooled_oof_target_selected_macro_f1,old.pooled_oof_p_source_macro_f1,old.pooled_oof_p0_macro_f1])
        oof_out.append(old.to_dict())
    audit.update(target_oof_rows_verified=len(oof_saved),calibration_selections_verified=300,repository_fold_leakage=0,
                 pooled_oof_summaries_verified=60,target_rank_rows_verified=180,target_gap_rows_verified=60)
    save('comparator_folds_audited.csv',folds_out);save('comparator_summary_audited.csv',oof_out)
    # Recompute source bootstrap selection frequency using precisely shared draws.
    rng=np.random.default_rng(SEED);draws={};stability=read('validation_rank_stability.csv')
    for seed in range(40,45):
        cell=next(r for r in cells if r['seed']==seed);ids=cell['split']['val_ids']
        oid=next(r['prompt_id'] for r in occ if r['cell_id']==cell['cell_id'])
        gy=label([vmap[oid][i][0] for i in ids]);pos=np.flatnonzero(gy);neg=np.flatnonzero(~gy)
        W=np.zeros((B,len(ids)),np.int64)
        for b in range(B):
            W[b]+=np.bincount(rng.choice(pos,len(pos)),minlength=len(ids));W[b]+=np.bincount(rng.choice(neg,len(neg)),minlength=len(ids))
        draws[seed]=(W,gy)
    stability_out=[]
    for cid,d0 in c[c.view=='T_All'].groupby('cell_id',sort=True):
        d=d0.reset_index(drop=True);ids=cellmap[cid]['split']['val_ids'];W,gy=draws[int(d.seed[0])]
        pm=np.asarray([label([vmap[json.loads(r.occurrence_ids)[-1]][i][1] for i in ids]) for r in d.itertuples()],dtype=np.int64)
        tp=W@(pm*gy).T;fp=W@(pm*(~gy)).T;fn=gy.sum()-tp;tn=(~gy).sum()-fp
        bs=counts(tn,fp,fn,tp)['weighted_f1'];pref=sorted(range(len(d)),key=lambda i:(-int(d.last_iteration[i]),d.prompt_sha256[i]))
        chosen=np.array([pref[j] for j in np.argmax(bs[:,pref]>=bs.max(1)[:,None]-TOL,axis=1)])
        freq=np.bincount(chosen,minlength=len(d))/B;ir=int(np.flatnonzero(d.is_p_source)[0])
        old=stability[(stability.cell_id==cid)].set_index('prompt_sha256')
        close(freq,[old.loc[h].source_validation_bootstrap_first_frequency for h in d.prompt_sha256])
        stability_out.append({'cell_id':cid,'model':d.model[0],'optimizer':d.optimizer[0],'seed':int(d.seed[0]),
          'n_unique':len(d),'p_source_reselection_frequency':freq[ir],
          'winner_frequency_max':freq.max(),'n_candidates_ever_selected':int((freq>0).sum())})
    save('source_stability_audited.csv',stability_out);audit['source_bootstrap_candidate_frequencies_verified']=380
    # Independent paired target bootstrap, all cells and both principal views.
    repos=sorted(target.repository.unique());ri={r:i for i,r in enumerate(repos)};ridx=target.repository.map(ri).to_numpy()
    rng=np.random.default_rng(SEED);W=np.array([np.bincount(rng.integers(0,len(repos),len(repos)),minlength=len(repos)) for _ in range(B)],dtype=float)
    boot={};ci_checks=[]
    for v in VIEWS:
        mask=masks[v];pos=mask&y;neg=mask&~y
        rp=np.zeros((len(repos),len(routes)));rn=np.zeros_like(rp)
        np.add.at(rp,ridx[pos],P[:,pos].T);np.add.at(rn,ridx[neg],P[:,neg].T)
        tp=W@rp;fp=W@rn;np_=W@np.bincount(ridx[pos],minlength=len(repos));nn=W@np.bincount(ridx[neg],minlength=len(repos))
        boot[v]=counts(nn[:,None]-fp,fp,np_[:,None]-tp,tp)
    for (cid,v),d0 in c[c.view.isin(VIEWS)].groupby(['cell_id','view'],sort=True):
        d=d0.reset_index(drop=True);ix=[lookup[(r.model,r.prompt_sha256)] for r in d.itertuples()];ir=int(np.flatnonzero(d.is_p_source)[0])
        bm=boot[v]['macro_f1'][:,ix];gap=100*(bm.max(1)-bm[:,ir]);span=100*np.ptp(bm,axis=1)
        old=cg[(cg.cell_id==cid)&(cg.view==v)].iloc[0]
        close(interval(gap),[old.selection_gap_ci95_low,old.selection_gap_ci95_high]);close(interval(span),[old.target_range_ci95_low,old.target_range_ci95_high])
        for pair,scol,tcol in PAIRS:
            bv=boot[v][tcol.removeprefix('target_')][:,ix];bt,br,_=rank_boot(d[scol],bv)
            rr=cr[(cr.cell_id==cid)&(cr.view==v)&(cr.metric_pair==pair)].iloc[0]
            close(interval(bt),[rr.kendall_tau_b_ci95_low,rr.kendall_tau_b_ci95_high])
            close(interval(br),[rr.spearman_rho_ci95_low,rr.spearman_rho_ci95_high])
            ci_checks.append({'cell_id':cid,'view':v,'metric_pair':pair,'all_intervals_match':True})
    save('bootstrap_independent_checks.csv',ci_checks);audit['bootstrap_rank_interval_rows_recomputed']=len(ci_checks)
    # Equal-cell summaries; do not pool candidates, votes, or overlapping seeds.
    cs=cs.merge(pd.DataFrame(stability_out)[['cell_id','p_source_reselection_frequency']],on='cell_id')
    co=pd.DataFrame(oof_out)[['cell_id','view','pooled_oof_target_selected_minus_p_source_pp','pooled_oof_target_selected_macro_f1']]
    cs=cs.merge(co,on=['cell_id','view'],how='left');save('cell_summary.csv',cs)
    save('posthoc_source_macro_selection.csv',cs[cs.view.isin(VIEWS)][['cell_id','model','optimizer','seed','view',
       'posthoc_source_macro_selected_hash','posthoc_macro_selector_changed','posthoc_macro_selector_gain_pp','posthoc_macro_selector_gap_pp']])
    gs=[]
    for (m,o,v),d in cs.groupby(['model','optimizer','view']):
        row={'model':m,'optimizer':o,'view':v,'n_cells':len(d),'n_unique_min':int(d.n_unique.min()),'n_unique_max':int(d.n_unique.max()),
         'low_resolution_cells':int(d.low_resolution.sum()),'positive_tau_cells':int((d.tau>TOL).sum()),
         'target_top_tie_cells':int(d.p_source_is_target_top_tie.sum()),'gap_le_1pp_cells':int((d.selection_gap_pp<=1+TOL).sum()),
         'source_above_uniform_cells':int((d.p_source_minus_uniform_pp>TOL).sum()),
         'oof_better_cells':int((d.pooled_oof_target_selected_minus_p_source_pp>TOL).sum())}
        for name in ['tau','rho','selection_gap_pp','target_range_pp','p_source_minus_uniform_pp','p_source_macro','p0_macro','pool_best_macro','uniform_expectation_macro',
             'p_source_reselection_frequency','pooled_oof_target_selected_minus_p_source_pp','pooled_oof_target_selected_macro_f1','posthoc_macro_selector_gain_pp']:
            for stat in ['mean','median','min','max']:row[name+'_'+stat]=getattr(d[name],stat)()
        gs.append(row)
    save('group_summary_audited.csv',gs)
    # Sensitivity summaries preserve denominators and NA.
    original_s=read('sensitivity_results.csv');srows=[]
    for (analysis,v,pair),d in original_s[original_s.cell_id.notna()].groupby(['analysis','view','metric_pair']):
        srows.append({'analysis':analysis,'view':v,'metric_pair':pair,'cells':len(d),'valid_tau':int(d.kendall_tau_b.notna().sum()),
          'tau_median':d.kendall_tau_b.median(),'tau_mean':d.kendall_tau_b.mean(),'positive_tau':int((d.kendall_tau_b>TOL).sum())})
    save('sensitivity_overview.csv',srows)
    audit['schema_notes']=[
      'Original candidate_diversity.csv repeats 30 cells for two views but omits view; use new candidate_diversity_by_view.csv.',
      'Original candidate_scores macro_f1_rank is tie-broken ordinal rank, not average rank; new candidate_scores_audited names both explicitly.',
      'Posthoc source-macro selector is a new offline diagnostic; not the frozen primary selection policy.']
    div=[]
    for (cid,v),d in c.groupby(['cell_id','view']):
        div.append({'cell_id':cid,'model':d.model.iloc[0],'optimizer':d.optimizer.iloc[0],'seed':int(d.seed.iloc[0]),'view':v,
           'n_unique':len(d),'occurrences':sum(len(json.loads(x)) for x in d.occurrence_ids),
           'distinct_target_macro':len(set(ranks(d.target_macro_f1))),'distinct_source_weighted':len(set(ranks(d.source_validation_weighted_f1))),
           'low_resolution':len(d)<5})
    save('candidate_diversity_by_view.csv',div)
    env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy']}
    (R/'analysis_environment.json').write_text(json.dumps(env,indent=2)+'\n')
    audit['status']='PASS';(R/'analysis_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(audit,indent=2,ensure_ascii=False));print(pd.DataFrame(gs).query('view=="T_All"')[[
        'model','optimizer','tau_mean','selection_gap_pp_median','p_source_minus_uniform_pp_mean','pooled_oof_target_selected_minus_p_source_pp_mean']].to_string(index=False))

if __name__=='__main__':main()
