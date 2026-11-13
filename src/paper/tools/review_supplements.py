"""Post-hoc review diagnostics; offline only, no source/target artifact writes.

Usage: python tools/review_supplements.py [--project-data PATH]
Reads the two dated analysis folders beside the manuscript by default.
All comparisons retain the historical source selection and exact prompt text.
"""
from __future__ import annotations
import argparse
import collections as co
import csv
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'review_2026-10-22'
EPS = 1e-12
MODELS = ['llama3_8b', 'qwen2_7b', 'granite3_2_8b']
NAMES = dict(zip(MODELS, ['Llama', 'Qwen2', 'Granite']))

def sha(b): return hashlib.sha256(b).hexdigest()
def csvout(name, rows):
    if not rows: return
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def f1(gold, pred):
    tp = sum(a == b == '1' for a,b in zip(gold,pred))
    tn = sum(a == b == '0' for a,b in zip(gold,pred))
    fp = sum(a == '0' and b == '1' for a,b in zip(gold,pred))
    fn = sum(a == '1' and b == '0' for a,b in zip(gold,pred))
    return ((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0) +
            (2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0))/2

def main():
    repository = ROOT.parent if (ROOT.parent/'data/datasets').is_dir() else ROOT.parents[1]
    ap=argparse.ArgumentParser(); ap.add_argument('--project-data',type=Path,default=repository)
    args=ap.parse_args(); b=args.project_data
    paths={'source':b/'data/source/source_30cells_analysis_slim_v1.zip',
           'vectors':b/'analysis/rq2/data/candidate_prediction_vectors.jsonl.gz',
           'gitreq':b/'data/datasets/gitreq_enriched_v3.csv',
           'scores':b/'analysis/rq2/data/original_tables/candidate_scores.csv'}
    OUT.mkdir(exist_ok=True); (OUT/'prompts').mkdir(exist_ok=True)
    z=zipfile.ZipFile(paths['source'])
    load=lambda n:[json.loads(x) for x in z.read(n).splitlines()]
    candidates=load('candidates.jsonl'); cells=load('cells.jsonl'); events=load('iteration_events.jsonl')
    vp=load('validation_predictions.jsonl'); sp=load('source_test_predictions.jsonl')
    byid={c['prompt_id']:c for c in candidates}; bycell=co.defaultdict(list)
    for c in candidates:
        assert sha(c['prompt_text'].encode())==c['prompt_sha256']
        bycell[c['cell_id']].append(c)
    for cs in bycell.values(): assert sorted(c['iteration'] for c in cs)==list(range(21))
    assert (len(candidates),len(cells),len(events),len(vp),len(sp))==(630,30,600,96390,12240)
    scores=list(csv.DictReader(paths['scores'].open()))
    scoremap={(r['cell_id'],r['prompt_sha256'],r['view']):r for r in scores}
    eventmap={(e['cell_id'],e['iteration']):e for e in events}
    policies=[]
    for c in cells:
        cid=c['cell_id']; cs=bycell[cid]; its={x['iteration']:x for x in cs}
        selected=[x for x in cs if x['is_selected_best']]
        assert len(selected)==1 and selected[0]['iteration']==c['best_iteration']
        ps=selected[0]; p0=its[0]; last=its[20]
        current=its[eventmap[cid,20]['current_after_iteration']]
        unique={x['prompt_sha256']:x for x in cs}
        for view in ['T_All','T_NFR']:
            sc=lambda x:float(scoremap[cid,x['prompt_sha256'],view]['target_macro_f1'])
            gen=[sc(x) for h,x in unique.items() if h!=p0['prompt_sha256']]
            policies.append(dict(cell_id=cid,model=c['model'],optimizer=c['optimizer'],seed=c['seed'],view=view,
                n_unique=len(unique),n_generated_unique=len(gen),p_source_iteration=ps['iteration'],
                final_current_iteration=current['iteration'],last_proposal_iteration=20,
                p_source_hash=ps['prompt_sha256'],last_proposal_hash=last['prompt_sha256'],
                final_current_hash=current['prompt_sha256'],p0_macro=sc(p0),p_source_macro=sc(ps),
                last_proposal_macro=sc(last),final_current_macro=sc(current),
                unique_uniform_macro=st.mean(sc(x) for x in unique.values()),
                generated_unique_uniform_macro=st.mean(gen) if gen else None,
                source_minus_last_pp=100*(sc(ps)-sc(last)),
                source_minus_final_current_pp=100*(sc(ps)-sc(current)),
                source_minus_generated_uniform_pp=100*(sc(ps)-st.mean(gen)) if gen else None,
                generated_uniform_minus_p0_pp=100*(st.mean(gen)-sc(p0)) if gen else None,
                last_minus_p0_pp=100*(sc(last)-sc(p0)),
                pool_gap_pp=100*(max(sc(x) for x in unique.values())-sc(ps))))
    csvout('selection_policy_cells.csv',policies)
    groups=[]
    for view in ['T_All','T_NFR']:
        for model in MODELS:
            for optimizer in ['pe2','btape']:
                rs=[r for r in policies if (r['view'],r['model'],r['optimizer'])==(view,model,optimizer)]
                ds=[r['source_minus_last_pp'] for r in rs]
                groups.append(dict(view=view,model=model,optimizer=optimizer,n=len(rs),
                    source_vs_last_win=sum(d>EPS for d in ds),source_vs_last_tie=sum(abs(d)<=EPS for d in ds),
                    source_vs_last_loss=sum(d<-EPS for d in ds),
                    source_minus_last_mean_pp=st.mean(ds),source_minus_last_min_pp=min(ds),source_minus_last_max_pp=max(ds),
                    source_minus_final_current_mean_pp=st.mean(r['source_minus_final_current_pp'] for r in rs),
                    source_minus_generated_uniform_mean_pp=st.mean(r['source_minus_generated_uniform_pp'] for r in rs),
                    generated_uniform_minus_p0_mean_pp=st.mean(r['generated_uniform_minus_p0_pp'] for r in rs),
                    last_minus_p0_mean_pp=st.mean(r['last_minus_p0_pp'] for r in rs)))
    csvout('selection_policy_groups.csv',groups)
    tex=[r'\begin{tabular}{llrrrr}',r'\toprule',
         r'Model & Optimizer & W/T/L & $p^*-p_{20}$ & $p^*-U_g$ & $U_g-p_0$ \\',r'\midrule']
    for r in groups[:6]:
        tex.append(f"{NAMES[r['model']]} & {'PE2' if r['optimizer']=='pe2' else 'BT-APE'} & "
          f"{r['source_vs_last_win']}/{r['source_vs_last_tie']}/{r['source_vs_last_loss']} & "
          f"{r['source_minus_last_mean_pp']:+.3f} & {r['source_minus_generated_uniform_mean_pp']:+.3f} & "
          f"{r['generated_uniform_minus_p0_mean_pp']:+.3f} \\")
        tex[-1] += '\\'
    tex += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables/rq2_selection_policy.tex').write_text('\n'.join(tex)+'\n')

    # Workload: cell accounting versus request-identity screening are separate.
    unique_cell={(c['cell_id'],c['prompt_sha256']) for c in candidates}
    cellmap={c['cell_id']:c for c in cells}
    required={(cid,h,int(i)) for cid,h in unique_cell for i in cellmap[cid]['split']['test_ids']}
    existing={(r['cell_id'],byid[r['prompt_id']]['prompt_sha256'],int(r['item_id'])) for r in sp}
    pending=required-existing
    request=lambda k:(cellmap[k[0]]['model'],k[1],k[2])
    reqroutes={request(k) for k in required}
    existing_routes={request(k) for k in existing}
    pendingroutes=reqroutes-existing_routes
    allpred=co.defaultdict(set)
    for r in sp+vp:
        p=byid[r['prompt_id']]
        key=(p['model'],p['prompt_sha256'],int(r['item_id']))
        allpred[key].add((r['gold'],r['prediction'],tuple(r['vote_labels'])))
    conflicts=[k for k,v in allpred.items() if len(v)>1]
    valreusable={k for k,v in allpred.items() if len(v)==1}
    all_remaining=reqroutes-valreusable
    timing=co.defaultdict(list)
    for c in candidates:
        u=c['validation_usage']
        assert u['calls']==459
        timing[c['model'],c['prompt_sha256']].append(u['duration_ns']/1e9/153)
    wr=[]
    for m in MODELS:
        pp=[k for k in pending if cellmap[k[0]]['model']==m]
        pr=[k for k in pendingroutes if k[0]==m]
        ar=[k for k in all_remaining if k[0]==m]
        wr.append(dict(model=m,cell_unique_candidates=sum(cellmap[cid]['model']==m for cid,h in unique_cell),
           gross_cell_items=sum(cellmap[k[0]]['model']==m for k in required),
           endpoint_reused_cell_items=sum(cellmap[k[0]]['model']==m for k in existing),
           pending_cell_items=len(pp),pending_cell_votes=len(pp)*3,
           unique_request_items_after_cross_cell_endpoint_screen=len(pr),
           unique_request_items_after_validation_screen=len(ar),
           conservative_cell_inference_hours=sum(st.median(timing[m,h]) for cid,h,i in pp)/3600,
           cross_cell_screen_inference_hours=sum(st.median(timing[m,h]) for _,h,i in pr)/3600,
           validation_screen_inference_hours=sum(st.median(timing[m,h]) for _,h,i in ar)/3600))
    csvout('source_test_workload.csv',wr)

    # Candidate format screening does NOT classify semantic defects.
    screens=[]
    for c in candidates:
        t=c['prompt_text']
        heading=re.findall(r'(?im)^\s*(?:\d+[.)]\s*)?(?:\*\*|#{1,4}\s*)?(Non[- ]security|Security)(?: requirements?)?(?:\*\*)?\s*:',t)
        extra=bool(re.search(r'(?i)here (?:are|is)|revised definitions|enhanced definitions|these (?:revised |enhanced )?definitions|the (?:revised|enhanced) definitions|changes (?:made|include)',t))
        counts=co.Counter(x.lower().replace(' ','-') for x in heading)
        if extra or any(v>1 for v in counts.values()):
            screens.append(dict(cell_id=c['cell_id'],iteration=c['iteration'],prompt_sha256=c['prompt_sha256'],
                 model=c['model'],optimizer=c['optimizer'],is_selected_best=c['is_selected_best'],
                 explanation_phrase_screen=extra,heading_counts=json.dumps(dict(counts)),prompt_text=t))
    csvout('candidate_format_screen.csv',screens)

    # Exact target vectors are used only for offline example extraction.
    with gzip.open(paths['vectors'],'rt') as f:
        header=json.loads(next(f)); routes=[json.loads(x) for x in f]
    routemap={(r['model'],r['prompt_sha256']):r for r in routes}
    items={r['gitreq_id']:r for r in csv.DictReader(paths['gitreq'].open())}
    for r in routes:
        assert len(r['prediction'])==6302 and len(r['votes'])==3
        assert all(len(v)==6302 and set(v)<={'0','1'} for v in r['votes'])
        majority=''.join('1' if sum(x=='1' for x in vv)>=2 else '0' for vv in zip(*r['votes']))
        assert majority==r['prediction']
    assert sha(paths['gitreq'].read_bytes())=='d765fb052feccb0fc0d031595b855834637d03662d798c090c309e32b6b772e7'
    assert len(items)==6302 and set(items)==set(header['gitreq_ids'])
    assert all(items[i]['binary_label']==('Security' if g=='1' else 'Non-security') for i,g in zip(header['gitreq_ids'],header['gold']))
    cases=[('qwen2_7b','btape',42),('llama3_8b','btape',44),('qwen2_7b','pe2',41),('qwen2_7b','pe2',43)]
    case_rows=[]; case_summary=[]
    for model,opt,seed in cases:
        cid=f'{model}__{opt}__seed-{seed}'; cs=bycell[cid]
        p0=next(x for x in cs if x['iteration']==0); ps=next(x for x in cs if x['is_selected_best'])
        unique={x['prompt_sha256']:x for x in cs}
        best=max(unique.values(),key=lambda x:(float(scoremap[cid,x['prompt_sha256'],'T_All']['target_macro_f1']),x['iteration']))
        for name,p in [('p0',p0),('p_source',ps),('pool_best',best)]:
            (OUT/'prompts'/f'{cid}__{name}.txt').write_text(p['prompt_text'],encoding='utf-8')
        for name,a,bp in [('p0_to_source',p0,ps),('source_to_pool_best',ps,best)]:
            diff=''.join(difflib.unified_diff(a['prompt_text'].splitlines(True),bp['prompt_text'].splitlines(True),fromfile=a['prompt_id'],tofile=bp['prompt_id']))
            (OUT/'prompts'/f'{cid}__{name}.diff').write_text(diff)
            ra=routemap[model,a['prompt_sha256']]; rb=routemap[model,bp['prompt_sha256']]
            pa=ra['prediction']; pb=rb['prediction']; gold=header['gold']; counts=co.Counter()
            for i,item_id in enumerate(header['gitreq_ids']):
                if pa[i]==pb[i]: continue
                transition='Fixed' if pb[i]==gold[i] else 'Broken'
                counts[transition]+=1
                d=items[item_id]
                case_rows.append(dict(cell_id=cid,contrast=name,requirement_id=item_id,category=d['original_label'],
                    gold_binary=d['binary_label'],before_prediction='Security' if pa[i]=='1' else 'Non-security',
                    after_prediction='Security' if pb[i]=='1' else 'Non-security',transition=transition,
                    requirement_text=d['requirement_text'],repository=d['repository'],issue_url=d['url'],
                    before_hash=a['prompt_sha256'],after_hash=bp['prompt_sha256'],
                    before_route=ra['route_id'],after_route=rb['route_id']))
            before=f1(gold,pa); after=f1(gold,pb)
            assert abs(before-float(scoremap[cid,a['prompt_sha256'],'T_All']['target_macro_f1']))<EPS
            assert abs(after-float(scoremap[cid,bp['prompt_sha256'],'T_All']['target_macro_f1']))<EPS
            case_summary.append(dict(cell_id=cid,contrast=name,n_unique=len(unique),before_iteration=a['iteration'],
                after_iteration=bp['iteration'],before_macro=before,after_macro=after,delta_pp=100*(after-before),
                fixed=counts['Fixed'],broken=counts['Broken'],net=counts['Fixed']-counts['Broken']))
    csvout('case_all_changed_items.csv',case_rows);csvout('case_summary.csv',case_summary)
    display=[('qwen2_7b__btape__seed-42','p0_to_source','NFR-000072'),
             ('qwen2_7b__btape__seed-42','p0_to_source','NFR-000003'),
             ('llama3_8b__btape__seed-44','p0_to_source','NFR-000056'),
             ('llama3_8b__btape__seed-44','p0_to_source','NFR-000015')]
    for seed in [41,43]:
        display += [(f'qwen2_7b__pe2__seed-{seed}','source_to_pool_best',i) for i in ['NFR-000194','NFR-000017']]
    displayed=[next(r for r in case_rows if (r['cell_id'],r['contrast'],r['requirement_id'])==key) for key in display]
    csvout('case_display_examples.csv',displayed)
    (OUT/'prompts/initial_definition.txt').write_text(candidates[0]['prompt_text'])
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
                  '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','—':'---','–':'--'}
    escaped=''.join(replacements.get(ch,ch) for ch in candidates[0]['prompt_text'])
    (ROOT/'tables/initial_definition.tex').write_text(escaped+'\n')
    overall={}
    for view in ['T_All','T_NFR']:
        rs=[r for r in policies if r['view']==view]; ds=[r['source_minus_last_pp'] for r in rs]
        overall[view]=dict(win=sum(d>EPS for d in ds),tie=sum(abs(d)<=EPS for d in ds),loss=sum(d<-EPS for d in ds),
            mean_pp=st.mean(ds),median_pp=st.median(ds),
            top3_gaps=[{k:r[k] for k in ['cell_id','n_unique','pool_gap_pp']} for r in sorted(rs,key=lambda r:-r['pool_gap_pp'])[:3]])
    # Historical route times are additional observed wall-time evidence, not promises.
    times=[]
    for m in MODELS:
        ts=[r['marker'].get('elapsed_seconds') for r in routes if r['model']==m]
        ts=[x for x in ts if isinstance(x,(float,int)) and x>0]
        times.append(dict(model=m,n_routes_with_elapsed=len(ts),median_hours_per_6302=st.median(ts)/3600,
                          min_hours=min(ts)/3600,max_hours=max(ts)/3600))
    csvout('target_route_timing_reference.csv',times)
    result=dict(analysis_date='2026-10-22',post_hoc=True,new_model_calls=0,
        inputs={k:{'path':str(p),'sha256':sha(p.read_bytes())} for k,p in paths.items()},
        candidates=len(candidates),cells=len(cells),unique_cell_candidates=len(unique_cell),
        historical_selection_unchanged=True,source_metrics_not_used_for_new_selection=True,
        last_proposal_comparison=overall,
        source_test_workload=dict(gross_cell_items=len(required),gross_votes=len(required)*3,
            existing_endpoint_cell_items=len(existing),remaining_cell_items=len(pending),remaining_cell_votes=len(pending)*3,
            required_cross_cell_request_items=len(reqroutes),pending_after_cross_cell_endpoint_screen=len(pendingroutes),
            pending_after_validation_screen=len(all_remaining),prediction_conflicts=len(conflicts),
            caveat='Cross-cell/validation reuse is a count screen only; full input/runtime fingerprint audit is required before inference reuse.',
            timing_basis='Median historical validation elapsed seconds per three-vote item for each model-prompt. Excludes startup, I/O and queue; not a wall-time guarantee.'),
        format_screen=dict(flagged_occurrences=len(screens),flagged_cell_texts=len({(r['cell_id'],r['prompt_sha256']) for r in screens}),
             parse_invalid_events=sum(e['proposal_parse_valid'] is not True for e in events),
             caveat='Regex screening only, not a semantic defect prevalence estimate. See frozen parser source.'),
        tests={'all_counts_match':True,'exact_prompt_hashes_match':True,'selected_iteration_matches':True,
               'case_scores_rebuilt_from_predictions':True,'all_votes_have_valid_shape':True,
               'target_majorities_rebuilt':True,'dataset_hash_and_id_gold_mapping_match':True})
    (OUT/'review_analysis_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'policy':overall,'workload':wr,'cases':case_summary,'format':result['format_screen']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
