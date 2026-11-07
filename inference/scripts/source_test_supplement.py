"""Post-hoc frozen candidate source-test evaluation; no proposer is imported."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from pe2_btape.data import load_csv
from pe2_btape.metrics import compute_metrics
from pe2_btape.prompts import classification_messages, extract_label, majority
from run_target_matrix import canonical, create_backend, immutable_json, read_json, sha_file
from rq3_deployment import atomic, emit, fingerprint, immutable_jl, jl, now, save, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'configs/source_test_supplement_v1.json'
LABELS = ['Non-security', 'Security']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def config(path):
    cfg = read_json(path)
    base = read_json(ROOT / cfg['base_spec'])
    spec = {**base, **cfg}
    spec['models'] = {m: base['models'][m] for m in cfg['model_order']}
    require(spec['gpu'] == 0 and spec['votes_per_item'] == 3, 'GPU/vote drift')
    return spec


def metrics(rows):
    require(all(r['prediction'] in LABELS and r['gold'] in LABELS for r in rows), 'Invalid score input')
    return compute_metrics([r['prediction'] for r in rows], [r['gold'] for r in rows], LABELS).as_dict()


def compare_metrics(a, b):
    for key in ('weighted_f1', 'macro_f1', 'accuracy', 'balanced_accuracy', 'matthews_correlation'):
        require(abs(a[key] - b[key]) <= 1e-12, f'Metric mismatch {key}: {a[key]} vs {b[key]}')
    require(a['invalid_count'] == b['invalid_count'] == 0, 'Invalid predictions')


def split_ids(split):
    """RQ2's normalized export uses pool/val/test, full source uses *_ids."""
    output={}
    for name in ('pool','val','test'):
        long=name+'_ids'
        if long in split and name in split:
            require(split[long]==split[name], 'Conflicting split aliases')
        output[long]=split[long] if long in split else split[name]
    return output


def check_evaluation(evaluation, ids, examples):
    rows = evaluation['predictions']
    require([r['id'] for r in rows] == ids, 'Evaluation membership/order mismatch')
    for r in rows:
        require(r['gold'] == examples[r['id']].label, 'Gold mismatch')
        require(len(r['votes']) == len(r['raw_outputs']) == 3, 'Missing true votes')
        parsed = [extract_label(v, LABELS) for v in r['raw_outputs']]
        require(parsed == r['votes'] and all(v in LABELS for v in parsed), 'Raw/parsed mismatch')
        require(majority(parsed) == r['prediction'], 'Majority mismatch')
    compare_metrics(metrics(rows), evaluation['metrics'])


def audit(spec):
    out = Path(spec['output_root']); out.mkdir(parents=True, exist_ok=True)
    require(sha_file(Path(spec['source_archive'])) == spec['source_archive_sha256'], 'Archive hash mismatch')
    require(sha_file(Path(spec['source_csv'])) == spec['source_csv_sha256'], 'Source CSV hash mismatch')
    rq2 = Path(spec['rq2_root'])
    inputs = {name: sha_file(rq2 / name) for name in
              ('candidate_manifest.jsonl', 'unique_candidate_manifest.jsonl', 'cell_manifest.jsonl',
               'source_validation_predictions.jsonl', 'RQ2_results/tables/candidate_scores.csv')}
    code = {p: sha_file(ROOT / p) for p in ('scripts/source_test_supplement.py',
             'scripts/run_target_matrix.py', 'scripts/rq3_deployment.py',
             'src/pe2_btape/backend.py', 'src/pe2_btape/prompts.py',
             'src/pe2_btape/data.py', 'src/pe2_btape/metrics.py', 'SOURCE_TEST_SUPPLEMENT_PROTOCOL.md')}
    contract = {'spec': spec, 'code': code, 'rq2_inputs': inputs,
                'rq3_frozen_sha256': sha_file(Path(spec['rq3_frozen_prompts']))}
    immutable_json(out / 'frozen_config.json', contract)
    if not (out / 'SOURCE_TEST_SUPPLEMENT_PROTOCOL.md').exists():
        atomic(out / 'SOURCE_TEST_SUPPLEMENT_PROTOCOL.md', (ROOT / 'SOURCE_TEST_SUPPLEMENT_PROTOCOL.md').read_text())
    candidates = jl(rq2 / 'candidate_manifest.jsonl')
    unique = jl(rq2 / 'unique_candidate_manifest.jsonl')
    cells = jl(rq2 / 'cell_manifest.jsonl')
    require((len(cells), len(candidates), len(unique)) == (30, 630, 380), 'Manifest counts mismatch')
    require(len({x['candidate_id'] for x in candidates}) == 630, 'Duplicate occurrence ID')
    require(len({(x['cell_id'], x['prompt_sha256']) for x in unique}) == 380, 'Unique pool mismatch')
    for c in candidates:
        require(sha(c['prompt_text']) == c['prompt_sha256'] and c['demo_ids'] == [], 'Prompt/demo drift')
    oldhash = {x['prompt_id']: x['source_result_sha256'] for x in read_json(Path(spec['rq3_frozen_prompts']))['prompts']}
    examples = load_csv(spec['source_csv']); require(len(examples) == 510, 'Wrong source size')
    runtimes = {}; seed_splits = {}; routes = []; reuse_count = 0; evidence = []
    normalized_votes = defaultdict(dict)
    for row in jl(rq2 / 'source_validation_predictions.jsonl'):
        normalized_votes[row['candidate_id']][int(row['item_id'])] = row
    require(sum(map(len, normalized_votes.values())) == 96390, 'Validation export count')
    for cell in cells:
        cid, model = cell['cell_id'], cell['model']
        path = Path(spec['source_matrix']) / 'cells' / model / cell['optimizer'] / f"seed-{cell['seed']}" / 'result-full.json'
        fullhash = sha_file(path); require(fullhash == oldhash[cid], f'Changed frozen full result: {cid}')
        x = read_json(path); split = x['split']
        require(split == split_ids(cell['split']), 'Split export membership/order mismatch')
        require([len(split[k]) for k in ('pool_ids', 'val_ids', 'test_ids')] == [153,153,204], 'Split sizes')
        ids = sum((split[k] for k in ('pool_ids','val_ids','test_ids')), [])
        require(len(ids) == len(set(ids)) == 510 and set(ids) == set(range(510)), 'Split partition')
        require(seed_splits.setdefault(cell['seed'], split) == split, 'Same-seed split mismatch')
        runtime = x['provenance']['hf_vllm_runtime']
        require(fingerprint(runtimes.setdefault(model, runtime)) == fingerprint(runtime), 'Source runtime changed within model')
        require(x['config']['voting_runs'] == 3 and x['config']['classification_num_predict'] == 64, 'Source vote/length mismatch')
        require(x['config'].get('classification_interface','appendix_f') == spec['models'][model]['interface'], 'Wrapper mismatch')
        occurrences = sorted((c for c in candidates if c['cell_id'] == cid), key=lambda c:c['iteration'])
        require([c['iteration'] for c in occurrences] == list(range(21)), 'Missing iteration')
        evaluations = [x['initialization']['validation']] + [e['validation'] for e in x['iterations']]
        require(len(x['states']) == len(evaluations) == 21, 'States incomplete')
        for c, state, ev in zip(occurrences, x['states'], evaluations):
            require(c['prompt_text'] == state['definitions'] and state['demo_ids'] == [], 'State prompt changed')
            check_evaluation(ev, split['val_ids'], examples)
            compare_metrics(ev['metrics'], c['source_validation_metrics'])
            for r in ev['predictions']:
                n = normalized_votes[c['candidate_id']][r['id']]
                require(n['vote_labels'] == r['votes'] and n['prediction'] == r['prediction'], 'Validation export votes differ')
        best = max(x['states'], key=lambda s:(s['val_weighted_f1'], s['iteration']))
        require(best == x['best_state'], 'Historical newest-tie selection mismatch')
        require([c['iteration'] for c in occurrences if c['is_selected_best']] == [best['iteration']], 'Selected marker mismatch')
        endpoint_map = {}
        for name, state in [('p0', x['states'][0]), ('p_star', best)]:
            ev = x['source_tests'][name]; check_evaluation(ev, split['test_ids'], examples)
            ph = sha(state['definitions'])
            # Preserve both endpoint observations. Reuse only if their raw evidence agrees.
            if ph in endpoint_map:
                prev = endpoint_map[ph]['evaluation']['predictions']
                require([(r['votes'],r['raw_outputs']) for r in prev] ==
                        [(r['votes'],r['raw_outputs']) for r in ev['predictions']], 'Conflicting endpoint duplicates')
            endpoint_map[ph] = {'endpoint':name,'evaluation':ev,'source_result_path':str(path),'source_result_sha256':fullhash}
        immutable_json(out / 'endpoints' / f'{cid}.json', endpoint_map)
        local = [c for c in unique if c['cell_id'] == cid]
        require({c['prompt_sha256'] for c in local} == {c['prompt_sha256'] for c in occurrences}, 'Unique text membership')
        for c in local:
            occ = [a for a in occurrences if a['prompt_sha256'] == c['prompt_sha256']]
            require(c['occurrence_ids'] == [a['candidate_id'] for a in occ], 'Lineage mismatch')
            rep = next(a for a in occ if a['iteration'] == c['representative_iteration'])
            compare_metrics(rep['source_validation_metrics'], c['source_validation_metrics'])
            rc = {'cell_id':cid,'model':model,'prompt_sha256':c['prompt_sha256'],
                  'prompt_text':c['prompt_text'],'test_ids':split['test_ids'],
                  'source_csv_sha256':spec['source_csv_sha256'],'runtime':fingerprint(runtime),
                  'wrapper':spec['models'][model]['interface'],'votes':3,'generation_cap':64,
                  'temperature':0,'top_p':1,'top_k':-1,'config_sha256':sha(canonical(contract))}
            route = {**c, 'route_fingerprint':sha(canonical(rc)), 'test_ids':split['test_ids'],
                     'endpoint_reusable':c['prompt_sha256'] in endpoint_map}
            routes.append(route); reuse_count += 204 * route['endpoint_reusable']
        evidence.append({'cell_id':cid,'source_result_sha256':fullhash,'unique_candidates':len(local),
                         'source_result_path':str(path)})
        emit('SOURCE_TEST_CELL_AUDIT_PASS', cell=cid, unique=len(local))
    for name, rows in [('candidate_manifest.jsonl',candidates),('unique_candidate_manifest.jsonl',unique),
                       ('cell_manifest.jsonl',cells),('route_manifest.jsonl',routes)]:
        immutable_jl(out / name, rows)
    immutable_json(out / 'model_runtimes.json', runtimes)
    report = {'status':'PASS','cells':30,'occurrences':630,'unique_routes':380,
              'candidate_items':380*204,'endpoint_reuse_candidate_items':reuse_count,
              'new_candidate_items':380*204-reuse_count,'new_logical_votes':3*(380*204-reuse_count),
              'config_sha256':sha(canonical(contract)),'source_evidence':evidence,
              'limitations':['Post-hoc diagnostic; old p* unchanged','No cross-cell endpoint reuse',
                             'No causal separation of validation noise, adaptive search and corpus shift']}
    immutable_json(out / 'input_audit.json', report)
    immutable_json(out / 'input_audit_summary.json', {k:v for k,v in report.items() if k!='source_evidence'})
    atomic(out / 'input_audit.md', '# 输入审计：通过\n\n'+canonical({k:v for k,v in report.items() if k!='source_evidence'})+'\n')
    emit('SOURCE_TEST_AUDIT_PASS', **{k:v for k,v in report.items() if k!='source_evidence'})


def route_dir(spec, r):
    return Path(spec['output_root']) / 'predictions' / r['cell_id'] / r['prompt_sha256']


def contexts(spec, model):
    from transformers import AutoTokenizer
    from pe2_btape.backend import _local_hf_artifact_metadata
    out = Path(spec['output_root']); mod = spec['models'][model]
    runtime = read_json(out / 'model_runtimes.json')[model]
    observed = _local_hf_artifact_metadata(mod['model_dir'], mod['revision'])
    for k,v in fingerprint(runtime)['local_artifact'].items():
        require(observed[k] == v, f'Local model artifact drift: {model}/{k}')
    tok = AutoTokenizer.from_pretrained(mod['model_dir'], local_files_only=True, trust_remote_code=False)
    require(sha(tok.chat_template or '') == runtime['tokenizer_chat_template_sha256'], 'Tokenizer template drift')
    examples = load_csv(spec['source_csv']); reports=[]
    for r in jl(out / 'route_manifest.jsonl'):
        if r['model'] != model: continue
        rows=[]
        for i in r['test_ids']:
            system,user = classification_messages(r['prompt_text'], [], examples[i].text, LABELS, mod['interface'])
            messages=[{'role':'system','content':system},{'role':'user','content':user}]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tok.encode(rendered, add_special_tokens=False)
            require(len(tokens)+64 <= 2048, f'Context overflow: {r["cell_id"]}/{i}')
            rows.append({'item_id':i,'text_sha256':sha(examples[i].text),
                         'rendered_request_sha256':sha(rendered),'input_tokens':len(tokens)})
        value={'route_fingerprint':r['route_fingerprint'],'items':rows}
        path = route_dir(spec,r) / 'context.json'; immutable_json(path,value)
        reports.append({'cell_id':r['cell_id'],'prompt_sha256':r['prompt_sha256'],
                        'context_sha256':sha_file(path),'max_input_tokens':max(x['input_tokens'] for x in rows)})
    immutable_json(out / 'context' / f'{model}.json', reports)
    emit('SOURCE_TEST_CONTEXT_PASS',model=model,routes=len(reports),maximum=max(r['max_input_tokens'] for r in reports))


def append_vote(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(canonical(row)+'\n'); f.flush(); os.fsync(f.fileno())


def read_journal(path):
    """Only recover a torn final JSONL fragment, preserving the exact fragment."""
    if not path.exists(): return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary=raw.rfind(b'\n')+1; tail=raw[boundary:]
        try: json.loads(tail)
        except (ValueError,UnicodeDecodeError):
            evidence=path.with_name(path.name+'.torn-'+sha(tail.hex())[:16])
            if not evidence.exists():
                with evidence.open('xb') as f: f.write(tail);f.flush();os.fsync(f.fileno())
            with path.open('r+b') as f: f.truncate(boundary);f.flush();os.fsync(f.fileno())
        else:
            with path.open('ab') as f: f.write(b'\n');f.flush();os.fsync(f.fileno())
    return jl(path)


def check_vote(v, r, c):
    require(v['route_fingerprint']==r['route_fingerprint'], 'Cached route fingerprint mismatch')
    require(v['item_id']==c['item_id'] and v['text_sha256']==c['text_sha256'], 'Cached item mismatch')
    require(v['rendered_request_sha256']==c['rendered_request_sha256'], 'Rendered request mismatch')
    require(v['input_tokens']==c['input_tokens'], 'Token count mismatch')
    require(v['vote_index'] in (1,2,3), 'Wrong vote index')
    require(v['label'] in LABELS and extract_label(v['raw'],LABELS)==v['label'], 'Invalid vote')
    require(v['finish_reason']=='stop', 'Truncated/non-stop completion')


def load_votes(path, r, ctx):
    votes={}
    for v in read_journal(path):
        require(v['item_id'] in ctx, 'Unexpected cached item')
        check_vote(v,r,ctx[v['item_id']]); key=(v['item_id'],v['vote_index'])
        require(key not in votes, 'Duplicate saved vote'); votes[key]=v
    return votes


def validate_rows(rows,r,ctx,examples):
    require([x['item_id'] for x in rows]==r['test_ids'], 'Incomplete/duplicate output IDs')
    for x in rows:
        require(x['gold']==examples[x['item_id']].label, 'Final gold mismatch')
        require([v['vote_index'] for v in x['votes']]==[1,2,3], 'Missing output votes')
        for v in x['votes']: check_vote(v,r,ctx[x['item_id']])
        require(x['prediction']==majority([v['label'] for v in x['votes']]), 'Final majority mismatch')


def completed(spec,r,ctx,examples):
    d=route_dir(spec,r); marker=d/'complete.json'
    if not marker.exists(): return False
    m=read_json(marker)
    require(m['route_fingerprint']==r['route_fingerprint'], 'Changed completed route')
    require(m['prediction_sha256']==sha_file(d/'predictions.jsonl'), 'Completed file corrupted')
    rows=jl(d/'predictions.jsonl'); validate_rows(rows,r,ctx,examples)
    compare_metrics(metrics(rows),m['metrics']); return True


def finalize(spec,r,votes,ctx,examples,provenance):
    rows=[]
    for i in r['test_ids']:
        vs=[votes[i,k] for k in (1,2,3)]
        rows.append({'item_id':i,'gold':examples[i].label,'prediction':majority([v['label'] for v in vs]),'votes':vs})
    validate_rows(rows,r,ctx,examples)
    d=route_dir(spec,r); immutable_jl(d/'predictions.jsonl',rows)
    immutable_json(d/'complete.json',{'route_fingerprint':r['route_fingerprint'],
                   'prediction_sha256':sha_file(d/'predictions.jsonl'),'items':len(rows),
                   'votes':len(rows)*3,'metrics':metrics(rows),'provenance':provenance})


def reuse_endpoints(spec,model):
    out=Path(spec['output_root']); examples=load_csv(spec['source_csv']); reused=0
    for r in jl(out/'route_manifest.jsonl'):
        if r['model']!=model or not r['endpoint_reusable']: continue
        ctx={x['item_id']:x for x in read_json(route_dir(spec,r)/'context.json')['items']}
        if completed(spec,r,ctx,examples): reused+=1;continue
        ep=read_json(out/'endpoints'/f'{r["cell_id"]}.json')[r['prompt_sha256']]
        votes={}
        for row in ep['evaluation']['predictions']:
            i=row['id']; req=row['request']
            system,user=classification_messages(r['prompt_text'],[],examples[i].text,LABELS,spec['models'][model]['interface'])
            require((system,user)==(req['system'],req['user']), 'Historical classifier rendering drift')
            require(sha(req['hf_rendered_prompt'])==ctx[i]['rendered_request_sha256'], 'Historical tokenizer rendering drift')
            require(len(row['vote_metadata'])==3, 'Missing endpoint vote provenance')
            for k,meta in enumerate(row['vote_metadata'],1):
                v={**ctx[i], 'route_fingerprint':r['route_fingerprint'],'vote_index':k,
                   'raw':row['raw_outputs'][k-1],'label':row['votes'][k-1],
                   'finish_reason':meta['finish_reason'],'output_tokens':meta['eval_count'],
                   'usage':meta,'source':'frozen_source_endpoint','timestamp':None,
                   'source_result_sha256':ep['source_result_sha256']}
                require(meta['vote_index']==k and meta['batch_size']==1 and meta['prompt_eval_count']==ctx[i]['input_tokens'], 'Endpoint metadata mismatch')
                check_vote(v,r,ctx[i]);votes[i,k]=v
        finalize(spec,r,votes,ctx,examples,{k:v for k,v in ep.items() if k!='evaluation'})
        reused+=1
    emit('SOURCE_TEST_ENDPOINT_REUSE_PASS',model=model,routes=reused,items=reused*204)


def infer(spec,model):
    out=Path(spec['output_root']); examples=load_csv(spec['source_csv'])
    routes=[r for r in jl(out/'route_manifest.jsonl') if r['model']==model]
    todo=[]
    for r in routes:
        ctx={x['item_id']:x for x in read_json(route_dir(spec,r)/'context.json')['items']}
        if not completed(spec,r,ctx,examples): todo.append((r,ctx))
    if not todo: emit('SOURCE_TEST_MODEL_ALREADY_COMPLETE',model=model);return
    memory=int(subprocess.check_output(['nvidia-smi','-i',str(spec['gpu']),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    require(memory<1024, f'GPU0 occupied ({memory} MiB); not touching other jobs')
    backend=create_backend(spec,model,spec['gpu'])
    expected=read_json(out/'model_runtimes.json')[model]
    require(fingerprint(backend.runtime_metadata())==fingerprint(expected),'Live runtime differs from frozen source')
    immutable_json(out/'live_runtimes'/f'{model}.json',backend.runtime_metadata())
    started=time.monotonic(); new_votes=0
    for route_index,(r,ctx) in enumerate(todo):
        journal=route_dir(spec,r)/'votes.partial.jsonl'
        votes=load_votes(journal,r,ctx)
        for item_index,i in enumerate(r['test_ids']):
            system,user=classification_messages(r['prompt_text'],[],examples[i].text,LABELS,spec['models'][model]['interface'])
            for k in (1,2,3):
                if (i,k) in votes: continue
                if (out/'PAUSE').exists(): emit('SOURCE_TEST_PAUSED',model=model,cell=r['cell_id'],item=i,vote=k);return
                try:
                    raw,record=backend.chat(system,user,kind='classification:source_test_supplement',num_predict=64)
                    v={**ctx[i],'route_fingerprint':r['route_fingerprint'],'vote_index':k,
                       'raw':raw,'label':extract_label(raw,LABELS),'finish_reason':record['finish_reason'],
                       'input_tokens':record['prompt_eval_count'],'rendered_request_sha256':record['hf_rendered_prompt_sha256'],
                       'output_tokens':record['eval_count'],'timestamp':now(),'source':'new_inference',
                       'usage':{key:val for key,val in record.items() if key!='hf_rendered_prompt'}}
                    check_vote(v,r,ctx[i])
                except Exception as exc:
                    append_vote(out/'errors.jsonl',{'time':now(),'cell':r['cell_id'],'prompt':r['prompt_sha256'],
                        'item':i,'vote':k,'error':repr(exc),'response':locals().get('v')})
                    raise
                append_vote(journal,v);votes[i,k]=v;new_votes+=1
                backend._calls.clear()
            if item_index%25==0 or item_index==203:
                progress={'model':model,'pid':os.getpid(),'cell':r['cell_id'],'prompt':r['prompt_sha256'],
                          'route_index_this_session':route_index,'routes_pending_at_start':len(todo),
                          'route_items':item_index+1,'new_votes_this_session':new_votes,
                          'elapsed_seconds':time.monotonic()-started,'time':now()}
                save(out/'progress'/f'{model}.json',progress);emit('SOURCE_TEST_PROGRESS',**progress)
        finalize(spec,r,votes,ctx,examples,{'source':'new_inference','code_contract':read_json(out/'input_audit.json')['config_sha256']})
        emit('SOURCE_TEST_ROUTE_COMPLETE',model=model,cell=r['cell_id'],prompt=r['prompt_sha256'])
        backend._render_cache.clear()
    emit('SOURCE_TEST_MODEL_COMPLETE',model=model,new_votes=new_votes,seconds=time.monotonic()-started)


def tied_ranks(values, tolerance=1e-12):
    order=sorted(range(len(values)),key=lambda i:-values[i]); ranks=[0.0]*len(values);p=0
    while p<len(order):
        q=p+1
        while q<len(order) and abs(values[order[q]]-values[order[p]])<=tolerance:q+=1
        for idx in order[p:q]:ranks[idx]=(p+1+q)/2
        p=q
    return ranks


def correlations(a,b):
    x,y=tied_ranks(a),tied_ranks(b);n=len(x)
    if n<2 or len(set(x))<2 or len(set(y))<2:return {'rho':None,'tau_b':None,'reason':'constant_or_insufficient'}
    xm,ym=statistics.mean(x),statistics.mean(y)
    rho=sum((i-xm)*(j-ym) for i,j in zip(x,y))/math.sqrt(sum((i-xm)**2 for i in x)*sum((j-ym)**2 for j in y))
    c=d=tx=ty=0
    for i in range(n):
        for j in range(i):
            dx,dy=x[i]-x[j],y[i]-y[j]
            if dx==0 and dy==0:continue
            if dx==0:tx+=1
            elif dy==0:ty+=1
            elif dx*dy>0:c+=1
            else:d+=1
    tau=(c-d)/math.sqrt((c+d+tx)*(c+d+ty))
    return {'rho':rho,'tau_b':tau,'reason':None}


def csv_write(path, rows):
    if not rows:return
    import io
    f=io.StringIO(); w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);atomic(path,f.getvalue())


def analyze(spec):
    out=Path(spec['output_root']);examples=load_csv(spec['source_csv']);routes=jl(out/'route_manifest.jsonl')
    target={}
    with (Path(spec['rq2_root'])/'RQ2_results/tables/candidate_scores.csv').open() as f:
        for row in csv.DictReader(f):
            if row['candidate_set_type']=='unique_all' and row['view'] in ('T_All','T_NFR'):
                key=(row['cell_id'],row['prompt_sha256'],row['view']); require(key not in target,'Duplicate target score');target[key]=row
    scores=[];groups=defaultdict(list)
    for r in routes:
        ctx={x['item_id']:x for x in read_json(route_dir(spec,r)/'context.json')['items']}
        require(completed(spec,r,ctx,examples),'Cannot analyze incomplete experiment')
        test=read_json(route_dir(spec,r)/'complete.json')['metrics']
        for view in ('T_All','T_NFR'):
            t=target[r['cell_id'],r['prompt_sha256'],view]
            row={k:r[k] for k in ('cell_id','model','optimizer','seed','prompt_sha256','is_p0','is_selected_best','last_iteration')}
            row.update(view=view,occurrence_ids=canonical(r['occurrence_ids']))
            for metric in ('weighted_f1','macro_f1','accuracy'):
                row['validation_'+metric]=r['source_validation_metrics'][metric]
                row['source_test_'+metric]=test[metric];row['target_'+metric]=float(t['target_'+metric])
            scores.append(row);groups[r['cell_id'],view].append(row)
    ranks=[];gaps=[]
    pairs=[('validation_weighted_f1','source_test_weighted_f1'),('validation_macro_f1','source_test_macro_f1'),
           ('source_test_weighted_f1','target_weighted_f1'),('source_test_macro_f1','target_macro_f1'),
           ('validation_weighted_f1','target_macro_f1')]
    for (cid,view),rows in groups.items():
        base={k:rows[0][k] for k in ('cell_id','model','optimizer','seed','view')};base.update(n_unique=len(rows),low_resolution=len(rows)<5)
        selected=[r for r in rows if r['is_selected_best']];p0=[r for r in rows if r['is_p0']]
        require(len(selected)==len(p0)==1,'Endpoint flags ambiguous');s=selected[0];p=p0[0]
        for a,b in pairs:
            ranks.append({**base,'metric_pair':a+' -> '+b,**correlations([r[a] for r in rows],[r[b] for r in rows])})
        for metric in ('source_test_weighted_f1','source_test_macro_f1','target_macro_f1'):
            values=[r[metric] for r in rows];gap=max(values)-s[metric];span=max(values)-min(values)
            require(gap>=-1e-12,'Negative oracle gap')
            gaps.append({**base,'metric':metric,'p_source_hash':s['prompt_sha256'],'p0_score':p[metric],
                         'p_source_score':s[metric],'pool_best_score':max(values),'gain_over_p0_pp':100*(s[metric]-p[metric]),
                         'selection_gap_pp':100*gap,'target_or_test_range_pp':100*span,
                         'normalized_gap':gap/span if span>1e-12 else None,
                         'uniform_unique_score':statistics.mean(values),
                         'p_source_average_rank':tied_ranks(values)[rows.index(s)]})
    dst=out/'analysis';csv_write(dst/'candidate_scores.csv',scores);csv_write(dst/'cell_rank_reference.csv',ranks);csv_write(dst/'cell_selection_gaps.csv',gaps)
    summary=[]
    for model in spec['model_order']:
        for opt in ('pe2','btape'):
            for view in ('T_All','T_NFR'):
                for a,b in pairs:
                    rows=[r for r in ranks if r['model']==model and r['optimizer']==opt and r['view']==view and r['metric_pair']==a+' -> '+b]
                    valid=[r for r in rows if r['tau_b'] is not None]
                    summary.append({'model':model,'optimizer':opt,'view':view,'metric_pair':a+' -> '+b,
                         'n_cells':len(rows),'valid_n':len(valid),'low_resolution_n':sum(r['low_resolution'] for r in rows),
                         'median_tau_b':statistics.median(r['tau_b'] for r in valid) if valid else None,
                         'mean_tau_b':statistics.mean(r['tau_b'] for r in valid) if valid else None,
                         'min_tau_b':min((r['tau_b'] for r in valid),default=None),
                         'max_tau_b':max((r['tau_b'] for r in valid),default=None)})
    csv_write(dst/'group_summary.csv',summary)
    check={'complete_routes':len(routes),'candidate_items':len(routes)*204,'logical_votes':len(routes)*204*3,
           'candidate_score_rows':len(scores),'rank_rows':len(ranks),'gap_rows':len(gaps),
           'source_selection_changed':False,'status':'PASS'}
    save(dst/'analysis_checks.json',check)
    report=['# SecReq 全候选测试集补充诊断','',
            '这是事后同语料参照，不改变历史 p*，不覆盖 RQ1/RQ2/RQ3，也没有重新优化任何提示。',
            '全部 380 个 cell 内唯一候选已评估；630 条迭代记录保留在候选映射中。',
            '以下为描述性结果，没有新增置信区间；有限候选池、小测试集与自适应搜索仍限制解释。','',
            '|模型|优化器|视图|比较|有效 n|tau-b 中位数|','|---|---|---|---|---:|---:|']
    for r in summary:
        report.append('|'+ '|'.join(str(r[k]) for k in ('model','optimizer','view','metric_pair','valid_n','median_tau_b'))+'|')
    report += ['', '## 解释边界',
       '验证→测试提供 SecReq 内部排名参照；测试→GitReq 提供另一种跨语料比较。两者不是 domain shift 或验证集过拟合的因果分解。',
       '应联合检查排名、实际分数范围及固定源冠军的选择损失。低候选数或恒分须报告，不能视为可靠性证据。',
       '所有 oracle gap 均为同一评估样本上的候选池内描述量；不得据此替换历史 p*。']
    atomic(dst/'SOURCE_TEST_补充诊断报告.md','\n'.join(report)+'\n')
    immutable_json(out/'experiment_complete.json',{'checks':check,'analysis_hashes':{p.name:sha_file(p) for p in sorted(dst.glob('*.csv'))}})
    emit('SOURCE_TEST_SUPPLEMENT_COMPLETE',**check)


def status(spec):
    out=Path(spec['output_root']); result=[]
    for model in spec['model_order']:
        routes=[r for r in jl(out/'route_manifest.jsonl') if r['model']==model]
        done=sum((route_dir(spec,r)/'complete.json').exists() for r in routes)
        result.append({'model':model,'routes':len(routes),'completion_markers':done,
                       'progress':read_json(out/'progress'/f'{model}.json') if (out/'progress'/f'{model}.json').exists() else None})
    emit('SOURCE_TEST_STATUS',models=result,paused=(out/'PAUSE').exists())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['audit','plan','run','worker','status','analyze'])
    ap.add_argument('--config',type=Path,default=DEFAULT);ap.add_argument('--model');args=ap.parse_args();spec=config(args.config)
    if args.mode=='status':return status(spec)
    if args.mode=='analyze':return analyze(spec)
    if args.mode=='worker':return infer(spec,args.model)
    out=Path(spec['output_root']);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(out/'controller.json',{'pid':os.getpid(),'started':now(),'code_root':str(ROOT),'mode':args.mode})
        audit(spec)
        if args.mode in ('audit','plan'):return
        for model in spec['model_order']:
            contexts(spec,model);reuse_endpoints(spec,model)
        save(out/'preflight_report.json',{'status':'PASS','models':spec['model_order'],'gpu':0,'votes':3,'batch_size':1})
        for model in spec['model_order']:
            if (out/'PAUSE').exists():emit('SOURCE_TEST_PAUSED');return
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--config',str(args.config.resolve()),'--model',model],check=True)
        if not (out/'PAUSE').exists():analyze(spec)


if __name__=='__main__':main()
