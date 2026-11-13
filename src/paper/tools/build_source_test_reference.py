"""Rebuild the post-hoc source-test table from frozen CSVs; standard library only."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/source-test-supplement'


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def median(rs,key):
    v=[float(r[key]) for r in rs if r[key]!='']
    assert v, 'No computable observations'
    return st.median(v)


def main():
    complete=json.loads((DATA/'experiment_complete.json').read_text())
    hashes={}
    for name in ('candidate_scores.csv','cell_rank_reference.csv','cell_selection_gaps.csv'):
        expected=complete['analysis_hashes'][name]
        hashes[name]=hashlib.sha256((DATA/'analysis'/name).read_bytes()).hexdigest()
        assert hashes[name]==expected, f'Changed source-test table: {name}'
    cs=rows(DATA/'analysis/candidate_scores.csv')
    rank=rows(DATA/'analysis/cell_rank_reference.csv')
    gap=rows(DATA/'analysis/cell_selection_gaps.csv')
    assert (len(cs),len(rank),len(gap))==(760,300,180)
    assert len({(r['cell_id'],r['prompt_sha256']) for r in cs})==380
    assert len({r['cell_id'] for r in cs})==30
    # Recheck each target score and selected/p0 flag against the independently
    # audited RQ2 inputs already present in the manuscript before this revision.
    old={ (r['cell_id'],r['prompt_sha256'],r['view']):r
          for r in rows((ROOT.parent if (ROOT.parent/'analysis').is_dir() else ROOT.parents[1])
                        /'analysis/rq2/data/original_tables/candidate_scores.csv')
          if r['candidate_set_type']=='unique_all' and r['view'] in ('T_All','T_NFR') }
    assert len(old)==760
    for r in cs:
        original=old[r['cell_id'],r['prompt_sha256'],r['view']]
        assert r['is_selected_best']==original['is_p_source']
        assert r['is_p0']==original['is_p0']
        for metric in ('weighted_f1','macro_f1','accuracy'):
            assert abs(float(r['target_'+metric])-float(original['target_'+metric]))<=1e-12
            assert abs(float(r['validation_'+metric])-float(original['source_validation_'+metric]))<=1e-12
    # Existing RQ1 source-test endpoint effects must also remain unchanged.
    for oldrow in rows(ROOT/'data/rq1/source_target_pairs.csv'):
        if oldrow['view']!='T_All':continue
        subset=[r for r in cs if r['cell_id']==oldrow['cell_id'] and r['view']=='T_All']
        p0=next(r for r in subset if r['is_p0']=='True')
        ps=next(r for r in subset if r['is_selected_best']=='True')
        assert ps['prompt_sha256']==oldrow['pstar_hash']
        delta=float(ps['source_test_macro_f1'])-float(p0['source_test_macro_f1'])
        assert abs(delta-float(oldrow['test_delta_macro_f1']))<=1e-12
    pairs=('validation_macro_f1 -> source_test_macro_f1',
           'source_test_macro_f1 -> target_macro_f1')
    body=[r'\begin{tabular}{lrrrr}',r'\toprule',
          r' & \multicolumn{2}{c}{Median Kendall $\tau_b$} & \multicolumn{2}{c}{Median gap (pp)} \\',
          r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
          r'Group & Val--test & Test--target & Source test & Target \\',r'\midrule']
    summaries=[]
    for model,label in [('llama3_8b','Llama'),('qwen2_7b','Qwen2'),('granite3_2_8b','Granite')]:
        for opt,olabel in [('pe2','PE2'),('btape','BT-APE')]:
            rr=[r for r in rank if r['model']==model and r['optimizer']==opt and r['view']=='T_All']
            gg=[r for r in gap if r['model']==model and r['optimizer']==opt and r['view']=='T_All']
            vals=[median([r for r in rr if r['metric_pair']==pair],'tau_b') for pair in pairs]
            vals += [median([r for r in gg if r['metric']==metric],'selection_gap_pp') for metric in ('source_test_macro_f1','target_macro_f1')]
            assert all(len([r for r in rr if r['metric_pair']==p])==5 for p in pairs)
            body.append(f'{label}--{olabel} & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
            summaries.append({'model':model,'optimizer':opt,'medians':vals})
    vals=[median([r for r in rank if r['view']=='T_All' and r['metric_pair']==p],'tau_b') for p in pairs]
    vals += [median([r for r in gap if r['view']=='T_All' and r['metric']==m],'selection_gap_pp') for m in ('source_test_macro_f1','target_macro_f1')]
    body += [r'\midrule','All 30 cells & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\',r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables/rq2_source_test_reference.tex').write_text('\n'.join(body)+'\n')
    checks={'input_hashes_match_completed_run':True,'historical_target_scores_unchanged':True,
            'historical_source_validation_scores_unchanged':True,'historical_selected_flags_unchanged':True,
            'rq1_source_test_endpoint_effects_unchanged':True,'unique_cell_candidates':380,
            'cells':30,'views':['T_All','T_NFR'],'input_hashes':hashes,
            'group_medians':summaries,'all_cell_medians':vals,
            'new_inference_in_this_script':False,'no_new_intervals':True}
    (ROOT/'research/source_test_asset_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
