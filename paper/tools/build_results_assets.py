"""Regenerate manuscript assets from copied, audited summary tables only.

No model calls; no changes to experiment files. The statistical analyses and
their conditional bootstrap intervals are already frozen in the input CSVs.
Run: uv run --isolated --no-project --with pandas --with matplotlib --with numpy
     python tools/build_results_assets.py
Use --only rq2_rank_transfer rq2_gap_range for a narrow figure-only rebuild.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import orx_figstyle as style
from build_candidate_diversity_table import render as render_diversity_table

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', nargs='+', choices=['rq1_transfer_gain', 'rq1_security_behavior',
                    'rq2_rank_transfer', 'rq2_gap_range', 'rq2_target_selection'],
                    help='Write only the named figures; leave tables and full-build checks unchanged.')
ONLY = set(parser.parse_args().only or [])
NAMES = {'llama3_8b': 'Llama', 'qwen2_7b': 'Qwen2', 'granite3_2_8b': 'Granite'}
GROUPS = [(m, o) for m in NAMES for o in ['pe2', 'btape']]
LABELS = [NAMES[m] + ' / ' + ('PE2' if o == 'pe2' else 'BT-APE') for m, o in GROUPS]
VIEWS = ['T_All', 'T_NFR']
DISPLAY = [r'$T_{\mathrm{All}}$', r'$T_{\mathrm{NFR}}$']
BLUE, RED, GRAY = style.PALETTE['blue'], style.PALETTE['red'], '#969696'
# sn-jnl.cls: text width 31 picas = 372 TeX points. No figure downscaling.
WIDTH = 372 / 72.27
style.TEXT = WIDTH
style.use_style()
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'savefig.dpi': 300})
G1 = pd.read_csv(ROOT / 'data/rq1/group_summary.csv')
C1 = pd.read_csv(ROOT / 'data/rq1/target_cells.csv')
C2 = pd.read_csv(ROOT / 'data/rq2/cell_summary.csv')
G2 = pd.read_csv(ROOT / 'data/rq2/group_summary_audited.csv')
K = pd.read_csv(ROOT / 'data/rq2/cell_rank_transfer.csv')
GAP = pd.read_csv(ROOT / 'data/rq2/cell_selection_gap.csv')
SENS = pd.read_csv(ROOT / 'data/rq2/sensitivity_overview.csv')
AUDITS = {}
PLACEMENTS = []


def panels(rows=1):
    """Physical-point layout: six groups, one summary + five distinct lanes.

    Data x coordinates are untouched. At native print size adjacent split
    centers are 4.6 pt apart, exceeding the 4.39 pt circle/triangle extent.
    """
    body, top, bottom, gap = 186, 42, 43, 50
    height = body * rows + top + bottom + gap * (rows - 1)
    fig = plt.figure(figsize=(WIDTH, height / 72))
    left, right, hgap = 83, 4, 17
    width = (WIDTH * 72 - left - right - hgap) / 2
    axes = []
    for row in range(rows):
        y = bottom + (rows - row - 1) * (body + gap)
        axes.append([fig.add_axes([(left + col * (width + hgap)) / (WIDTH * 72),
                                  y / height, width / (WIDTH * 72), body / height])
                     for col in range(2)])
    for pair in axes:
        pair[1].sharey(pair[0])
    return fig, np.array(axes) if rows > 1 else np.array(axes[0])


def summary_y(row):
    return row * 31 + 2


def split_y(row, k):
    return row * 31 + 8 + k * 4.6


def select(df, m, o, v):
    return df[(df.model == m) & (df.optimizer == o) & (df.view == v)]


def group_axes(ax, xlabel, labels=True):
    for i in range(0, 6, 2):
        ax.axhspan(i*31-1, i*31+29, color='#F5F5F5', zorder=0)
    ax.set_yticks([i * 31 + 15 for i in range(6)], LABELS)
    ax.tick_params(axis='y', labelleft=labels)
    if labels:
        ax.set_ylabel('Model / optimizer')
    ax.set_ylim(186, -1)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', visible=True)


def export(fig, name):
    if ONLY and name not in ONLY:
        plt.close(fig)
        return
    issues = style._audit(fig)
    # Verify displayed centers in physical points, not raster screenshot pixels.
    fig.canvas.draw()
    for ax in fig.axes:
        runs = [c for c in ax.collections if c.get_gid() == 'source-split']
        points = [ax.transData.transform(c.get_offsets()[0]) * 72 / fig.dpi for c in runs]
        if points:
            assert len(points) == 30, (name, len(points))
            minimum = min(np.linalg.norm(a-b) for i, a in enumerate(points) for b in points[i+1:])
            assert minimum >= np.sqrt(14)+.65, (name, minimum)
        for c in runs:
            PLACEMENTS.append({'figure': name, 'panel': fig.axes.index(ax),
                               'x': float(c.get_offsets()[0][0]), 'y': float(c.get_offsets()[0][1])})
    AUDITS[name] = issues
    if issues:
        raise AssertionError((name, issues))
    style.save(fig, str(ROOT / 'figures' / name), formats=('pdf', 'svg', 'png'))


def run_points(ax, values, row, low=None, color=BLUE):
    for k, val in enumerate(values):
        lr = bool(low[k]) if low is not None else False
        ax.scatter(val, split_y(row, k), s=14, marker='^' if lr else 'o',
                   facecolor='white' if lr else color, edgecolor=color, linewidth=.65, zorder=4,
                   gid='source-split')


def rank_legend(fig, ci=True):
    h = [Line2D([], [], marker='o', color=BLUE, ls='', markersize=4, label='Source split'),
         Line2D([], [], marker='^', color=BLUE, mfc='white', ls='', markersize=4, label='2–4 texts'),
         Line2D([], [], marker='D', color='black', ls='', markersize=4, label='Median (5 splits)')]
    if ci:
        h.append(Line2D([], [], color=GRAY, lw=.7, label='Per-split 95% CI'))
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(.53,1.01), ncol=2)


# Fig. 2: seed variation and uncertainty in a fixed-run mean are distinct.
fig, axes = panels()
for j, (ax, v) in enumerate(zip(axes, VIEWS)):
    for i, (m, o) in enumerate(GROUPS):
        cells = select(C1, m, o, v).sort_values('seed')
        g = select(G1, m, o, v).iloc[0]
        run_points(ax, cells.delta_macro_f1.to_numpy() * 100, i)
        ax.plot([g.macro_f1_ci_low_pp, g.macro_f1_ci_high_pp], [summary_y(i)]*2, color='black', lw=1.2, zorder=5)
        ax.scatter(g.gain_pp, summary_y(i), marker='D', color='black', s=16, zorder=6)
    group_axes(ax, 'Macro-F1 gain (pp)\n' + DISPLAY[j], labels=j == 0)
    ax.axvline(0, color='.4', ls='--', lw=.7)
    ax.set_xlim(-3.5, 15.6); ax.set_xticks([0, 5, 10, 15])
fig.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', markersize=4, label='Source split'),
                    Line2D([], [], marker='D', color='black', lw=1.2, markersize=4, label='Mean + conditional 95% CI')],
           loc='upper center', bbox_to_anchor=(.53,1), ncol=1)
style.panel_labels(axes); export(fig, 'rq1_transfer_gain')

# Fig. 3: behavior complements aggregate Macro-F1, not a second score table.
fig, axes = panels()
for j, (metric, label) in enumerate([('security_recall', 'Security recall change (pp)'),
                                    ('security_precision', 'Security precision change (pp)')]):
    ax = axes[j]
    for i, (m, o) in enumerate(GROUPS):
        c = select(C1, m, o, 'T_All').sort_values('seed')
        g = select(G1, m, o, 'T_All').iloc[0]
        run_points(ax, c['delta_' + metric].to_numpy() * 100, i)
        ax.plot([g[metric + '_ci_low_pp'], g[metric + '_ci_high_pp']], [summary_y(i)]*2, c='black', lw=1.2, zorder=5)
        ax.scatter(g[metric + '_gain_pp'], summary_y(i), c='black', s=16, marker='D', zorder=6)
    group_axes(ax, label, labels=j == 0); ax.axvline(0, color='.4', ls='--', lw=.7)
axes[0].set_xlim(-12, 39); axes[0].set_xticks([-10, 0, 10, 20, 30])
axes[1].set_xlim(-11, 8); axes[1].set_xticks([-10, -5, 0, 5])
fig.legend(handles=[Line2D([], [], marker='o', color=BLUE, ls='', markersize=4, label='Source split'),
                    Line2D([], [], marker='D', color='black', lw=1.2, markersize=4, label='Mean + conditional 95% CI')],
           loc='upper center', bbox_to_anchor=(.53,1), ncol=1)
style.panel_labels(axes); export(fig, 'rq1_security_behavior')

# Fig. 4: all 30 cells in both primary views; no favorable-example filtering.
fig, axes = panels()
for j, (ax, v) in enumerate(zip(axes, VIEWS)):
    for i, (m, o) in enumerate(GROUPS):
        d = select(K, m, o, v)
        d = d[d.metric_pair == 'primary_weighted_to_macro'].sort_values('seed')
        for k, r in enumerate(d.itertuples()):
            y = split_y(i,k)
            ax.plot([r.kendall_tau_b_ci95_low, r.kendall_tau_b_ci95_high], [y,y], c=GRAY, lw=.65)
        run_points(ax, d.kendall_tau_b.to_numpy(), i, d.low_resolution.to_numpy())
        ax.scatter(d.kendall_tau_b.median(), summary_y(i), c='black', marker='D', s=16, zorder=6)
    group_axes(ax, r'Kendall $\tau_b$' + '\n' + DISPLAY[j], labels=j == 0)
    ax.axvline(0, c='.4', ls='--', lw=.7); ax.set_xlim(-1.1, 1.1); ax.set_xticks([-1, -.5, 0, .5, 1])
rank_legend(fig); style.panel_labels(axes); export(fig, 'rq2_rank_transfer')

# Fig. 5: same cells, but absolute gap and full score range use different scales.
fig, axes = panels(rows=2)
for row, metric in enumerate(['selection_gap_pp', 'target_range_pp']):
    for j, v in enumerate(VIEWS):
        ax = axes[row, j]
        for i, (m, o) in enumerate(GROUPS):
            d = select(C2, m, o, v).sort_values('seed')
            for k, r in enumerate(d.itertuples()):
                ci = GAP[(GAP.cell_id == r.cell_id) & (GAP.view == v)].iloc[0]
                prefix = 'selection_gap' if row == 0 else 'target_range'
                y = split_y(i,k)
                ax.plot([ci[prefix+'_ci95_low'], ci[prefix+'_ci95_high']], [y,y], c=GRAY, lw=.65)
            run_points(ax, d[metric].to_numpy(), i, d.low_resolution.to_numpy())
            ax.scatter(d[metric].median(), summary_y(i), c='black', marker='D', s=16, zorder=6)
        group_axes(ax, ('Selection gap (pp)' if row == 0 else 'Target score range (pp)') + '\n' + DISPLAY[j], j == 0)
        if row == 0:
            ax.axvline(1, c='.4', ls='--', lw=.7)
            ax.set_xlim(-.15, 5.65); ax.set_xticks([0,1,2,3,4,5])
        else:
            ax.set_xlim(-1, 46); ax.set_xticks([0,10,20,30,40])
rank_legend(fig); style.panel_labels(axes.ravel()); export(fig, 'rq2_gap_range')

# Fig. 6: held-out, target-supervised comparator; no invented fold-level CI.
fig, axes = panels()
for j, (ax, v) in enumerate(zip(axes, VIEWS)):
    for i, (m, o) in enumerate(GROUPS):
        d = select(C2, m, o, v).sort_values('seed')
        vals = d.pooled_oof_target_selected_minus_p_source_pp.to_numpy()
        run_points(ax, vals, i, d.low_resolution.to_numpy())
        ax.scatter(np.median(vals), summary_y(i), c='black', marker='D', s=16, zorder=6)
    group_axes(ax, 'OOF advantage (pp)\n' + DISPLAY[j], j == 0)
    ax.axvline(0, c='.4', ls='--', lw=.7); ax.set_xlim(-1, 4.6); ax.set_xticks([0,1,2,3,4])
rank_legend(fig, ci=False); style.panel_labels(axes); export(fig, 'rq2_target_selection')


def table_file(name, spec, header, rows):
    if ONLY:
        return
    lines = [r'\begin{tabular}{' + spec + '}', r'\toprule', header + r' \\', r'\midrule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')


rows = []
for m, o in GROUPS:
    a, n = [select(G1,m,o,v).iloc[0] for v in VIEWS]
    rows.append([NAMES[m], 'PE2' if o=='pe2' else 'BT-APE',
                 f'{a.source_test_gain_macro_pp:+.2f}',
                 f'{100*a.p0_macro_f1:.2f}', f'{100*a.pstar_macro_f1_mean:.2f}',
                 f'{100*n.p0_macro_f1:.2f}', f'{100*n.pstar_macro_f1_mean:.2f}'])
table_file('rq1_endpoints.tex', 'llrrrrr',
           r'Origin & Optimizer & Source $\Delta$ & \multicolumn{2}{c}{$T_{\mathrm{All}}$} & \multicolumn{2}{c}{$T_{\mathrm{NFR}}$} \\'+ '\n' +
           r' & & (pp) & $p_0$ & $p^*$ & $p_0$ & $p^*$', rows)

if not ONLY:
    (ROOT/'tables/rq2_diversity.tex').write_text(render_diversity_table(ROOT))

rows=[]
for label,kind in [('Unique candidates (primary)','primary'),('Source macro-F1 ranks','macro'),
                   (r'Only cells with $n\geq5$','n5'),('All proposal occurrences','occurrence_preserving'),
                   ('Generated-only candidates','generated_only'),('No standalone security word','label_word_free')]:
    vals=[]
    for v in VIEWS:
        if kind in ['primary','macro','n5']:
            q=K[(K.view==v)&(K.metric_pair==('aligned_macro_to_macro' if kind=='macro' else 'primary_weighted_to_macro'))]
            if kind=='n5':q=q[q.n_unique>=5]
            median=q.kendall_tau_b.median();count=q.kendall_tau_b.count()
        else:
            vv=v if kind!='label_word_free' else ('T_NoSecurityWord_All' if v=='T_All' else 'T_NoSecurityWord_NFR')
            q=SENS[(SENS.analysis==kind)&(SENS.view==vv)&(SENS.metric_pair=='primary_weighted_to_macro')].iloc[0]
            median=q.tau_median;count=q.valid_tau
        vals += [f'{median:.3f}', str(int(count))]
    rows.append([label]+vals)
table_file('rq2_sensitivity.tex','lrrrr',
           r'Ranking analysis & \multicolumn{2}{c}{$T_{\mathrm{All}}$} & \multicolumn{2}{c}{$T_{\mathrm{NFR}}$} \\'+ '\n' +
           r' & Median $\tau_b$ & Cells & Median $\tau_b$ & Cells',rows)

# Automated cross-RQ and narrative invariants. Exact values stay in the CSVs.
assert len(C2[C2.view=='T_All']) == 30
assert len(C1[C1.view=='T_All']) == 30
for v in VIEWS:
    c1=C1[C1.view==v].set_index('cell_id'); c2=C2[C2.view==v].set_index('cell_id')
    assert set(c1.index)==set(c2.index)
    assert np.allclose(c1.loc[c2.index,'pstar_macro_f1'], c2.p_source_macro, atol=1e-12, rtol=0)
    assert (c2.selection_gap_pp>=-1e-12).all()
    assert not c2.posthoc_macro_selector_changed.any()
    assert c2.n_unique.sum()==380
    assert (c2.low_resolution).sum()==6
a=C2[C2.view=='T_All'];n=C2[C2.view=='T_NFR']
assert (a.tau>0).sum()==24 and (n.tau>0).sum()==23
assert a.p_source_is_target_top_tie.sum()==10 and n.p_source_is_target_top_tie.sum()==9
assert (a.selection_gap_pp<=1+1e-12).sum()==21
assert (n.selection_gap_pp<=1+1e-12).sum()==21
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'data').rglob('*.csv'))}
checks={'new_inference_calls':0,'experiment_files_modified':False,'cross_rq_endpoint_agreement':True,
        'historical_source_selection_unchanged':True,'figure_width_inches':WIDTH,
        'font_sizes_pt':{'tick_min':8,'label':8,'legend':8},'figure_audits':AUDITS,
        'input_csv_sha256':hashes,'python_plotting_versions':{'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':mpl.__version__}}
if ONLY:
    (ROOT/'research/figure_readability_rq12_audit.json').write_text(json.dumps(
        {'figure_audits':AUDITS, 'placements':PLACEMENTS, 'input_csv_sha256':hashes,
         'x_jitter':False,'split_spacing_pt':4.6 * 186 / 187,
         'circle_outer_diameter_pt':float(np.sqrt(14)+.65)},indent=2)+'\n')
    print('Figure-only rebuild completed:', ', '.join(sorted(AUDITS)))
    print('Tables, other figure files, and the full-build audit file were not written.')
else:
    (ROOT/'research/results_assets_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('Five audited vector figures, three data-generated tables, and cross-RQ checks completed.')
