"""Render frozen RQ2 group counts with separate columns for the two views.

Standard library only; no scoring, resampling, or model calls. With --check,
verify the saved table without writing it. Every count has denominator five.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'llama3_8b': 'Llama', 'qwen2_7b': 'Qwen2', 'granite3_2_8b': 'Granite'}


def render(root=ROOT):
    with (root / 'data/rq2/group_summary_audited.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['model'], r['optimizer'], r['view']): r for r in rows}
    assert len(lookup) == len(rows), 'Duplicate summary keys'
    lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Origin & Optimizer & Unique $n$ & $n<5$ & \multicolumn{2}{c}{Target top} & \multicolumn{2}{c}{Gap $\leq1$ pp} \\',
             r'\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r' & & range & cells & All & NFR & All & NFR \\', r'\midrule']
    for model, name in NAMES.items():
        for optimizer in ['pe2', 'btape']:
            a, n = [lookup[model, optimizer, view] for view in ['T_All', 'T_NFR']]
            assert a['n_cells'] == n['n_cells'] == '5'
            for field in ['n_unique_min', 'n_unique_max', 'low_resolution_cells']:
                assert a[field] == n[field], (model, optimizer, field)
            counts = [a['low_resolution_cells'], a['target_top_tie_cells'],
                      n['target_top_tie_cells'], a['gap_le_1pp_cells'], n['gap_le_1pp_cells']]
            assert all(0 <= int(value) <= 5 for value in counts)
            cells = [name, 'PE2' if optimizer == 'pe2' else 'BT-APE',
                     a['n_unique_min'] + '--' + a['n_unique_max']] + counts
            lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'tables/rq2_diversity.tex'
    expected = render()
    if args.check:
        assert output.read_text() == expected, 'Saved diversity table differs from frozen CSV'
    else:
        output.write_text(expected)
    print('RQ2 diversity table: six groups; separate All/NFR columns; all counts out of five.')


if __name__ == '__main__':
    main()
