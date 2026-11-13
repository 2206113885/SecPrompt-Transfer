"""Recompute the paper's analyses from frozen observations (CPU; no model calls)."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_files():
    entries = (ROOT / 'MANIFEST.sha256').read_text().splitlines()
    for line in entries:
        expected, name = line.split('  ', 1)
        assert sha((ROOT / name).read_bytes()) == expected, name
    return len(entries)


def stage(work):
    """Restore the original engines' input layout in disposable working space."""
    for name in ('analysis', 'paper'):
        shutil.copytree(ROOT / 'src' / name, work / name)
    for name in ('paper/tools', 'paper/figures', 'analysis/source_duplicates/figure'):
        shutil.copy2(ROOT / 'src/figure_style.py', work / name / 'orx_figstyle.py')
    for source in sorted((ROOT / 'data').rglob('*')):
        if not source.is_file() or source.name in ('reference.zip', 'README.md'):
            continue
        dest = work / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.xz':
            dest.with_suffix('').write_bytes(lzma.decompress(source.read_bytes()))
        elif source.name == 'predictions.npz' and source.parent.name != 'source_test':
            with np.load(source, allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
            shape = arrays['predictions'].shape
            for key in ('gold', 'item_ids'):
                assert arrays[key].shape == (shape[1],), (source, key)
                arrays[key] = np.broadcast_to(arrays[key], shape)
            np.savez_compressed(dest, **arrays)
        else:
            shutil.copy2(source, dest)
    with zipfile.ZipFile(ROOT / 'data/reference.zip') as z:
        for entry in json.loads(z.read('manifest.json')):
            relative = Path(entry['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            raw = z.read(entry['member'])
            assert sha(raw) == entry['sha256'], entry['path']
            dest = work / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(raw, mtime=0) if entry['encoding'] == 'gzip' else raw)


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def duplicate_check(work):
    out = work / 'reproduced/source_duplicates'
    run(work / 'analysis/source_duplicates/audit_duplicates.py',
        '--repository-root', work, '--output-dir', out)
    expected = json.loads((work / 'analysis/source_duplicates/audit-results.json').read_text())
    actual = json.loads((out / 'audit-results.json').read_text())
    compared = []
    for key in ('source', 'split_audit', 'source_test_sensitivity',
                'validation_sensitivity', 'checks', 'input_sha256'):
        assert actual[key] == expected[key], ('duplicate audit', key)
        compared.append(key)
    return compared


def compare_tables(work):
    checks = []
    for rq, folder in (('rq1', 'rq1'), ('rq2', 'rq2'), ('rq3', 'rq3-systematic')):
        count = 0
        for reference in sorted((work / 'paper/data' / folder).glob('*.csv')):
            actual = work / 'reproduced' / rq
            if rq != 'rq3':
                actual /= 'tables'
            actual /= reference.name
            if not actual.exists():
                continue
            left, right = pd.read_csv(reference), pd.read_csv(actual)
            pd.testing.assert_frame_equal(left, right, check_dtype=False,
                                          check_exact=False, rtol=0, atol=1e-12)
            checks.append({'rq': rq, 'table': reference.name, 'rows': len(left)})
            count += 1
        assert count, ('No rebuilt tables compared', rq)
    return checks


def collect(work, full):
    made = work / 'reproduced'
    for source in made.glob('source_test_*.csv'):
        dest = RESULTS / 'source_test' / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source, dest)
    for source in (made / 'source_duplicates').glob('*'):
        if source.is_file():
            dest = RESULTS / 'source_duplicates' / source.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(source, dest)
    if not full:
        return
    for rq in ('rq1', 'rq2', 'rq3'):
        folder = made / rq / ('tables' if rq != 'rq3' else '')
        for source in folder.glob('*'):
            if source.suffix not in ('.csv', '.json'):
                continue
            dest = RESULTS / 'tables' / rq / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for source in (made / 'paper/tables').glob('*.tex'):
        dest = RESULTS / 'tables' / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    figures = RESULTS / 'figures'
    figures.mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'src/workflow/Fig1.pdf', figures / 'Fig1.pdf')
    for source in (made / 'paper/figures').glob('*.pdf'):
        shutil.copy2(source, figures / source.name)
    shutil.copy2(made / 'source_duplicates/figure/Fig11-source-duplicates.pdf', figures / 'Fig11.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='verify scores, selections and duplicate-text sensitivity (default)')
    modes.add_argument('--all', action='store_true', help='also rebuild bootstrap analyses, tables and figures')
    args = parser.parse_args()
    checked = check_files()
    RESULTS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.work-', dir=RESULTS) as temporary:
        work = Path(temporary)
        stage(work)
        os.environ['REPLICATION_WORKSPACE'] = str(work)
        spec = importlib.util.spec_from_file_location('verified_predictions', ROOT / 'src/verify_predictions.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        engine.basic_check()
        duplicate_fields = duplicate_check(work)
        report = json.loads((work / 'reproduced/verification.json').read_text())
        report['package_files_verified'] = checked
        report['duplicate_audit_fields_matched'] = duplicate_fields
        report['mode'] = 'all' if args.all else 'check'
        if args.all:
            engine.full_analysis()
            report['statistical_tables_matched'] = compare_tables(work)
            shutil.copy2(ROOT / 'src/build_assets.py', work / 'verify_assets.py')
            run(work / 'verify_assets.py')
            figdir = work / 'reproduced/source_duplicates/figure'
            figdir.mkdir()
            shutil.copy2(work / 'analysis/source_duplicates/figure/build_figure.py', figdir)
            shutil.copy2(ROOT / 'src/figure_style.py', figdir / 'orx_figstyle.py')
            run(figdir / 'build_figure.py')
            report['asset_scripts'] = json.loads((work / 'reproduced/asset_checks.json').read_text())
            report['source_duplicate_figure'] = 'PASS'
        collect(work, args.all)
    (RESULTS / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: results/verification.json' + ('; results/tables/; results/figures/' if args.all else ''))


if __name__ == '__main__':
    main()
