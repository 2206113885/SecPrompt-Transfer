"""Compare rebuilt statistical tables and scan release files before publication."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXCLUDE={'reproduced','.git','.venv','__pycache__'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
                  and not set(p.relative_to(ROOT).parts)&EXCLUDE and p.name!='.DS_Store')


def main():
    checks={'new_model_calls':0,'score_tolerance':1e-12,'comparisons':[]}
    for rq,folder in [('rq1','rq1'),('rq2','rq2'),('rq3','rq3-systematic')]:
        for reference in sorted((ROOT/'paper/data'/folder).glob('*.csv')):
            got=ROOT/'reproduced'/rq/('tables' if rq!='rq3' else '')/reference.name
            if not got.exists():continue
            left,right=pd.read_csv(reference),pd.read_csv(got)
            pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
            checks['comparisons'].append({'reference':str(reference.relative_to(ROOT)),
                                          'rows':len(left),'status':'PASS'})
    assert len(checks['comparisons'])>=15
    patterns={
        'huggingface_token':re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
        'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
        'api_key':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
        'private_key':re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
    }
    findings=[]; scanned=0
    def scan(data,name):
        nonlocal scanned
        scanned+=1
        for kind,pat in patterns.items():
            if pat.search(data):findings.append({'file':name,'kind':kind})
    for p in files():
        if p.suffix in {'.npz','.pdf','.png','.svg'}:continue
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if not n.endswith('/'):scan(z.read(n),str(p.relative_to(ROOT))+'!'+n)
        elif p.suffix=='.gz':
            with gzip.open(p,'rb') as f:scan(f.read(),str(p.relative_to(ROOT)))
        else:scan(p.read_bytes(),str(p.relative_to(ROOT)))
    checks['secret_pattern_scan']={'members_scanned':scanned,'findings':findings}
    assert not findings, findings
    checks['largest_file_bytes']=max(p.stat().st_size for p in files())
    assert checks['largest_file_bytes']<100_000_000
    checks['status']='PASS'
    dest=ROOT/'verification';dest.mkdir(exist_ok=True)
    (dest/'release_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Evidence checksums exclude generated outputs and mutable prose/figure assets.
    frozen=[]
    for p in files():
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='data' or (rel.parts[0]=='analysis' and 'data' in rel.parts) or rel.parts[:2]==('paper','data'):
            frozen.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(rel))
    (ROOT/'MANIFEST.sha256').write_text('\n'.join(frozen)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
