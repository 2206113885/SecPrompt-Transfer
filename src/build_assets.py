"""Exercise retained asset scripts on a disposable output copy, without inference."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
dest=ROOT/'reproduced/paper'
shutil.copytree(ROOT/'paper',dest,dirs_exist_ok=True)
(dest/'research').mkdir(exist_ok=True)
(dest/'tables').mkdir(exist_ok=True)
commands=['tools/review_supplements.py','tools/test_review_supplements.py',
          'tools/build_source_test_reference.py','tools/build_concise_definition_table.py',
          'tools/build_results_assets.py','figures/build_rq3_figures.py',
          'figures/build_rq3_matrix_overview.py']
checks=[]
for cmd in commands:
    print('ASSET CHECK',cmd,flush=True)
    subprocess.run([sys.executable,str(dest/cmd)],check=True,cwd=dest)
    checks.append({'script':cmd,'status':'PASS'})
(ROOT/'reproduced/asset_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
