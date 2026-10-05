"""Exercise the publisher in a temporary checkout and local bare remote only."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]

def run(*args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT).strip()

def main():
    with tempfile.TemporaryDirectory(prefix='luck-workflow-check-') as td:
        checkout=Path(td)/'checkout';checkout.mkdir()
        remote=Path(td)/'remote.git'
        run('git','init','--bare','--initial-branch=master',str(remote))
        run('git','init','--initial-branch=master',str(checkout))
        for name in ('content','themes','layouts','assets','static','tidytuesday'):
            shutil.copytree(ROOT/name,checkout/name,ignore=shutil.ignore_patterns('__pycache__','*.pkl','data','*.ipynb'))
        for name in ('hugo.toml','.gitignore'):
            shutil.copy2(ROOT/name,checkout/name)
        (checkout/'research/luck').mkdir(parents=True)
        shutil.copy2(ROOT/'research/luck/releases.json',checkout/'research/luck/releases.json')
        (checkout/'notes-user-owned.txt').write_text('Keep this source file unchanged.\n')
        for prefix in (checkout,checkout/'public'):
            css=prefix/'assets/css/stylesheet.old.css'
            css.parent.mkdir(parents=True,exist_ok=True)
            css.write_text('/* obsolete generated stylesheet */\n')
        run('git','config','user.name','Release Verification',cwd=checkout)
        run('git','config','user.email','release-check@example.invalid',cwd=checkout)
        run('git','remote','add','origin',str(remote),cwd=checkout)
        run('git','add','.',cwd=checkout)
        run('git','commit','-m','Temporary fixture',cwd=checkout)
        run('git','push','-u','origin','master',cwd=checkout)
        spec=importlib.util.spec_from_file_location('luck_publisher',ROOT/'scripts/publish_luck_series.py')
        publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
        publisher.ROOT=checkout
        class Clock:
            @staticmethod
            def now(tz):return datetime(2026,10,12,13,0,1,tzinfo=timezone.utc)
            fromisoformat=staticmethod(datetime.fromisoformat)
        publisher.datetime=Clock
        old_argv=sys.argv;sys.argv=['publish_luck_series.py','--apply']
        try:
            publisher.main()
            first=run('git','rev-parse','HEAD',cwd=checkout)
            publisher.main()
            assert run('git','rev-parse','HEAD',cwd=checkout)==first,'Repeat created an unnecessary commit'
            assert not run('git','status','--porcelain',cwd=checkout)
            assert run('git','rev-parse','origin/master',cwd=checkout)==first
            releases=json.loads((checkout/'research/luck/releases.json').read_text())['releases']
            for r in releases:
                assert (checkout/'post'/r['slug']/'index.html').exists()==(r['part']==1)
            assert not (checkout/'assets/css/stylesheet.old.css').exists()
            assert (checkout/'notes-user-owned.txt').read_text()=='Keep this source file unchanged.\n'
            committed=run('git','diff-tree','--no-commit-id','--name-only','-r','HEAD',cwd=checkout).splitlines()
            assert all(not p.startswith(('content/','research/','scripts/','layouts/','themes/','static/')) for p in committed)
            (checkout/'notes-user-owned.txt').write_text('Unrelated unfinished edit.\n')
            try:publisher.main()
            except SystemExit as error:
                assert 'unrelated changes' in str(error)
            else:raise AssertionError('Dirty checkout was not rejected')
            assert not run('git','diff','--cached','--name-only',cwd=checkout)
        finally:sys.argv=old_argv
    result={'fixture':'temporary checkout and local bare remote; no production push',
            'checks':['due-only publication','clean checkout after build','local remote updated',
                      'repeat is idempotent','obsolete generated output removed',
                      'source preserved','dirty checkout rejected without staging'],
            'status':'passed'}
    (ROOT/'research/luck/results/workflow-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
