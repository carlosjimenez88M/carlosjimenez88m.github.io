"""Exercise the publisher in a temporary checkout and local bare remote only."""
from datetime import datetime, timedelta, timezone
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
            patterns=['__pycache__','*.pkl','data','*.ipynb']
            if name=='tidytuesday':patterns.append('index.html')
            shutil.copytree(ROOT/name,checkout/name,ignore=shutil.ignore_patterns(*patterns))
        for name in ('hugo.toml','.gitignore'):
            shutil.copy2(ROOT/name,checkout/name)
        # Root assets include deployed fingerprinted CSS alongside source.
        # Mirror those outputs into public so their role matches production.
        for stylesheet in (checkout/'assets/css').glob('stylesheet.*.css'):
            mirror=checkout/'public'/stylesheet.relative_to(checkout)
            mirror.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(stylesheet,mirror)
        (checkout/'research/luck').mkdir(parents=True)
        manifest=json.loads((ROOT/'research/luck/releases.json').read_text())
        first=min(manifest['releases'],key=lambda r:datetime.fromisoformat(r['blog_at']))
        release_clock=datetime.fromisoformat(first['blog_at'])+timedelta(seconds=1)
        old_manifest=json.loads(json.dumps(manifest))
        old_first=next(r for r in old_manifest['releases'] if r['part']==first['part'])
        old_first['blog_at']=(release_clock+timedelta(days=1)).isoformat()
        (checkout/'research/luck/releases.json').write_text(json.dumps(old_manifest,indent=2)+'\n')
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
        # A remote update makes I due, while the checkout initially says it is
        # not due. The publisher must fetch and then recompute the manifest.
        editor=Path(td)/'remote-editor'
        run('git','clone',str(remote),str(editor))
        run('git','config','user.name','Release Verification',cwd=editor)
        run('git','config','user.email','release-check@example.invalid',cwd=editor)
        (editor/'research/luck/releases.json').write_text(json.dumps(manifest,indent=2)+'\n')
        run('git','add','research/luck/releases.json',cwd=editor)
        run('git','commit','-m','Advance the first release in the remote manifest',cwd=editor)
        run('git','push','origin','master',cwd=editor)
        spec=importlib.util.spec_from_file_location('luck_publisher',ROOT/'scripts/publish_luck_series.py')
        publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
        publisher.ROOT=checkout
        class Clock:
            @staticmethod
            def now(tz):return release_clock.astimezone(tz)
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
            run('git','restore','notes-user-owned.txt',cwd=checkout)
            # A static root file must never overwrite an independently tracked
            # source at the same path, even outside protected directories.
            (checkout/'static/notes-user-owned.txt').write_text('Colliding static output.\n')
            run('git','add','static/notes-user-owned.txt',cwd=checkout)
            run('git','commit','-m','Add a deliberate collision fixture',cwd=checkout)
            try:publisher.main()
            except SystemExit as error:
                assert 'tracked source file' in str(error)
            else:raise AssertionError('Tracked root source collision was not rejected')
            assert (checkout/'notes-user-owned.txt').read_text()=='Keep this source file unchanged.\n'
            assert not run('git','status','--porcelain',cwd=checkout)
            naive=json.loads(json.dumps(manifest))
            naive['releases'][0]['blog_at']='2026-10-05T08:00:00'
            (checkout/'research/luck/releases.json').write_text(json.dumps(naive))
            try:publisher.releases_at(release_clock)
            except SystemExit as error:
                assert 'explicit timezone' in str(error)
            else:raise AssertionError('Naive release timestamp was not rejected')
        finally:sys.argv=old_argv
    result={'fixture':'temporary checkout and local bare remote; no production push',
            'checks':['due-only publication','clean checkout after build','local remote updated',
                      'repeat is idempotent','obsolete generated output removed',
                      'source preserved','dirty checkout rejected without staging',
                      'release boundaries recomputed after pull',
                      'tracked root source collision rejected before copying',
                      'timezone-aware release timestamps required'],
            'status':'passed'}
    (ROOT/'research/luck/results/workflow-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
