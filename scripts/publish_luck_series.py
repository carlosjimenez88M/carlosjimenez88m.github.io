"""Rebuild due luck essays and publish only generated files to the Pages branch.

Default is a dry run. --apply uses the real current time, requires a clean master
checkout, fast-forwards from origin, builds in a temporary folder, stages only
generated output, and pushes. It never accesses newsletter credentials.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PROTECTED={'.git','.env','content','research','scripts','layouts','themes','static','assets','archetypes','resources'}

def is_protected(path):
    # Hugo's fingerprinted stylesheet is generated under assets/css; the
    # adjacent assets/css/extended directory contains editable source files.
    if path.parts[:2]==('assets','css') and path.name.startswith('stylesheet.') and path.suffix=='.css':
        return False
    return path.parts[0] in PROTECTED

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    now=datetime.now(timezone.utc)
    releases=json.loads((ROOT/'research/luck/releases.json').read_text())['releases']
    due=[r for r in releases if datetime.fromisoformat(r['blog_at'])<=now]
    print(json.dumps({'checked_at':now.isoformat(),'due':[r['slug'] for r in due],'apply':args.apply},indent=2))
    if not args.apply or not due:return
    if git('branch','--show-current')!='master':raise SystemExit('Expected master checkout; stopped without publishing.')
    if git('status','--porcelain'):raise SystemExit('Checkout has unrelated changes; stopped without staging them.')
    subprocess.run(['git','pull','--ff-only','origin','master'],cwd=ROOT,check=True)
    if git('status','--porcelain'):raise SystemExit('Checkout changed during synchronization; stopped.')
    with tempfile.TemporaryDirectory(prefix='luck-release-') as td:
        build=Path(td)/'site'
        subprocess.run(['hugo','--destination',str(build),'--cacheDir',td+'/cache',
                        '--noBuildLock','--minify','--clock',now.isoformat()],cwd=ROOT,check=True)
        for release in releases:
            exists=(build/'post'/release['slug']/'index.html').is_file()
            if exists!=(release in due):raise SystemExit('Release boundary failed; stopped before copying files.')
        files=[p.relative_to(build) for p in build.rglob('*') if p.is_file()]
        if any(is_protected(p) for p in files):raise SystemExit('Generated output collides with source paths.')
        old=[Path(p) for p in git('ls-files','public').splitlines()]
        tracked_paths={p.relative_to('public') for p in old}
        staged=set()
        for rel in tracked_paths-set(files):
            if is_protected(rel):raise SystemExit('Old output collides with source paths.')
            for prefix in (ROOT,ROOT/'public'):
                dst=prefix/rel
                if dst.is_file():dst.unlink()
                staged.add(str(dst.relative_to(ROOT)))
        for rel in files:
            for prefix in (ROOT,ROOT/'public'):
                dst=prefix/rel
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(build/rel,dst)
                staged.add(str(dst.relative_to(ROOT)))
        subprocess.run(['git','add','-A','--',*sorted(staged)],cwd=ROOT,check=True)
        # Syntax-highlighted legacy examples can preserve trailing whitespace
        # inside generated HTML; retain conflict checks without rejecting it.
        subprocess.run(['git','-c','core.whitespace=-blank-at-eol,-blank-at-eof',
                        'diff','--cached','--check'],cwd=ROOT,check=True)
        if git('diff','--cached','--name-only'):
            subprocess.run(['git','commit','-m',f"Publish luck series through part {due[-1]['part']}"],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
        print('Published due essays; verify their public URLs after the Pages rebuild.')

if __name__=='__main__':main()
