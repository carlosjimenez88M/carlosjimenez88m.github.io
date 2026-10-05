"""Rebuild due luck essays and publish only generated files to the Pages branch.

Default is a dry run. --apply uses the real current time, requires a clean master
checkout, fast-forwards from origin, builds in a temporary folder, stages only
generated output, and pushes. It never accesses newsletter credentials.
"""
import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

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

def mounted_sources(tracked):
    """Identify tracked files mounted as static, including paired root output."""
    config=tomllib.loads((ROOT/'hugo.toml').read_text())
    sources=set()
    for mount in config.get('module',{}).get('mounts',[]):
        if not mount.get('target','').startswith('static'):
            continue
        source=Path(mount['source'])
        for path in tracked:
            if source not in path.parents:
                continue
            relative=path.relative_to(source).as_posix()
            includes=mount.get('includeFiles',[])
            excludes=mount.get('excludeFiles',[])
            if ((not includes or any(fnmatch(relative,pattern) for pattern in includes))
                    and not any(fnmatch(relative,pattern) for pattern in excludes)):
                sources.add(path)
    return sources

def releases_at(now):
    releases=json.loads((ROOT/'research/luck/releases.json').read_text())['releases']
    for release in releases:
        for field in ('blog_at','newsletter_at'):
            value=datetime.fromisoformat(release[field].replace('Z','+00:00'))
            if value.tzinfo is None or value.utcoffset() is None:
                raise SystemExit('Release timestamps must contain an explicit timezone.')
    releases.sort(key=lambda r:(datetime.fromisoformat(r['blog_at']),r['part']))
    due=[r for r in releases if datetime.fromisoformat(r['blog_at'])<=now]
    return releases,due

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    now=datetime.now(timezone.utc)
    releases,due=releases_at(now)
    print(json.dumps({'checked_at':now.isoformat(),'due':[r['slug'] for r in due],'apply':args.apply},indent=2))
    if not args.apply:return
    if git('branch','--show-current')!='master':raise SystemExit('Expected master checkout; stopped without publishing.')
    if git('status','--porcelain'):raise SystemExit('Checkout has unrelated changes; stopped without staging them.')
    subprocess.run(['git','pull','--ff-only','origin','master'],cwd=ROOT,check=True)
    if git('status','--porcelain'):raise SystemExit('Checkout changed during synchronization; stopped.')
    # Synchronization can change the manifest; use only the fetched boundaries.
    now=datetime.now(timezone.utc)
    releases,due=releases_at(now)
    if not due:
        print('No essays are due after synchronization.')
        return
    with tempfile.TemporaryDirectory(prefix='luck-release-') as td:
        build=Path(td)/'site'
        build_env=os.environ.copy()
        build_env['HUGO_RESOURCEDIR']=td+'/resources'
        subprocess.run(['hugo','--destination',str(build),'--cacheDir',td+'/cache',
                        '--noBuildLock','--minify','--clock',now.isoformat()],cwd=ROOT,env=build_env,check=True)
        for release in releases:
            exists=(build/'post'/release['slug']/'index.html').is_file()
            if exists!=(release in due):raise SystemExit('Release boundary failed; stopped before copying files.')
        files=[p.relative_to(build) for p in build.rglob('*') if p.is_file()]
        if any(is_protected(p) for p in files):raise SystemExit('Generated output collides with source paths.')
        old=[Path(p) for p in git('ls-files','public').splitlines()]
        tracked_paths={p.relative_to('public') for p in old}
        tracked={Path(p) for p in git('ls-files').splitlines()}
        root_sources={p for p in tracked if p.parts[0]!='public' and p not in tracked_paths}
        root_sources.update(mounted_sources(tracked))
        affected=set(files)|tracked_paths
        preserve_root=set()
        for rel in affected:
            collision=next((source for source in root_sources
                            if source==rel or source in rel.parents or rel in source.parents),None)
            if collision is not None:
                # TidyTuesday figures are both mounted static output and source
                # at the same root path. Preserve the source rather than copy
                # onto it; only an identical output may share that location.
                if (collision==rel and (build/rel).is_file() and (ROOT/rel).is_file()
                        and (build/rel).read_bytes()==(ROOT/rel).read_bytes()):
                    preserve_root.add(rel)
                    continue
                raise SystemExit(f'Generated output {rel} collides with a tracked source file {collision}; stopped before copying files.')
        staged=set()
        for rel in tracked_paths-set(files):
            if is_protected(rel):raise SystemExit('Old output collides with source paths.')
            for prefix in (ROOT,ROOT/'public'):
                dst=prefix/rel
                if dst.is_file():dst.unlink()
                staged.add(str(dst.relative_to(ROOT)))
        for rel in files:
            for prefix in (ROOT,ROOT/'public'):
                if prefix==ROOT and rel in preserve_root:
                    continue
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
