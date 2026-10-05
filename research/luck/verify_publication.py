"""Check every release boundary and the rendered essays' local links."""
from datetime import datetime, timedelta
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import unquote, urljoin, urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
BASE = 'https://carlosdanieljimenez.com'


class MainLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_main = False
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'main':
            self.in_main = True
        if self.in_main:
            field = 'href' if tag == 'a' else 'src' if tag == 'img' else None
            if field and attrs.get(field):
                self.links.append(attrs[field])

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_main = False


def main():
    releases = json.loads((ROOT / 'research/luck/releases.json').read_text())['releases']
    times = [datetime.fromisoformat(r['blog_at']) for r in releases]
    for release in releases:
        source = (ROOT / f"content/post/{release['slug']}.md").read_text()
        assert f"publishDate: {release['blog_at']}" in source
    clocks = [datetime.now(ZoneInfo('America/Bogota')), min(times) - timedelta(seconds=1)]
    clocks.extend(t + timedelta(seconds=1) for t in times)
    checks = []
    with tempfile.TemporaryDirectory(prefix='luck-boundaries-') as td:
        env = os.environ.copy()
        env['HUGO_RESOURCEDIR'] = td + '/resources'
        for index, clock in enumerate(clocks):
            build = Path(td) / str(index)
            subprocess.run(['hugo', '--destination', str(build), '--cacheDir', td + '/cache',
                            '--noBuildLock', '--minify', '--clock', clock.isoformat()],
                           cwd=ROOT, env=env, check=True, capture_output=True)
            expected = [r['part'] for r, t in zip(releases, times) if t <= clock]
            published = []
            missing = []
            count = 0
            for release in releases:
                canonical = release.get('canonical_url', f"{BASE}/post/{release['slug']}/")
                article = build / urlsplit(canonical).path.lstrip('/') / 'index.html'
                if not article.exists():
                    continue
                published.append(release['part'])
                legacy=build/'post'/release['slug']/'index.html'
                if release.get('canonical_url'):
                    assert legacy.exists() and canonical in legacy.read_text()
                    assert 'http-equiv=refresh' in legacy.read_text()
                parser = MainLinks()
                parser.feed(article.read_text())
                for link in parser.links:
                    if link.startswith('#'):
                        continue
                    parsed = urlsplit(urljoin(canonical, link))
                    if parsed.netloc != urlsplit(BASE).netloc:
                        continue
                    count += 1
                    relative = unquote(parsed.path).lstrip('/')
                    target = build / relative
                    if parsed.path.endswith('/'):
                        target = target / 'index.html'
                    if not target.is_file():
                        missing.append(parsed.path)
            assert published == expected, (clock, expected, published)
            assert not missing, (clock, missing)
            checks.append({'clock': clock.isoformat(), 'published_parts': published,
                           'checked_local_links': count, 'missing_local_links': missing})
    (ROOT / 'research/luck/results/publication-checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Verified: current publication and every release boundary; all rendered local links exist.')


if __name__ == '__main__':
    main()
