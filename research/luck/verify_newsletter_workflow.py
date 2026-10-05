"""Test Buttondown release behavior with simulated HTTP sessions only.

No .env access, external HTTP, subscribers, or real sends. Receipts and temporary
newsletter fixtures stay in a temporary directory; only the check summary is
written to results/newsletter-workflow-check.json.
"""
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import requests

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('luck_scheduler', ROOT / 'scripts/schedule_luck_newsletters.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class Response:
    def __init__(self, data=None, status=200, text='', content=b''):
        self.data, self.status_code, self.text, self.content = data, status, text, content
        self.ok = 200 <= status < 300

    def json(self):
        return deepcopy(self.data)


class FakeAPI:
    def __init__(self, emails):
        self.headers = {'Authorization': 'Token test-only-placeholder'}
        self.emails = {email['id']: deepcopy(email) for email in emails}
        self.calls = []
        self.publish_failure = None
        self.create_failure = None

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, deepcopy(kwargs)))
        if url == scheduler.API + '/emails' and method == 'GET':
            assert 'publish_date__start' not in kwargs.get('params', {})
            assert 'publish_date__end' not in kwargs.get('params', {})
            page = kwargs.get('params', {}).get('page', 1)
            values = list(self.emails.values())
            return Response({'results': values[(page - 1):page], 'count': len(values)})
        if url == scheduler.API + '/emails' and method == 'POST':
            data = deepcopy(kwargs['json'])
            if self.create_failure == 'timeout_before':
                raise requests.Timeout('simulated')
            data.update(id=f'test-created-{len(self.emails)}', absolute_url='https://buttondown.com/test/archive/example/')
            data.setdefault('filters', {'filters': [], 'groups': []})
            self.emails[data['id']] = data
            if self.create_failure == 'timeout_after':
                raise requests.Timeout('simulated')
            return Response(data)
        suffix = url.removeprefix(scheduler.API + '/emails/')
        email_id = suffix.split('/')[0]
        if email_id not in self.emails:
            return Response(status=404)
        email = self.emails[email_id]
        if method == 'GET':
            return Response(email)
        if method == 'PATCH':
            email.update(kwargs['json'])
            return Response(email)
        if method == 'POST' and suffix.endswith('/publish'):
            assert 'publish_date' not in kwargs['json']
            if self.publish_failure == 'timeout_before':
                raise requests.Timeout('simulated')
            email.update(kwargs['json'])
            email['status'] = 'about_to_send'
            if self.publish_failure == 'timeout_after':
                raise requests.Timeout('simulated')
            return Response(email)
        raise AssertionError('Unexpected simulated API request')


class FakePublic:
    def __init__(self, releases):
        self.headers = {}
        self.releases = releases
        self.calls = []
        self.article_status = 200
        self.image_status = 200
        self.wrong_title = False

    def get(self, url, **kwargs):
        assert not self.headers.get('Authorization')
        self.calls.append(url)
        if url.endswith('.png'):
            return Response(status=self.image_status, content=b'\x89PNG\r\n\x1a\nmock')
        release = next(r for r in self.releases if '/post/' + r['slug'] + '/' in url)
        title = 'Wrong essay' if self.wrong_title else release['subject'].split(' — Part')[0]
        return Response(status=self.article_status, text=f'<link rel="canonical" href="{url}"><h1>{title}</h1>')


class NewsletterChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='luck-newsletter-check-')
        self.root = Path(self.directory.name)
        (self.root / 'research/luck/newsletters').mkdir(parents=True)
        self.releases = deepcopy(json.loads((ROOT / 'research/luck/releases.json').read_text())['releases'])
        self.now = scheduler.aware_time(self.releases[0]['blog_at']) + timedelta(seconds=1)
        for index, release in enumerate(self.releases):
            release['newsletter_at'] = (self.now + timedelta(days=index + 1)).isoformat()
        self.write_manifest()
        self.bodies = {}
        for release in self.releases:
            body = f"Full reviewed essay {release['part']}\n\n![Figure]({scheduler.BASE}/img/luck/test.png)\n"
            self.bodies[release['part']] = body
            (self.root / f"research/luck/newsletters/part-{release['part']}.md").write_text(body)
        emails = [dict(id=f"test-part-{r['part']}", slug=r['slug'], subject=r['subject'],
                       status='scheduled', body=self.bodies[r['part']], publish_date=r['newsletter_at'],
                       canonical_url=f"{scheduler.BASE}/post/{r['slug']}/", archival_mode='enabled',
                       absolute_url='https://buttondown.com/test/archive/' + r['slug'] + '/',
                       filters={'filters': [], 'groups': []}) for r in self.releases]
        self.api = FakeAPI(emails)
        self.public = FakePublic(self.releases)

    def tearDown(self):
        self.directory.cleanup()

    def write_manifest(self):
        (self.root / 'research/luck/releases.json').write_text(json.dumps({'releases': self.releases}))

    def sync(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return scheduler.synchronize(self.root, self.api, self.public, self.now, **kwargs)

    def posts(self):
        return [call for call in self.api.calls if call[0] == 'POST']

    def test_body_subject_date_and_canonical_sync_reuse_ids(self):
        email = self.api.emails['test-part-2']
        email.update(body='Old body', subject='Old title', canonical_url='https://example.invalid/old/',
                     publish_date=(self.now + timedelta(days=30)).isoformat())
        self.sync(sync_bodies=True, sync_dates=True)
        self.assertEqual(email['body'], self.bodies[2])
        self.assertEqual(email['subject'], self.releases[1]['subject'])
        self.assertEqual(scheduler.aware_time(email['publish_date']), scheduler.aware_time(self.releases[1]['newsletter_at']))
        self.assertEqual(email['canonical_url'], f"{scheduler.BASE}/post/{self.releases[1]['slug']}/")
        self.assertFalse(self.posts())
        self.assertEqual(len(self.api.emails), 3)

    def test_scheduled_date_mismatch_requires_flag(self):
        self.api.emails['test-part-1']['publish_date'] = (self.now + timedelta(days=30)).isoformat()
        with self.assertRaisesRegex(SystemExit, 'date or status differs'):
            self.sync()
        self.assertFalse(self.posts())

    def test_immediate_existing_send_and_repeat_are_idempotent(self):
        self.releases[0]['newsletter_at'] = (self.now - timedelta(hours=1)).isoformat()
        self.write_manifest()
        self.api.emails['test-part-1']['body'] = 'Old version'
        self.sync(publish_part=1)
        self.assertEqual(len(self.posts()), 1)
        self.assertTrue(self.posts()[0][1].endswith('/test-part-1/publish'))
        self.assertEqual(self.api.emails['test-part-1']['body'], self.bodies[1])
        self.assertTrue(self.public.calls)
        self.sync(publish_part=1)
        self.assertEqual(len(self.posts()), 1)
        self.sync()
        self.assertEqual(len(self.posts()), 1)

    def test_public_article_404_prevents_send(self):
        self.public.article_status = 404
        with self.assertRaisesRegex(SystemExit, 'article returned HTTP 404'):
            self.sync(publish_part=1)
        self.assertFalse(self.posts())

    def test_public_image_404_prevents_send(self):
        self.public.image_status = 404
        with self.assertRaisesRegex(SystemExit, 'PNG returned HTTP 404'):
            self.sync(publish_part=1)
        self.assertFalse(self.posts())

    def test_wrong_article_title_prevents_send(self):
        self.public.wrong_title = True
        with self.assertRaisesRegex(SystemExit, 'title or canonical'):
            self.sync(publish_part=1)
        self.assertFalse(self.posts())

    def test_authorized_session_cannot_be_used_for_public_urls(self):
        self.public.headers['Authorization'] = 'Token test-only-placeholder'
        with self.assertRaisesRegex(SystemExit, 'without Authorization'):
            self.sync(publish_part=1)
        self.assertFalse(self.public.calls)
        self.assertFalse(self.posts())

    def test_publish_timeout_after_acceptance_recovers_without_duplicate(self):
        self.api.publish_failure = 'timeout_after'
        self.sync(publish_part=1)
        self.sync(publish_part=1)
        self.assertEqual(len(self.posts()), 1)

    def test_publish_unknown_outcome_is_not_retried(self):
        self.api.publish_failure = 'timeout_before'
        with self.assertRaisesRegex(SystemExit, 'Publish outcome is unknown'):
            self.sync(publish_part=1)
        self.api.publish_failure = None
        with self.assertRaisesRegex(SystemExit, 'earlier publish outcome remains unknown'):
            self.sync(publish_part=1)
        self.assertEqual(len(self.posts()), 1)

    def test_creation_timeout_recovers_by_slug_across_all_dates(self):
        del self.api.emails['test-part-1']
        self.api.create_failure = 'timeout_after'
        self.sync()
        self.sync()
        self.assertEqual(len(self.posts()), 1)
        self.assertEqual(len(self.api.emails), 3)

    def test_creation_unknown_outcome_is_not_retried(self):
        del self.api.emails['test-part-1']
        self.api.create_failure = 'timeout_before'
        with self.assertRaisesRegex(SystemExit, 'Creation outcome is unknown'):
            self.sync()
        self.api.create_failure = None
        with self.assertRaisesRegex(SystemExit, 'Earlier creation outcome is unknown'):
            self.sync()
        self.assertEqual(len(self.posts()), 1)

    def test_saved_id_cannot_point_to_another_slug(self):
        self.sync()
        receipt = self.root / '.research-cache/luck-buttondown-receipts.json'
        saved = json.loads(receipt.read_text())
        saved[self.releases[0]['slug']]['id'] = 'test-part-2'
        receipt.write_text(json.dumps(saved))
        with self.assertRaisesRegex(SystemExit, 'Receipt and existing slug disagree|stable slug'):
            self.sync(publish_part=1)
        self.assertFalse(self.posts())

    def test_naive_manifest_time_is_rejected(self):
        self.releases[0]['blog_at'] = '2026-10-05T08:00:00'
        self.write_manifest()
        with self.assertRaisesRegex(SystemExit, 'explicit timezone'):
            self.sync()
        self.assertFalse(self.api.calls)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NewsletterChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {'fixture': 'temporary files and simulated HTTP sessions; no .env reads or real API calls',
               'tests_run': result.testsRun, 'status': 'passed' if result.wasSuccessful() else 'failed',
               'checks': ['same-id body/subject/date/canonical updates', 'date mismatch rejected without flag',
                          'existing-id immediate publication and idempotent rerun', 'article and image 404 block send',
                          'correct article title required', 'API credential isolated from public requests',
                          'unknown mutation recovery without POST retry', 'all-date slug recovery',
                          'receipt identity validation', 'timezone-aware timestamps required']}
    (ROOT / 'research/luck/results/newsletter-workflow-check.json').write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
