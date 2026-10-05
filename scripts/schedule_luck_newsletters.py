"""Synchronize authorized Buttondown essays without creating duplicate sends.

--sync-bodies updates edited scheduled bodies; --sync-dates moves the same ids
only to future times. --publish-part N sends that existing email immediately,
after checking its public article and every PNG used by the email. API and public
HTTP sessions are separate. Receipts checkpoint mutation intent and readback;
an ambiguous send is never retried automatically. Credentials are loaded only
by main(), and are never included in public requests or printed output.
Buttondown can acknowledge immediate publication while temporarily reporting
scheduled with a due timestamp; this is recorded as pending delivery, not sent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.buttondown.com/v1'
BASE = 'https://carlosdanieljimenez.com'
FINISHED = {'sent', 'about_to_send', 'in_flight', 'throttled', 'resending', 'partially_sent'}


class UnknownOutcome(Exception):
    """A mutation may have reached the service; retrieve before proceeding."""


def aware_time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit('Release timestamps must contain an explicit timezone.')
    return parsed


def api_request(session, method, path, **kwargs):
    try:
        response = session.request(method, API + path, timeout=45, **kwargs)
    except requests.RequestException:
        if method == 'POST':
            raise UnknownOutcome('Buttondown mutation outcome is unknown.') from None
        raise SystemExit('Buttondown request failed; no credentials or response body logged.') from None
    if not response.ok:
        if method == 'POST' and response.status_code >= 500:
            raise UnknownOutcome('Buttondown mutation outcome is unknown.')
        raise SystemExit(f'Buttondown returned HTTP {response.status_code}; no response body or credential logged.')
    try:
        return response.json()
    except ValueError:
        if method == 'POST':
            raise UnknownOutcome('Buttondown mutation returned an unreadable response.') from None
        raise SystemExit('Buttondown returned an unreadable response.') from None


def list_candidates(session):
    # Search all dates: a moved email must still be recoverable by its fixed slug.
    candidates = []
    page = 1
    while True:
        listing = api_request(session, 'GET', '/emails',
                              params={'page': page, 'excluded_fields': 'body'})
        results = listing.get('results', [])
        candidates.extend(results)
        if not results or (listing.get('count') is not None and len(candidates) >= listing['count']):
            return candidates
        page += 1


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_h1 = False
        self.headings = []
        self.current = []
        self.canonicals = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'h1':
            self.in_h1 = True
            self.current = []
        if tag == 'link' and 'canonical' in attrs.get('rel', '').split():
            self.canonicals.append(attrs.get('href', ''))

    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_h1:
            self.headings.append(' '.join(''.join(self.current).split()))
            self.in_h1 = False

    def handle_data(self, data):
        if self.in_h1:
            self.current.append(data)


def check_public_article(session, release, body):
    if session.headers.get('Authorization'):
        raise SystemExit('Public verification requires a session without Authorization.')
    url = f"{BASE}/post/{release['slug']}/"
    try:
        response = session.get(url, timeout=45)
        if response.status_code != 200:
            raise SystemExit(f'Public article returned HTTP {response.status_code}; newsletter not sent.')
        parser = ArticleParser()
        parser.feed(response.text)
        expected_title = html.unescape(release['subject'].split(' — Part')[0])
        if expected_title not in parser.headings or url not in parser.canonicals:
            raise SystemExit('Public article title or canonical URL differs; newsletter not sent.')
        images = sorted(set(re.findall(r'!\[[^\]]*\]\((https?://[^\s)]+\.png(?:\?[^\s)]*)?)\)', body)))
        if not images:
            raise SystemExit('No PNG illustrations found in the exported newsletter; newsletter not sent.')
        for image_url in images:
            image_response = session.get(image_url, timeout=45)
            if image_response.status_code != 200:
                raise SystemExit(f'Newsletter PNG returned HTTP {image_response.status_code}; newsletter not sent.')
            if not image_response.content.startswith(b'\x89PNG\r\n\x1a\n'):
                raise SystemExit('Newsletter illustration is not a PNG response; newsletter not sent.')
    except requests.RequestException:
        raise SystemExit('Public verification failed; newsletter not sent.') from None
    return {'article': url, 'png_images': len(images)}


def synchronize(root, api_session, public_session, now, *, sync_bodies=False,
                sync_dates=False, publish_part=None, current_time=None):
    if now.tzinfo is None or now.utcoffset() is None:
        raise SystemExit('Current time must contain an explicit timezone.')
    releases = json.loads((root / 'research/luck/releases.json').read_text())['releases']
    for release in releases:
        aware_time(release['blog_at'])
        aware_time(release['newsletter_at'])
    if publish_part is not None and publish_part not in {r['part'] for r in releases}:
        raise SystemExit('Requested part is missing from the release manifest.')
    receipt = root / '.research-cache/luck-buttondown-receipts.json'
    saved = json.loads(receipt.read_text()) if receipt.exists() else {}
    receipt.parent.mkdir(exist_ok=True)
    clock = current_time or (lambda: datetime.now(timezone.utc))

    def checkpoint():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(saved, indent=2) + '\n')
        temporary.replace(receipt)

    def record(slug, email, *, delivery_pending=False):
        prior = saved.get(slug, {})
        saved[slug] = {k: email[k] for k in
                       ('id', 'subject', 'status', 'publish_date', 'canonical_url', 'absolute_url')}
        saved[slug]['body_sha256'] = hashlib.sha256(email['body'].encode()).hexdigest()
        saved[slug]['verified_at'] = clock().isoformat()
        if prior.get('public_check'):
            saved[slug]['public_check'] = prior['public_check']
        if delivery_pending:
            saved[slug]['publication_state'] = 'accepted_pending_delivery'
            saved[slug]['pending_request'] = 'publish'
            saved[slug]['publish_accepted_at'] = prior.get('publish_accepted_at', clock().isoformat())
            if prior.get('request_started_at'):
                saved[slug]['request_started_at'] = prior['request_started_at']
        checkpoint()
        report = {k: saved[slug][k] for k in ('id', 'subject', 'status', 'publish_date')}
        if delivery_pending:
            report['publication_state'] = saved[slug]['publication_state']
        print(json.dumps(report))

    def verified_due_schedule(email, release, body, canonical):
        # Check a fresh time after readback: the service's immediate timestamp
        # can be a few seconds later than the invocation's initial `now`.
        return (email['status'] == 'scheduled' and email.get('publish_date') is not None
                and aware_time(email['publish_date']) <= clock()
                and email['body'] == body and email['subject'] == release['subject']
                and email['slug'] == release['slug'] and email['canonical_url'] == canonical)

    candidates = list_candidates(api_session)
    for release in sorted(releases, key=lambda r: r['part']):
        slug = release['slug']
        body = (root / f"research/luck/newsletters/part-{release['part']}.md").read_text()
        when = aware_time(release['newsletter_at'])
        canonical = f'{BASE}/post/{slug}/'
        matches = [e for e in candidates if e.get('slug') == slug and e.get('status') != 'deleted']
        if len(matches) > 1:
            raise SystemExit('Duplicate slug exists; stopped for review.')
        prior = saved.get(slug, {})
        if prior.get('id'):
            email = api_request(api_session, 'GET', '/emails/' + prior['id'])
            if matches and matches[0]['id'] != email['id']:
                raise SystemExit('Receipt and existing slug disagree; stopped without mutation.')
        elif matches:
            email = api_request(api_session, 'GET', '/emails/' + matches[0]['id'])
            saved[slug] = {'id': email['id']}
            checkpoint()
        else:
            if publish_part == release['part']:
                raise SystemExit('Immediate publication requires an existing email; no duplicate was created.')
            if prior.get('pending_request'):
                raise SystemExit('Earlier creation outcome is unknown and no matching email was found; no POST retry attempted.')
            if when <= now:
                raise SystemExit('Refusing to create an overdue send.')
            saved[slug] = {'pending_request': 'create', 'request_started_at': now.isoformat()}
            checkpoint()
            try:
                email = api_request(api_session, 'POST', '/emails', json={
                    'subject': release['subject'], 'slug': slug, 'body': body,
                    'status': 'scheduled', 'publish_date': when.astimezone(timezone.utc).isoformat(),
                    'canonical_url': canonical, 'archival_mode': 'enabled',
                    'metadata': {'series': 'statistical-luck-2026', 'part': release['part']}})
            except UnknownOutcome:
                recovered = [e for e in list_candidates(api_session)
                             if e.get('slug') == slug and e.get('status') != 'deleted']
                if len(recovered) != 1:
                    raise SystemExit('Creation outcome is unknown; no POST retry attempted. Check the existing slug before retrying.') from None
                email = recovered[0]
            except SystemExit:
                saved.pop(slug, None)
                checkpoint()
                raise
            saved[slug] = {'id': email['id']}
            checkpoint()
            email = api_request(api_session, 'GET', '/emails/' + email['id'])
        if email['slug'] != slug:
            raise SystemExit('Retrieved email does not match the expected stable slug.')
        if email['status'] in FINISHED:
            if email['canonical_url'] != canonical:
                raise SystemExit('Queued or sent email has an unexpected canonical URL; stopped for review.')
            if aware_time(release['blog_at']) > now and publish_part != release['part']:
                raise SystemExit('A future installment has already been queued or sent; stopped for review.')
            record(slug, email)
            continue
        if email['status'] not in {'scheduled', 'draft'}:
            raise SystemExit(f"Email is {email['status']}; stopped without sending or creating a replacement.")
        if email.get('archival_mode') != 'enabled' or any(email.get('filters', {}).get(k) for k in ('filters', 'groups')):
            raise SystemExit('Email archive or audience differs from the authorized unrestricted newsletter.')
        if prior.get('pending_request') == 'publish':
            if verified_due_schedule(email, release, body, canonical):
                record(slug, email, delivery_pending=True)
                continue
            raise SystemExit('An earlier publish outcome remains unknown or has a future/different readback; no POST retry attempted.')
        if publish_part == release['part']:
            public_check = check_public_article(public_session, release, body)
            saved[slug] = {'id': email['id'], 'pending_request': 'publish',
                           'request_started_at': now.isoformat(), 'public_check': public_check}
            checkpoint()
            try:
                api_request(api_session, 'POST', '/emails/' + email['id'] + '/publish', json={
                    'subject': release['subject'], 'body': body, 'slug': slug,
                    'canonical_url': canonical, 'archival_mode': 'enabled'})
            except UnknownOutcome:
                recovered = api_request(api_session, 'GET', '/emails/' + email['id'])
                if recovered['status'] not in FINISHED and not verified_due_schedule(recovered, release, body, canonical):
                    raise SystemExit('Publish outcome is unknown; readback is not queued or sent. No POST retry attempted.') from None
                email = recovered
            except SystemExit:
                saved[slug].pop('pending_request', None)
                checkpoint()
                raise
            else:
                saved[slug]['publish_accepted_at'] = clock().isoformat()
                checkpoint()
                email = api_request(api_session, 'GET', '/emails/' + email['id'])
            delivery_pending = verified_due_schedule(email, release, body, canonical)
            if email['status'] not in FINISHED and not delivery_pending:
                raise SystemExit('Publish readback is not queued or due-scheduled; no POST retry attempted.')
            if (email['body'] != body or email['subject'] != release['subject']
                    or email['slug'] != slug or email['canonical_url'] != canonical):
                raise SystemExit('Published content differs from the reviewed essay; no repeat send attempted.')
            record(slug, email, delivery_pending=delivery_pending)
            continue
        if when <= now:
            raise SystemExit('An overdue email is still unsent; use --publish-part explicitly after verifying its public article.')
        changes = {}
        if email['canonical_url'] != canonical:
            if not sync_bodies:
                raise SystemExit('Canonical URL differs; use --sync-bodies to synchronize the same id.')
            changes.update(canonical_url=canonical, slug=slug)
        if email['body'] != body:
            if not sync_bodies:
                raise SystemExit('Body differs from the local essay; use --sync-bodies to synchronize the same id.')
            changes['body'] = body
        if email['subject'] != release['subject']:
            if not sync_bodies:
                raise SystemExit('Subject differs from the local essay; use --sync-bodies to synchronize the same id.')
            changes['subject'] = release['subject']
        if email.get('publish_date') is None or aware_time(email['publish_date']) != when or email['status'] != 'scheduled':
            if not sync_dates:
                raise SystemExit('Scheduled date or status differs; use --sync-dates to synchronize the same id.')
            changes.update(status='scheduled', publish_date=when.astimezone(timezone.utc).isoformat())
        if changes:
            api_request(api_session, 'PATCH', '/emails/' + email['id'], json=changes)
            email = api_request(api_session, 'GET', '/emails/' + email['id'])
        if (email['status'] != 'scheduled' or aware_time(email['publish_date']) != when
                or email['subject'] != release['subject'] or email['body'] != body
                or email['slug'] != slug or email['canonical_url'] != canonical):
            raise SystemExit('Scheduled email readback differs from the reviewed release.')
        record(slug, email)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync-bodies', action='store_true', help='Update edited bodies and subjects on the existing ids.')
    parser.add_argument('--sync-dates', action='store_true', help='Move existing scheduled ids to their future manifest dates.')
    parser.add_argument('--publish-part', type=int, choices=(1, 2, 3), help='Send this existing part now, after public page and PNG verification.')
    args = parser.parse_args()
    key = dotenv_values(ROOT / '.env').get('BUTTONDOWN_API_KEY')
    if not key:
        raise SystemExit('BUTTONDOWN_API_KEY missing in .env')
    with requests.Session() as api_session, requests.Session() as public_session:
        api_session.headers['Authorization'] = 'Token ' + key
        synchronize(ROOT, api_session, public_session, datetime.now(timezone.utc),
                    sync_bodies=args.sync_bodies, sync_dates=args.sync_dates, publish_part=args.publish_part)


if __name__ == '__main__':
    main()
