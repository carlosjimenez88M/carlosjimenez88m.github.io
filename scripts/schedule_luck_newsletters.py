"""Schedule the three authorized complete essays in Buttondown, with readback.

Reads the existing API credential from the ignored .env; never logs it. The local
receipt is checkpointed after every POST, and matching slugs are checked before
creation so a retry cannot silently duplicate a send. Run after exporting bodies.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import requests
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]
API='https://api.buttondown.com/v1'
RECEIPT=ROOT/'.research-cache/luck-buttondown-receipts.json'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync-bodies',action='store_true',help='Update edited bodies of matching future scheduled emails without creating duplicate sends.')
    args=parser.parse_args()
    key=dotenv_values(ROOT/'.env').get('BUTTONDOWN_API_KEY')
    if not key:raise SystemExit('BUTTONDOWN_API_KEY missing in .env')
    session=requests.Session()
    session.headers['Authorization']='Token '+key
    def request(method,path,**kwargs):
        response=session.request(method,API+path,timeout=45,**kwargs)
        if not response.ok:
            raise SystemExit(f'Buttondown returned HTTP {response.status_code}; no response body or credential logged.')
        return response.json()
    releases=json.loads((ROOT/'research/luck/releases.json').read_text())['releases']
    saved=json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    RECEIPT.parent.mkdir(exist_ok=True)
    # Matching dates and exact slugs recover creations with an unknown outcome.
    candidates=[];page=1
    while True:
        listing=request('GET','/emails',params={'publish_date__start':'2026-10-12',
                        'publish_date__end':'2026-10-26','page':page})
        results=listing.get('results',[]);candidates.extend(results)
        if not results or len(candidates)>=listing.get('count',len(candidates)):break
        page+=1
    for r in releases:
        slug=r['slug']
        body=(ROOT/f"research/luck/newsletters/part-{r['part']}.md").read_text()
        when=datetime.fromisoformat(r['newsletter_at'])
        if when<=datetime.now(timezone.utc):raise SystemExit('Refusing to create an overdue send.')
        matches=[e for e in candidates if e['slug']==slug and e['status']!='deleted']
        if len(matches)>1:raise SystemExit('Duplicate slug exists; stopped for review.')
        if slug in saved:
            result=request('GET','/emails/'+saved[slug]['id'])
        elif matches:
            result=request('GET','/emails/'+matches[0]['id'])
        else:
            result=request('POST','/emails',json={
                'subject':r['subject'],'slug':slug,'body':body,
                'status':'scheduled','publish_date':when.astimezone(timezone.utc).isoformat(),
                'canonical_url':f'https://carlosdanieljimenez.com/post/{slug}/',
                'archival_mode':'enabled',
                'metadata':{'series':'statistical-luck-2026','part':r['part']}})
            saved[slug]={'id':result['id']}
            RECEIPT.write_text(json.dumps(saved,indent=2)+'\n')
        # Retrieve the persisted email independently, not merely the POST response.
        persisted=request('GET','/emails/'+result['id'])
        assert persisted['status']=='scheduled', 'Status differs from scheduled'
        assert datetime.fromisoformat(persisted['publish_date'].replace('Z','+00:00'))==when
        if persisted['body']!=body and args.sync_bodies:
            request('PATCH','/emails/'+persisted['id'],json={'body':body})
            persisted=request('GET','/emails/'+result['id'])
            assert persisted['status']=='scheduled'
            assert datetime.fromisoformat(persisted['publish_date'].replace('Z','+00:00'))==when
        assert persisted['subject']==r['subject']
        assert persisted['body']==body, 'Body differs from local complete essay'
        assert persisted['canonical_url']==f'https://carlosdanieljimenez.com/post/{slug}/'
        assert persisted['archival_mode']=='enabled'
        assert not persisted.get('filters',{}).get('filters'), 'Unexpected audience restriction'
        assert not persisted.get('filters',{}).get('groups'), 'Unexpected audience group restriction'
        saved[slug]={k:persisted[k] for k in ('id','subject','status','publish_date','canonical_url','absolute_url')}
        saved[slug]['body_sha256']=hashlib.sha256(body.encode()).hexdigest()
        saved[slug]['verified_at']=datetime.now(timezone.utc).isoformat()
        RECEIPT.write_text(json.dumps(saved,indent=2)+'\n')
        print(json.dumps({k:saved[slug][k] for k in ('id','subject','status','publish_date')}))

if __name__=='__main__':main()
