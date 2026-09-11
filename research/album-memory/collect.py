"""Retrieve private analysis inputs. Public manifest includes hashes, never lyrics."""
from pathlib import Path
import os,json,re,hashlib,time,unicodedata,importlib.util
from dotenv import load_dotenv
from lyricsgenius import Genius
ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'.research-cache/album-memory'; PRIVATE.mkdir(parents=True,exist_ok=True)
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
load_dotenv(ROOT/'.env')
ALBUMS=[
 {'id':'rhcp','artist':'Red Hot Chili Peppers','album':'Californication','year':1999,'tracks':['Around the World','Parallel Universe','Scar Tissue','Otherside','Get on Top','Californication','Easily','Porcelain','Emit Remmus','I Like Dirt','This Velvet Glove','Savior','Purple Stain','Right on Time',"Road Trippin'"]},
 {'id':'roach','artist':'Papa Roach','album':'Infest','year':2000,'tracks':['Infest','Last Resort','Broken Home','Dead Cell','Between Angels and Insects','Blood Brothers','Revenge','Snakes','Never Enough','Binge','Thrown Away','Tightrope']},
 {'id':'beatles','artist':'The Beatles','album':'Abbey Road','year':1969,'tracks':['Come Together','Something',"Maxwell's Silver Hammer",'Oh! Darling',"Octopus's Garden","I Want You (She's So Heavy)",'Here Comes the Sun','Because','You Never Give Me Your Money','Sun King','Mean Mr. Mustard','Polythene Pam','She Came In Through the Bathroom Window','Golden Slumbers','Carry That Weight','The End','Her Majesty']},
 {'id':'natalia','artist':'Natalia Lafourcade','album':'De todas las flores','year':2022,'tracks':['Vine solita','De todas las flores','Pasan los días','Llévame viento','El lugar correcto','Pajarito colibrí','María la curandera','Caminar bonito','Mi manera de querer','Muerte','Canta la arena','Que te vaya bonito Nicolás']},
]
def normalize(s):
 return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
def clean(raw):
 raw=re.sub(r'^.*?Lyrics','',raw,count=1,flags=re.S)
 lines=[]
 for line in raw.splitlines():
  line=line.strip()
  if not line or re.match(r'^\[.*\]$',line):continue
  if re.search('You might also like|Contributors|Translations',line):continue
  line=re.sub(r'\d*Embed$','',line).strip()
  if line:lines.append(line)
 return lines

def main():
 g=Genius(os.environ['GENIUS_API_TOKEN'],timeout=25,retries=1,remove_section_headers=False,skip_non_songs=True)
 records=[]
 for a in ALBUMS:
  for n,title in enumerate(a['tracks'],1):
   ident=f"{a['id']}-{n:02d}"; path=PRIVATE/f'{ident}.json'
   if path.exists(): data=json.loads(path.read_text())
   else:
    try:
     song=g.search_song(title,a['artist'])
     if not song:raise ValueError('not_found')
     if normalize(song.artist)!=normalize(a['artist']):raise ValueError('artist_mismatch')
     if normalize(song.title)!=normalize(title):raise ValueError('title_mismatch')
     lines=clean(song.lyrics)
     if not lines:raise ValueError('empty_lyrics')
     data={'id':ident,'artist':a['artist'],'album':a['album'],'title':title,'position':n,'source_url':song.url,'genius_id':song._body['id'],'lines':lines}
     path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
     time.sleep(.3)
    except Exception as e:
     print(ident,title,'FAILED',type(e).__name__,str(e)[:140],flush=True)
     records.append({'id':ident,'title':title,'status':'missing','error_type':type(e).__name__});continue
   text='\n'.join(data['lines'])
   record={k:v for k,v in data.items() if k!='lines'}
   record.update(status='ok',line_count=len(data['lines']),word_count=len(text.split()),sha256=hashlib.sha256(text.encode()).hexdigest())
   records.append(record);print(ident,title,'OK',len(data['lines']),'lines',flush=True)
  (OUT/'corpus_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 (OUT/'albums.json').write_text(json.dumps(ALBUMS,ensure_ascii=False,indent=2))
 print('Complete:',sum(r['status']=='ok' for r in records),'/',len(records),flush=True)
if __name__=='__main__':main()
