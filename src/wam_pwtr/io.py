"""Portable artifact resolution and integrity checks."""
import copy,hashlib,json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def resolve(root,relative):
 if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('artifact path must be relative and contained')
 return Path(root).resolve()/relative

def dataset(training,assets):
 """Read caller-supplied training metadata and resolve its binary asset paths."""
 training=Path(training);index=read(training/'assets.json')
 for name,key in [('pool.json','pool_manifest_sha256'),('reference_scores.json','reference_scores_sha256')]:
  if sha(training/name)!=index[key]:raise ValueError('modified training data: '+name)
 m=read(training/'pool.json')
 for item in [m['initial_checkpoint'],*m['teacher_checkpoints'].values()]:item['path']=str(resolve(assets,item['path']))
 for ep in m['episodes']:
  ep['behavior_checkpoint']['path']=str(resolve(assets,ep['behavior_checkpoint']['path']))
  for r in ep['records']:r['path']=str(resolve(assets,r['path']))
 return m,index

def verify_assets(training,assets,*,hash_files=False,purpose='all'):
 """Check the record library and the model inputs needed for one use case.

 ``all`` checks the complete externally supplied inventory. Training needs the
 initial student and two teachers; reference inference needs only PWTR-u96.
 """
 roles={'all':None,
        'train':('initial','move_stapler_pad','open_microwave'),
        'infer':('pwtr_u96',)}
 if purpose not in roles:raise ValueError('asset purpose must be all, train or infer')
 m,index=dataset(training,assets);missing=[];bad=[]
 files={r['path']:r for ep in m['episodes'] for r in ep['records']}
 for role in (index['models'] if purpose=='all' else roles[purpose]):
  item=index['models'][role];files[str(resolve(assets,item['path']))]=item
 for path,item in files.items():
  p=Path(path)
  if not p.is_file():missing.append(path)
  elif p.stat().st_size!=item['size_bytes']:bad.append(dict(path=path,error='size'))
  elif hash_files and sha(p)!=item['sha256']:bad.append(dict(path=path,error='sha256'))
 return dict(status='READY' if not missing and not bad else 'MISSING_OR_INVALID_ASSETS',purpose=purpose,expected_files=len(files),missing=missing,invalid=bad,hashes_checked=hash_files)

def verify_release(release):
 root=Path(release);count=0
 for line in (root/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);p=resolve(root,name)
  if not p.is_file() or sha(p)!=expected:raise ValueError('release file missing/changed: '+name)
  count+=1
 return dict(status='VERIFIED',files=count,scope='release-file integrity')
