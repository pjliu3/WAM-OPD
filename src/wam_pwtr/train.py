"""Run the fixed 96-update mixed-pool PWTR training recipe."""
import json,os,shutil,struct,math
from pathlib import Path
from .io import read,sha,digest,dataset,verify_release
from .core import SPEC,sampled_view,schedule

def configuration(manifest,fastwam):
 import fastwam_opd
 return dict(activation_checkpointing=True,cpu_optimizer_offload=True,fastwam_root=str(Path(fastwam).resolve()),
  initial_checkpoint=manifest['initial_checkpoint']['path'],legacy_source=str(Path(fastwam_opd.__file__).resolve().parent.parent),
  mixed_precision='bf16',model_config='sim_robotwin.yaml',sampling=dict(action_horizon=32,num_inference_steps=10,rand_device='cpu',sigma_shift=None),
  score_batch_size=1,strength=.1,student_device='cuda:0',teacher_device='cuda:1',
  teachers=[dict(checkpoint=item['path'],name=task,tasks=[task]) for task,item in sorted(manifest['teacher_checkpoints'].items())])

def check_fastwam_source(release,fastwam):
 lock=read(Path(release)/'provenance/implementation.json')['external_fastwam_source_hashes']
 for name,h in lock.items():
  p=Path(fastwam)/name
  if not p.is_file() or sha(p)!=h:raise ValueError('FastWAM source missing or differs from recorded run: '+name)

def train(release,training,assets,fastwam,output):
 import torch
 from wam_opd_v2.fresh_backend import FreshBackend
 from wam_opd_v2.fresh_runtime import durable_json
 from wam_opd_v2.fastwam_backend import capture_rng
 from .runtime import Runner
 if torch.cuda.device_count()!=2:raise ValueError('expose exactly two GPUs with CUDA_VISIBLE_DEVICES')
 if os.environ.get('FASTWAM_OPD_FP32_ROPE','0')!='0':raise ValueError('recorded recipe requires FASTWAM_OPD_FP32_ROPE=0')
 verify_release(release)
 output=Path(output).resolve()
 if output.exists():raise ValueError('use a new output directory; no automatic resume')
 output.mkdir(parents=True);status=output/'status.json'
 def write(phase,**kw):durable_json(status,dict(status=phase,**kw))
 write('VALIDATING')
 try:
  if shutil.disk_usage(output).free<288*1024**3:raise ValueError('require at least 288 GiB filesystem headroom; also check your site quota')
  m,_=dataset(training,assets);view,q=sampled_view(m);sched=schedule(view)
  if sched!=read(Path(training)/'schedule.json') or sum(r['decision_count'] for r in sched)!=1309:raise ValueError('schedule changed')
  check_fastwam_source(release,fastwam)
  torch.set_num_threads(8)
  class Backend(FreshBackend):
   def score_records(self,role,records,version):
    if self.score_count+len(records)>6148:raise ValueError('fixed scoring budget exceeded')
    self.score_count+=len(records);return super().score_records(role,records,version)
   def save_checkpoint(self,path,payload):
    if shutil.disk_usage(output).free<170*1024**3:raise ValueError('insufficient checkpoint headroom')
    super().save_checkpoint(path,payload)
   def export_model(self,path,*,completed_updates):
    if shutil.disk_usage(output).free<79*1024**3:raise ValueError('insufficient inference export headroom')
    super().export_model(path,completed_updates=completed_updates)
  config=configuration(m,fastwam);backend=Backend(config,m,seed=0);backend.score_count=0
  refs=read(Path(training)/'reference_scores.json');records=[r for ep in m['episodes'] for r in ep['records']]
  selected=[]
  for task in sorted({r['task_name'] for r in records}):selected+=sorted((r for r in records if r['task_name']==task),key=lambda r:r['record_id'])[:2]
  def rng_token():
   import hashlib
   s=capture_rng();h=hashlib.sha256(repr(s['python']).encode());h.update(s['torch'].cpu().numpy().tobytes())
   for t in s['cuda']:h.update(t.cpu().numpy().tobytes())
   if 'numpy' in s:
    v=s['numpy'];h.update(repr((v[0],v[2:])).encode());h.update(v[1].tobytes())
   return h.hexdigest()
  token=rng_token();backend.prime_student()
  if token!=rng_token():raise ValueError('initialization changed caller RNG')
  for role in ('initial','teacher','student'):
   values=backend.score_records(role,selected,0);reference=refs['initial' if role=='student' else role]
   for r in selected:
    if struct.pack('!d',values[r['record_id']])!=struct.pack('!d',reference[r['record_id']]):raise ValueError('reference score differs from the fixed-score manifest')
   if token!=rng_token():raise ValueError('scoring changed caller RNG')
  source_lock=read(Path(release)/'provenance/implementation.json')
  training_files={name:sha(Path(training)/name) for name in ('assets.json','pool.json','reference_scores.json','schedule.json')}
  eid=digest(dict(release_version='0.1.0rc6',source=source_lock,packaged_code={str(p.relative_to(Path(release))):sha(p) for p in sorted((Path(release)/'src').rglob('*.py'))},training_files=training_files,backend=backend.identities,manifest=m,schedule=sched,spec=SPEC))
  subset={role:{rid:refs[role][rid] for rid in q} for role in ('initial','teacher')}
  subset['provenance']=dict(origin='external_training_bundle',sha256=training_files['reference_scores.json'],initial_numeric_preflight='bitwise_passed')
  runner=Runner(view,sched,backend,output,probabilities=q,references=subset,experiment_id=eid)
  durable_json(output/'run_config.json',config);durable_json(output/'identities.json',runner.identities)
  durable_json(output/'training_bundle.json',dict(root=str(Path(training).resolve()),files=training_files))
  write('TRAINING',experiment_id=eid);runner.run();a=runner.query_accounting()
  if runner.completed_updates!=96 or backend.score_count!=6148 or a['committed_queries']!=1309 or a['replayed_queries'] or a['unknown_query_attempts']:raise ValueError('final budget differs')
  write('COMPLETE',experiment_id=eid,updates=96,score_records=6148,query_accounting=a)
  return read(status)
 except BaseException as e:write('FAILED',error=repr(e));raise
