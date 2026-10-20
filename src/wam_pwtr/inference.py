"""Generate normalized action latents from a saved observation history."""
from pathlib import Path
from .io import dataset,resolve,sha
from .evaluate import check_evaluation_sources

def infer_record(release,training,assets,fastwam,record_id,output,seed=0):
 import torch
 from fastwam_opd.model_loader import load_fastwam_model
 from fastwam_opd.record_io import load_opd_record
 from fastwam_opd.fastwam_adapter import FastWAMActionAdapter,autocast_for
 m,index=dataset(training,assets);check_evaluation_sources(release,fastwam)
 rows=[r for e in m['episodes'] for r in e['records'] if r['record_id']==record_id]
 if len(rows)!=1:raise ValueError('unknown/duplicate record_id')
 row=rows[0];model_spec=index['models']['pwtr_u96'];checkpoint=resolve(assets,model_spec['path'])
 if sha(checkpoint)!=model_spec['sha256'] or sha(row['path'])!=row['sha256']:raise ValueError('model/record hash mismatch')
 out=Path(output)
 if out.exists():raise ValueError('output already exists')
 model=load_fastwam_model(fastwam_root=fastwam,config_name='sim_robotwin.yaml',checkpoint=checkpoint,device='cuda:0',mixed_precision='bf16',load_text_encoder=False,strict_full_checkpoint=True)
 model.eval();adapter=FastWAMActionAdapter(model);record=load_opd_record(row['path'])
 with torch.no_grad(),autocast_for(model):
  path=adapter.collect_student_trajectory(record.condition,num_inference_steps=10,sigma_shift=None,seed=seed,rand_device='cpu')
 out.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(normalized_action_latents=path.latents[:,-1],record_id=record_id,seed=seed,checkpoint_sha256=model_spec['sha256']),out)
 return dict(status='SAVED',output=str(out),shape=list(path.latents[:,-1].shape),scope='normalized action-latent inference on one saved history')
