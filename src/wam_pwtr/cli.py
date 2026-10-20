"""Mixed-pool PWTR using explicitly supplied local training metadata and assets."""
import argparse,json
from pathlib import Path
from .io import read,verify_assets,verify_release
from .core import sampled_view,schedule

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-root',type=Path,default=Path('.'),help='code release containing SHA256SUMS and provenance/source locks')
 sub=p.add_subparsers(dest='command',required=True)
 v=sub.add_parser('preview');sub.add_parser('verify-release')
 d=sub.add_parser('doctor');d.add_argument('--asset-root',required=True,type=Path);d.add_argument('--hash-files',action='store_true');d.add_argument('--purpose',choices=('all','train','infer'),default='all',help='check all assets, training inputs, or reference-inference inputs')
 t=sub.add_parser('train');t.add_argument('--asset-root',required=True,type=Path);t.add_argument('--fastwam-root',required=True,type=Path);t.add_argument('--output',required=True,type=Path)
 i=sub.add_parser('infer-record');i.add_argument('--asset-root',required=True,type=Path);i.add_argument('--fastwam-root',required=True,type=Path);i.add_argument('--record-id',required=True);i.add_argument('--output',required=True,type=Path);i.add_argument('--seed',type=int,default=0)
 for command in (v,d,t,i):command.add_argument('--training-root',required=True,type=Path,help='external directory containing assets.json, pool.json, reference_scores.json and schedule.json')
 a=p.parse_args();root=a.release_root.resolve()
 if a.command=='verify-release':result=verify_release(root)
 elif a.command=='preview':
  v,q=sampled_view(read(a.training_root/'pool.json'));s=schedule(v)
  if s!=read(a.training_root/'schedule.json'):raise ValueError('training schedule mismatch')
  result=dict(trajectories=len(v['episodes']),selected_records=len(q),updates=len(s),teacher_queries=sum(r['decision_count'] for r in s),model_queries=0)
 elif a.command=='doctor':
  result=verify_assets(a.training_root,a.asset_root,hash_files=a.hash_files,purpose=a.purpose);print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='READY' else 2)
 elif a.command=='train':
  from .train import train
  result=train(root,a.training_root,a.asset_root,a.fastwam_root,a.output)
 else:
  from .inference import infer_record
  result=infer_record(root,a.training_root,a.asset_root,a.fastwam_root,a.record_id,a.output,a.seed)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
