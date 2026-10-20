import copy,json
from pathlib import Path
import pytest
from synthetic_training import manifest
from wam_pwtr.core import sampled_view,schedule,reweight
from wam_opd_v2.runtime import TrainingRunner
ROOT=Path(__file__).resolve().parents[1]

def inputs():
 v,q=sampled_view(manifest());return v,q,schedule(v)

@pytest.mark.parametrize('change',[{'strength':0.},{'seed':1},{'checkpoint_every':24},{'refresh_interval':12},{'probabilities':None},{'reweight':None}])
def test_only_fixed_pwtr_recipe(tmp_path,change):
 v,q,s=inputs()
 class Backend:identities={}
 kwargs=dict(probabilities=q,reweight=reweight);kwargs.update(change)
 with pytest.raises(ValueError):TrainingRunner(v,s,Backend(),tmp_path,**kwargs)

def test_temporal_order_is_required(tmp_path):
 v,q,s=inputs()
 class Backend:identities={}
 s[0]['record_ids'].reverse();s[0]['decision_indices'].reverse()
 with pytest.raises(ValueError):TrainingRunner(v,s,Backend(),tmp_path,probabilities=q,reweight=reweight)

@pytest.mark.parametrize('change',[{'strength':0.},{'eta':1.},{'cap':1.}])
def test_weight_recipe_is_fixed(change):
 v,q,s=inputs();rows=[dict(r,phi_current=0.,phi_initial=0.,phi_teacher=0.) for e in v['episodes'] for r in e['records']]
 with pytest.raises(ValueError):reweight(rows,q,**change)

def test_public_source_has_no_alternate_training_modules():
 for name in ('cycle.py','router.py','lora_reload.py','fresh_context_io.py','fast_rope.py'):
  assert not (ROOT/'src/fastwam_opd'/name).exists()
 assert not (ROOT/'src/wam_opd_v2/sampling.py').exists()
