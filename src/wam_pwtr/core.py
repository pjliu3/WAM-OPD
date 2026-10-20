"""Deterministic endpoint-inclusive temporal sampling and sampled-prefix weights."""
from copy import deepcopy
import random
from wam_opd_v2.weights import reweight_records

SPEC = dict(version='sampled_pwtr_v1', sampling='floor(j*(T-1)/(min(16,T)-1))',
            short='all_unique_no_padding', prefix='exclusive_selected_decisions',
            normalization='task_condition_q', strength=.1, eta=2/3, cap=2.,
            updates=96, refresh=24, seed=0, resume=False, evaluation_episodes=0)


def indices(length):
    if type(length) is not int or length<1:raise ValueError('positive decision count required')
    n=min(16,length)
    return [0] if n==1 else [j*(length-1)//(n-1) for j in range(n)]


def sampled_view(manifest):
    """Construct a selected-decision view while preserving record IDs, indices and paths."""
    view=deepcopy(manifest);q={};n=len(view['episodes'])
    if not n:raise ValueError('empty pool')
    for ep in view['episodes']:
        full=ep['records'];ep['records']=[full[j] for j in indices(len(full))]
        for r in ep['records']:
            if r['record_id'] in q:raise ValueError('duplicate record')
            q[r['record_id']]=1/n/len(ep['records'])
    view['sampling_view']='sampled_pwtr_v1_original_indices_preserved'
    return view,q


def schedule(view,seed=0,updates=96):
    rng=random.Random(seed);out=[]
    for u in range(1,updates+1):
        ep=rng.choice(view['episodes']);records=ep['records']
        out.append(dict(update=u,**{k:ep[k] for k in ('trajectory_id','task_name','condition','behavior_version')},
                        record_ids=[r['record_id'] for r in records],decision_indices=[r['decision_index'] for r in records],
                        decision_count=len(records),replacement=False,duplicate_count=0,duplicate_record_ids=[]))
    return out


def reweight(records,probabilities,*,strength=.1,eta=2/3,cap=2.):
    # The inherited numerical kernel requires consecutive indices. Remap copies
    # ONLY inside weight aggregation; never alter record paths or backend inputs.
    groups={};original={r['record_id']:r for r in records}
    for r in records:groups.setdefault(r['trajectory_id'],[]).append(r)
    mapped=[]
    for rows in groups.values():
        rows=sorted(rows,key=lambda r:r['decision_index'])
        if len({r['decision_index'] for r in rows})!=len(rows):raise ValueError('duplicate decision index')
        for k,r in enumerate(rows):mapped.append(dict(r,decision_index=k,trajectory_length=len(rows)))
    out=reweight_records(mapped,probabilities,strength=strength,eta=eta,cap=cap)
    for r in out:
        orig=original[r['record_id']];r['sampled_rank']=r['decision_index'];r['decision_index']=orig['decision_index']
        if 'trajectory_length' in orig:r['trajectory_length']=orig['trajectory_length']
        else:r.pop('trajectory_length',None)
    lookup={r['record_id']:r for r in out}
    return [lookup[r['record_id']] for r in records]
