"""PWTR training from initialization with validated checkpoints.

Each sampled trajectory contributes one optimizer update. Initial and final
checkpoints include the model and optimizer state and are validated by readback.
"""
import copy
import json
import os
from pathlib import Path
import random
import re

from .runtime import TrainingRunner, atomic_json, canonical_digest, checked_scores, file_digest


def durable_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class FreshTrainingRunner(TrainingRunner):
    @property
    def identities(self):
        return self._identities

    @identities.setter
    def identities(self, base):
        # Parent __init__ creates/validates the ledger after this assignment.
        # Include fresh recipe fields before that first durable identity event.
        self._identities = dict(base, fresh=copy.deepcopy(self.fresh_identity),
            recipe=canonical_digest(dict(base_recipe=base['recipe'], **self.fresh_identity)))

    def __init__(self, *args, experiment_id, arm_id, weighting_spec,
                 verified_references=None, **kwargs):
        if (not isinstance(experiment_id, str) or re.fullmatch('[0-9a-f]{64}', experiment_id) is None
                or not isinstance(arm_id, str) or re.fullmatch('[a-zA-Z0-9_-]+', arm_id) is None
                or not isinstance(weighting_spec, dict) or not weighting_spec):
            raise ValueError('fresh experiment, arm and complete weighting specification are required')
        self.fresh_identity = dict(experiment_id=experiment_id, arm_id=arm_id,
            weighting_spec=copy.deepcopy(weighting_spec), weighting_spec_sha256=canonical_digest(weighting_spec))
        if verified_references is None:
            raise ValueError("PWTR requires verified fixed references; reference recomputation is unsupported")
        self._provided_references = copy.deepcopy(verified_references)
        super().__init__(*args, **kwargs)

    def prepare_references(self):
        if self.references is not None:
            return self.references
        supplied = self._provided_references
        if (not isinstance(supplied, dict) or set(supplied) != {'initial','teacher','provenance'}
                or not isinstance(supplied['provenance'], dict) or not supplied['provenance']):
            raise ValueError('PWTR requires the verified fixed references and their provenance')
        self.references = dict(initial=checked_scores(supplied['initial'],self.records),
            teacher=checked_scores(supplied['teacher'],self.records),
            provenance=copy.deepcopy(supplied['provenance']))
        return self.references

    def refresh_weights(self):
        self.backend.prime_student()
        refs = self.prepare_references()
        scores = checked_scores(self.backend.score_records('student',self.records,self.completed_updates),self.records)
        rows = [dict(row,phi_current=scores[row['record_id']],phi_initial=refs['initial'][row['record_id']],
                     phi_teacher=refs['teacher'][row['record_id']]) for row in self.records]
        weighted = self.reweight(rows,self.probabilities,strength=self.strength,eta=2/3,cap=2.)
        if len(weighted) != len(self.records):
            raise ValueError('weighted rows must cover the selected-point inventory')
        weights = checked_scores({r['record_id']:r['weight'] for r in weighted},self.records)
        if any(w < 0 for w in weights.values()):raise ValueError('negative PWTR weight')
        self.snapshot = dict(version=self.completed_updates,scores=scores,weights=weights,
            weighted_records=weighted,identities=self.identities,**copy.deepcopy(self.fresh_identity))
        atomic_json(self.run_dir/f'weights_{self.completed_updates:06d}.json',self.snapshot)

    def _validate_state(self, state):
        done = state.get('completed_updates')
        if (state.get('format') != 'wam_opd_v2_checkpoint' or state.get('version') != 1
                or state.get('fresh') != self.fresh_identity or state.get('identities') != self.identities
                or type(done) is not int or not 0 <= done <= len(self.schedule)
                or state.get('next_update') != done+1 or state.get('schedule') != self.schedule
                or state.get('manifest') != self.manifest):
            raise ValueError('fresh checkpoint identity, schedule or update differs')
        n=state.get('query_ledger_length')
        if (type(n) is not int or n<1 or n>len(self.ledger_events)
                or canonical_digest(self.ledger_events[:n]) != state.get('query_ledger_prefix_sha256')):
            raise ValueError('fresh checkpoint query ledger is incomplete or changed')
        rng=random.Random(self.training_seed);expected={}
        for row in self.schedule[:done]:
            for slot,rid in enumerate(row['record_ids']):expected[(row['update'],slot)]=(rid,rng.randrange(0,2**63))
        if state.get('training_rng') != rng.getstate():raise ValueError('fresh training RNG differs from schedule')
        attempts={};completed=set()
        for event in self.ledger_events[1:n]:
            if event['event']=='started':
                key=(event['update'],event['decision_slot'])
                if expected.get(key)!=(event['record_id'],event['query_seed']):raise ValueError('query prefix differs')
                attempts[event['attempt']]=key
            elif event['event']=='completed':completed.add(attempts[event['attempt']])
        if completed != set(expected):raise ValueError('fresh checkpoint lacks completed query prefix')
        snapshot=state.get('snapshot')
        if (not isinstance(snapshot,dict) or snapshot.get('identities')!=self.identities
                or snapshot.get('version')!=(max(1,done)-1)//24*24
                or any(snapshot.get(k)!=v for k,v in self.fresh_identity.items())):
            raise ValueError('fresh weight snapshot is stale or belongs to another specification')
        weights=checked_scores(snapshot['weights'],self.records)
        if any(w<0 for w in weights.values()):raise ValueError('negative checkpoint weight')
        refs=state.get('references')
        checked_scores(snapshot['scores'],self.records)
        if not isinstance(refs,dict) or not isinstance(refs.get('provenance'),dict):
            raise ValueError('weighted checkpoint lacks fixed-reference provenance')
        for role in ('initial','teacher'):checked_scores(refs[role],self.records)
        if refs!=self._provided_references:
            raise ValueError('saved fixed references differ from verified input')
        return done

    @staticmethod
    def _receipt_path(path):
        return Path(path).with_name(Path(path).name+'.receipt.json')

    def checkpoint(self):
        if self.completed_updates not in (0,96):
            raise ValueError("PWTR saves only the complete initial and final states")
        state=dict(format='wam_opd_v2_checkpoint',version=1,identities=self.identities,
            next_update=self.completed_updates+1,completed_updates=self.completed_updates,
            schedule=self.schedule,manifest=self.manifest,references=self.references,snapshot=self.snapshot,
            query_ledger_length=len(self.ledger_events),query_ledger_prefix_sha256=canonical_digest(self.ledger_events),
            training_rng=self.rng.getstate(),backend=self.backend.state_dict(),fresh=copy.deepcopy(self.fresh_identity))
        activity=copy.deepcopy(self.backend.activity_inventory())
        inventory=self.backend.checkpoint_inventory(state['backend'])
        self._validate_state(state)
        self.backend.validate_checkpoint_state(state['backend'],activity,self.completed_updates)
        target=self.run_dir/'checkpoint.pt';temporary=target.with_name(f'.{target.name}.{os.getpid()}.tmp')
        self.backend.save_checkpoint(temporary,state)
        with temporary.open('r+b') as stream:os.fsync(stream.fileno())
        digest=file_digest(temporary)
        restored=self.backend.load_checkpoint(temporary)
        self._validate_state(restored)
        self.backend.validate_checkpoint_state(restored['backend'],activity,self.completed_updates)
        if self.backend.checkpoint_inventory(restored['backend'])!=inventory:
            raise ValueError('saved model/buffer/optimizer/RNG inventory differs from live state')
        del restored,state
        receipt=dict(schema='wam_opd_v2_fresh_complete_save',status='COMPLETE',
            checkpoint=str(target.absolute()),checkpoint_sha256=digest,checkpoint_bytes=temporary.stat().st_size,
            completed_updates=self.completed_updates,identities_sha256=canonical_digest(self.identities),
            fresh=copy.deepcopy(self.fresh_identity),activity=activity,
            backend_inventory_sha256=inventory,
            query_ledger_length=len(self.ledger_events),query_ledger_prefix_sha256=canonical_digest(self.ledger_events))
        receipt['receipt_sha256']=canonical_digest(receipt)
        os.replace(temporary,target)
        durable_json(self._receipt_path(target),receipt)
        self.checkpoint_updates=self.completed_updates
        self._committed_source=target

    def _complete_save(self,path):
        path=Path(path);receipt_path=self._receipt_path(path)
        if not path.is_file() or not receipt_path.is_file():raise ValueError('complete checkpoint receipt is missing')
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('schema')!='wam_opd_v2_fresh_complete_save' or receipt.get('status')!='COMPLETE'
                or receipt.get('receipt_sha256')!=canonical_digest({k:v for k,v in receipt.items() if k!='receipt_sha256'})
                or receipt.get('identities_sha256')!=canonical_digest(self.identities)
                or receipt.get('fresh')!=self.fresh_identity
                or path.stat().st_size!=receipt.get('checkpoint_bytes')
                or file_digest(path)!=receipt.get('checkpoint_sha256')):
            raise ValueError('complete checkpoint receipt/hash/identity differs')
        try:same_file=os.path.samefile(receipt.get('checkpoint',''),path)
        except (OSError,ValueError,TypeError):same_file=False
        if not same_file:raise ValueError('checkpoint path is not the saved physical file')
        return receipt

    def resume(self, path):
        raise ValueError('PWTR runs from the initial model; checkpoint resume is unsupported')

    def run(self, *, stop_after=None):
        if stop_after is not None:
            raise ValueError('PWTR executes the fixed 96-update schedule without partial-run exports')
        if self._run_started:
            raise ValueError('PWTR run already attempted; restart is unsupported')
        try:
            self.backend.prime_student()
        except BaseException:
            self._run_started = True
            raise
        return super().run()

    def export_for_evaluation(self):
        raise NotImplementedError('the public PWTR runner owns the final hardlink export')
