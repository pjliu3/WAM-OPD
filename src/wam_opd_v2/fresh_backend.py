"""Single-record scoring, resident task teachers, and optimizer-state tracking."""
import hashlib
from pathlib import Path
import time

import torch

from .checkpoint_state_gate import validate_cpu_backend_state
from .fastwam_backend import isolated_evaluation
from .performance_backend import PerformanceBackend
from .runtime import canonical_digest


class FreshBackend(PerformanceBackend):
    def __init__(self, config, manifest, *, seed=0):
        if type(config.get('score_batch_size')) is not int or config['score_batch_size'] != 1:
            raise ValueError('PWTR requires score_batch_size=1')
        self._fresh_primed = False
        self._fresh_updates = 0
        self._parameter_updates = None
        self._teacher_load_count = 0
        self._teacher_load_seconds = 0.
        super().__init__(config, manifest, seed=seed)

    def prime_student(self):
        if not self._fresh_primed:
            # Model construction is isolated from the training/query RNG stream.
            with isolated_evaluation():
                adapter = super()._student()
            named = list(self.student_model.named_parameters())
            by_id = {id(parameter): name for name, parameter in named if parameter.requires_grad}
            actual = [id(parameter) for parameter in self.optimizer.parameters]
            if len(actual) != len(set(actual)) or set(actual) != set(by_id):
                raise ValueError('optimizer parameters must cover trainable model parameters exactly once')
            # FastWAM registers aliases before MoT. Parameter names must follow
            # the actual optimizer objects; never reorder gradients or masters.
            self.parameter_names = [by_id[identity] for identity in actual]
            self._parameter_updates = {name: 0 for name in self.parameter_names}
            self._fresh_primed = True
            return adapter
        return self.student_adapter

    def _student(self):
        return self.prime_student()

    def _new_model(self, checkpoint, device):
        start=time.perf_counter()
        model=super()._new_model(checkpoint,device)
        if str(Path(checkpoint).resolve()) in self._fixed_teacher_paths:
            self._teacher_load_count+=1
            self._teacher_load_seconds+=time.perf_counter()-start
        return model

    def begin_update(self, expected_decisions=16):
        self._update_started=time.perf_counter()
        return super().begin_update(expected_decisions=expected_decisions)

    def finish_update(self):
        names = [name for name, master in zip(self.parameter_names, self.optimizer.masters)
                 if master.grad is not None]
        started=time.perf_counter()
        result = super().finish_update()
        optimizer_seconds=time.perf_counter()-started
        for name in names:
            self._parameter_updates[name] += 1
        self._fresh_updates += 1
        return dict(result,update_seconds=time.perf_counter()-self._update_started,
                    optimizer_step_seconds=optimizer_seconds,
                    teacher_model_load_count=self._teacher_load_count,
                    teacher_model_load_seconds=self._teacher_load_seconds)

    def activity_inventory(self):
        self.prime_student()
        return dict(schema='wam_opd_v2_fresh_activity', completed_updates=self._fresh_updates,
                    parameter_updates=dict(self._parameter_updates),
                    cuda_rng_states=torch.cuda.device_count() if torch.cuda.is_available() else 0)

    def state_dict(self):
        self.prime_student()
        return dict(super().state_dict(), fresh_activity=self.activity_inventory())

    @staticmethod
    def checkpoint_inventory(state):
        """Bind all model/buffer/optimizer metadata and exact saved RNG bytes.

        Tensor contents are protected by the completed file SHA. This separate
        live-before-save inventory detects omissions during serialization.
        """
        import numpy as np
        def visit(value, rng=False):
            if isinstance(value,torch.Tensor):
                item=dict(type='tensor',shape=list(value.shape),dtype=str(value.dtype),
                          stride=list(value.stride()),device=str(value.device),layout=str(value.layout))
                if rng:
                    raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
                    item['sha256']=hashlib.sha256(memoryview(raw)).hexdigest()
                return item
            if isinstance(value,np.ndarray):
                return dict(type='ndarray',shape=list(value.shape),dtype=str(value.dtype),
                            sha256=hashlib.sha256(value.tobytes()).hexdigest())
            if isinstance(value,dict):
                return ['dict',[(type(k).__name__,str(k),visit(v,rng or k=='rng')) for k,v in value.items()]]
            if isinstance(value,(list,tuple)):
                return [type(value).__name__,[visit(v,rng) for v in value]]
            if value is None or type(value) in (bool,int,float,str):return value
            raise ValueError('unsupported checkpoint inventory value')
        return canonical_digest(visit(state))

    def validate_checkpoint_state(self, state, activity, completed_updates):
        if (not isinstance(activity, dict) or set(activity) != {
                'schema', 'completed_updates', 'parameter_updates', 'cuda_rng_states'}
                or activity.get('schema') != 'wam_opd_v2_fresh_activity'
                or type(activity.get('completed_updates')) is not int
                or activity['completed_updates'] != completed_updates
                or state.get('fresh_activity') != activity):
            raise ValueError('fresh optimizer activity inventory differs from complete save receipt')
        # At update 0 every explicitly observed count is zero. The CPU helper's
        # positive bound is 1; a nonzero slot still fails the trusted count check.
        if completed_updates == 0 and any(activity['parameter_updates'].values()):
            raise ValueError('fresh update-zero activity must be empty')
        return validate_cpu_backend_state(state, completed_updates=max(1, completed_updates),
            trusted_parameter_updates=activity['parameter_updates'],
            expected_cuda_devices=activity['cuda_rng_states'])

    def load_state_dict(self, state):
        raise ValueError('PWTR training-state restore is unsupported')

    def load_checkpoint(self, path):
        # Revalidation must not duplicate the entire CPU optimizer allocation.
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
