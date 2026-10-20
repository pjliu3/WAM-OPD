"""CPU structural checks against an independently trusted activity inventory.

The inventory maps *every* trainable name to the number of optimizer steps in
which it actually had a gradient. It must come from evidence independent of the
checkpoint under inspection. This module does not create or authenticate that
evidence, and cannot prove whole-checkpoint provenance or tensor correctness.
The fresh PWTR backend records this inventory before each complete save.
"""
import math
import random


def validate_cpu_backend_state(state, *, completed_updates,
                               trusted_parameter_updates=None,
                               expected_cuda_devices):
    """Check structure/activity only; never initialize CUDA or restore RNG.

    A successful result is conditional on the caller's trusted inventory. It is
    not a provenance certificate or permission to resume training.
    """
    import numpy as np
    import torch

    def require(test, message):
        if not test:
            raise ValueError(message)

    def cpu_tensor(value, *, shape=None, dtype=None):
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.layout == torch.strided, 'state tensor must be dense CPU data')
        require(shape is None or tuple(value.shape) == tuple(shape), 'state tensor shape differs')
        require(dtype is None or value.dtype == dtype, 'state tensor dtype differs')
        require(bool(torch.isfinite(value).all()), 'state tensor contains nonfinite data')

    require(type(completed_updates) is int and completed_updates > 0, 'invalid completed update count')
    require(type(expected_cuda_devices) is int and expected_cuda_devices >= 0, 'invalid CUDA RNG inventory')
    require(isinstance(state, dict) and {'model','optimizer','rng','trainable_names'} <= set(state),
            'backend model/optimizer/RNG inventory is incomplete')
    names = state['trainable_names']; model = state['model']; optimizer = state['optimizer']
    require(isinstance(names, (list,tuple)) and bool(names) and all(type(n) is str for n in names)
            and len(names) == len(set(names)), 'trainable name inventory is invalid')
    require(isinstance(model, dict) and set(names) <= set(model), 'trainable model tensor is missing')
    require(isinstance(optimizer, dict) and {'cpu_offload','masters','adamw'} <= set(optimizer)
            and type(optimizer['cpu_offload']) is bool, 'optimizer fields are missing')
    masters = optimizer['masters'] if optimizer['cpu_offload'] else [model[n] for n in names]
    require(isinstance(masters,(list,tuple)) and len(masters) == len(names), 'master inventory differs')
    if not optimizer['cpu_offload']:
        require(optimizer['masters'] is None, 'non-offloaded optimizer must not contain masters')
    for name, master in zip(names, masters):
        cpu_tensor(model[name])
        require(model[name].dtype in (torch.float32,torch.bfloat16), 'unexpected forward parameter dtype')
        cpu_tensor(master, shape=model[name].shape, dtype=torch.float32)
        require(torch.equal(master.to(dtype=model[name].dtype),model[name]),
                'forward parameter differs from its stored FP32 master')

    adam = optimizer['adamw']
    require(isinstance(adam,dict) and set(adam) == {'state','param_groups'}, 'Adam fields are invalid')
    groups = adam['param_groups']; slots = adam['state']
    require(isinstance(groups,list) and len(groups) == 1 and isinstance(groups[0],dict),
            'Adam parameter groups differ from the recipe')
    group = groups[0]; ids = group.get('params')
    require(isinstance(ids,list) and len(ids) == len(names) and all(type(i) is int for i in ids)
            and len(ids) == len(set(ids)), 'Adam parameter inventory differs')
    require(group.get('lr') == 1e-6 and tuple(group.get('betas',())) == (.9,.95)
            and group.get('weight_decay') == 0 and group.get('eps') == 1e-8
            and group.get('amsgrad') is False and group.get('foreach') is False
            and group.get('maximize') is False, 'Adam hyperparameters differ from the recipe')
    require(isinstance(slots,dict) and all(type(i) is int for i in slots) and set(slots) <= set(ids),
            'Adam state references an unknown parameter')
    observed = {}
    for name, pid, master in zip(names,ids,masters):
        if pid not in slots:
            observed[name] = 0
            continue
        slot = slots[pid]
        require(isinstance(slot,dict) and set(slot) == {'step','exp_avg','exp_avg_sq'},
                'Adam moment/step inventory is incomplete')
        cpu_tensor(slot['step'],shape=(),dtype=torch.float32)
        step = float(slot['step'])
        require(math.isfinite(step) and step.is_integer() and 0 < step <= completed_updates,
                'Adam step is invalid')
        for key in ('exp_avg','exp_avg_sq'):
            cpu_tensor(slot[key],shape=master.shape,dtype=torch.float32)
        require(bool((slot['exp_avg_sq'] >= 0).all()), 'Adam variance contains negative values')
        observed[name] = int(step)

    rng = state['rng']
    require(isinstance(rng,dict) and set(rng) == {'python','numpy','torch','cuda'},
            'complete Python/NumPy/Torch/CUDA RNG fields are required')
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState(0).set_state(rng['numpy'])
        cpu_tensor(rng['torch'],dtype=torch.uint8)
        require(rng['torch'].ndim == 1, 'invalid CPU Torch RNG state')
        torch.Generator(device='cpu').set_state(rng['torch'])
    except (TypeError,RuntimeError,KeyError) as error:
        raise ValueError('invalid CPU random state') from error
    require(isinstance(rng['cuda'],list) and len(rng['cuda']) == expected_cuda_devices,
            'CUDA RNG state inventory is incomplete')
    for value in rng['cuda']:
        cpu_tensor(value,dtype=torch.uint8)
        require(value.ndim == 1 and value.numel() >= 8, 'invalid CUDA RNG byte state')

    require(trusted_parameter_updates is not None,
            'no independent optimizer activity inventory; completeness cannot be proved')
    require(isinstance(trusted_parameter_updates,dict) and set(trusted_parameter_updates) == set(names)
            and all(type(n) is int and 0 <= n <= completed_updates
                    for n in trusted_parameter_updates.values()), 'trusted activity inventory is invalid')
    require(observed == trusted_parameter_updates, 'Adam state/step differs from trusted activity inventory')
    return dict(structure_valid=True, optimizer_activity_matches_trusted_inventory=True,
                complete_anchor_proven=False, active_parameters=sum(n > 0 for n in observed.values()),
                unused_parameters=sum(n == 0 for n in observed.values()))
