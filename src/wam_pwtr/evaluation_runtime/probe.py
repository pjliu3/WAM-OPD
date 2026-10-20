"""Import/configuration probe: no model factory, simulator constructor, or CUDA call."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fastwam-root', required=True)
    parser.add_argument('--robotwin-root', required=True)
    parser.add_argument('--dataset-stats', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--expected-dataset-stats', type=json.loads)
    args = parser.parse_args(argv)
    from wam_pwtr.robotwin_policy import preflight_policy
    policy = preflight_policy(args.fastwam_root, args.dataset_stats,
                              expected_dataset_stats=args.expected_dataset_stats)
    root = Path(args.robotwin_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    # Curobo constructs a small default Pose while importing MotionGen. Its
    # default device is CUDA even though no planner is constructed. Restrict
    # that import-only default to CPU in this disposable probe process. The
    # actual evaluation worker never applies this adaptation.
    from curobo.types.base import TensorDeviceType
    original_defaults = TensorDeviceType.__init__.__defaults__
    if not original_defaults or not isinstance(original_defaults[0], torch.device):
        raise RuntimeError('unsupported Curobo import-default interface')
    TensorDeviceType.__init__.__defaults__ = (torch.device('cpu'), *original_defaults[1:])
    try:
        # No SAPIEN engine, renderer, scene, task instance, or planner is made.
        for task in args.tasks:
            module = importlib.import_module('envs.' + task)
            if not isinstance(getattr(module, task), type):
                raise TypeError('RoboTwin task export must be a class')
    finally:
        TensorDeviceType.__init__.__defaults__ = original_defaults
    if torch.cuda.is_initialized():
        raise RuntimeError('zero-model preflight unexpectedly initialized CUDA')
    print(json.dumps(dict(status='RUNTIME_IMPORTS_PASSED', policy=policy,
                          tasks_imported=args.tasks, cuda_initialized=False,
                          cpu_dependency_probe_override='Curobo TensorDeviceType default device=cpu in isolated probe only; real worker unchanged',
                          model_instances_created=0, environment_instances_created=0)))


if __name__ == '__main__':
    main()
