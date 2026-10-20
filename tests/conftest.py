"""Synthetic local bundles for code regression tests; no experimental data."""
import json

import pytest


@pytest.fixture(scope='session')
def synthetic_evaluation_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_evaluation_bundle')
    tasks = {}
    instructions = {}
    for task_index, task in enumerate(('move_stapler_pad', 'open_microwave')):
        tasks[task] = {'seeds': {}}
        instructions[task] = {}
        for condition_index, condition in enumerate(('clean', 'randomized')):
            seeds = [8000000 + 1000 * task_index + 100 * condition_index + index
                     for index in range(100)]
            tasks[task]['seeds'][condition] = seeds
            instructions[task][condition] = {
                str(seed): {'instruction': f'Synthetic test instruction {task_index}-{condition_index}-{index}.'}
                for index, seed in enumerate(seeds)
            }
    scenes = dict(episodes=400, noise_ids=[0], tasks=tasks,
                  action_seed_contract='environment_seed * 1000 + replan_index')
    (root / 'evaluation_scenes.json').write_text(json.dumps(scenes))
    (root / 'evaluation_instructions.json').write_text(json.dumps(instructions))
    return root
