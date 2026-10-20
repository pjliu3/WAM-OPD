"""Closed-loop execution with strict process and policy-failure separation."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys


def advance_instruction_rng():
    """Match the evaluation protocol's 50-entry instruction-selection RNG step.

    Randomized lighting uses the same NumPy RNG. Advance it by one draw before
    applying the per-scene instruction from the evaluation manifest.
    """
    import numpy as np
    np.random.choice(50)


def run_case(env, policy, case, env_args, *, episode_index, before_rollout=None):
    """Run one supplied seed exactly once; exceptions are never policy failures."""
    try:
        env.setup_demo(now_ep_num=episode_index, seed=case['seed'], is_test=True, **env_args)
        if before_rollout is not None:
            before_rollout()
        env.set_instruction(instruction=case['instruction'])
        policy.reset(case['seed'], case['instruction'])
        if not isinstance(env.step_lim, int) or env.step_lim <= 0:
            raise ValueError('RoboTwin returned an invalid action limit')
        while env.take_action_cnt < env.step_lim:
            observation = env.get_obs()
            actions = policy.infer(observation)
            if len(actions.shape) != 2 or actions.shape[0] != 32 or actions.shape[1] != 14:
                raise ValueError(f'policy must return a [32,14] action chunk, got {actions.shape}')
            for action in actions[:24]:
                count = env.take_action_cnt
                env.take_action(action, action_type='qpos')
                if env.take_action_cnt != count + 1:
                    raise RuntimeError('RoboTwin take_action did not advance exactly one action')
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
            if env.eval_success:
                break
        success = bool(env.eval_success)
        return dict(case, status='success' if success else 'failure', success=success,
                    actions=int(env.take_action_cnt))
    finally:
        # Release this episode even when setup/model/action execution raises.
        env.close_env(clear_cache=((episode_index + 2) % env_args['clear_cache_freq'] == 0))


def environment_arguments(release, robotwin_root, task, condition, output):
    """Resolve the recorded task configuration without loading a simulator."""
    import yaml
    release, robotwin_root = Path(release), Path(robotwin_root)
    def read(path):
        with Path(path).open() as handle:
            return yaml.safe_load(handle)
    task_config = 'demo_' + condition
    args = read(release / 'configs/evaluation' / (task_config + '.yml'))
    config_root = robotwin_root / 'task_config'
    cameras = read(config_root / '_camera_config.yml')
    camera = cameras[args['camera']['head_camera_type']]
    embodiments = read(config_root / '_embodiment_config.yml')
    if args['embodiment'] != ['aloha-agilex']:
        raise ValueError('the paper protocol requires aloha-agilex')
    robot_dir = Path(embodiments['aloha-agilex']['file_path'])
    if not robot_dir.is_absolute():
        robot_dir = robotwin_root / robot_dir
    robot_dir = robot_dir.resolve()
    robot_config = read(robot_dir / 'config.yml')
    args.update(task_name=task, task_config=task_config, ckpt_setting='supplied_checkpoint',
                policy_name='wam_pwtr.robotwin_policy', eval_mode=True, defer_eval_render=True,
                head_camera_h=camera['h'], head_camera_w=camera['w'],
                left_robot_file=str(robot_dir), right_robot_file=str(robot_dir),
                left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                dual_arm_embodied=True, save_path=str(Path(output) / 'unused_data'))
    return args


def main(argv=None):
    """One task/condition worker. The parent rejects every nonzero exit code."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    ns = parser.parse_args(argv)
    request = json.loads(Path(ns.request).read_text())
    output = Path(request['output'])
    os.chdir(request['robotwin_root'])
    sys.path.insert(0, request['robotwin_root'])
    from wam_pwtr.robotwin_policy import create_policy
    from wam_pwtr.evaluate import verify_checkpoint_identity
    cases = request['cases']
    task, condition = cases[0]['task'], cases[0]['condition']
    args = environment_arguments(request['release'], request['robotwin_root'], task, condition, output)
    verify_checkpoint_identity(request['checkpoint'], request['expected_checkpoint_sha256'])
    module = importlib.import_module('envs.' + task)
    task_class = getattr(module, task)
    if not isinstance(task_class, type):
        raise TypeError('RoboTwin task export must be a class')
    policy = create_policy(checkpoint=request['checkpoint'], fastwam_root=request['fastwam_root'],
                           dataset_stats=request['dataset_stats'], text_cache=request['text_cache'],
                           expected_dataset_stats=request['expected_dataset_stats'],
                           expected_text_cache=request['expected_text_cache'],
                           device='cuda:0')
    env = task_class()
    env.suc = env.test_num = 0
    try:
        with (output / 'outcomes.jsonl').open('x') as handle:
            for index, case in enumerate(cases):
                try:
                    row = run_case(env, policy, case, args, episode_index=index,
                                   before_rollout=advance_instruction_rng)
                except BaseException as error:
                    row = dict(case, status='infrastructure_failure', success=None,
                               error=repr(error))
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    raise
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                env.test_num += 1
                env.suc += int(row['success'])
                print(f"{task}/{condition} {index + 1}/{len(cases)}: {row['status']}", flush=True)
        verify_checkpoint_identity(request['checkpoint'], request['expected_checkpoint_sha256'])
    finally:
        close = getattr(policy, 'close', None)
        if callable(close):
            close()
    # Normal interpreter shutdown lets the parent validate the worker exit code,
    # including native runtime cleanup.


if __name__ == '__main__':
    main()
