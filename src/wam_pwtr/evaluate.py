"""Checkpoint -> fixed RoboTwin task rollouts -> independently validated results.

Importing this module, parsing arguments, and selecting cases do not import
PyTorch or SAPIEN. Only the owned condition worker constructs models/environments.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .io import read, sha, verify_release

TASKS = ('move_stapler_pad', 'open_microwave')
CONDITIONS = ('clean', 'randomized')
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def case_key(case):
    return case['task'], case['condition'], case['seed'], case['noise_id']


def read_evaluation_input(evaluation_root, name):
    path = Path(evaluation_root).expanduser().resolve() / name
    if not path.is_file():
        raise ValueError(f'missing local evaluation bundle input: {path}; '
                         'provide --evaluation-root with your locally prepared evaluation bundle; '
                         'evaluation data and text features are not included in this code release')
    return read(path)


def load_cases(evaluation_root, *, tasks=None, conditions=None, episodes_per_condition=None):
    scenes = read_evaluation_input(evaluation_root, 'evaluation_scenes.json')
    instructions = read_evaluation_input(evaluation_root, 'evaluation_instructions.json')
    if (scenes['episodes'] != 400 or scenes['noise_ids'] != [0] or
        scenes['action_seed_contract'] != 'environment_seed * 1000 + replan_index'):
        raise ValueError('evaluation scene protocol differs from the fixed 400-case panel')
    tasks = TASKS if tasks is None else tuple(tasks)
    conditions = CONDITIONS if conditions is None else tuple(conditions)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError('invalid or repeated task selection')
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError('invalid or repeated condition selection')
    if episodes_per_condition is not None and (type(episodes_per_condition) is not int or not 1 <= episodes_per_condition <= 100):
        raise ValueError('episodes-per-condition must be an integer from 1 to 100')
    cases = []
    # Validate the entire supplied panel even when selecting a smoke subset.
    for task in TASKS:
        for condition in CONDITIONS:
            seeds = scenes['tasks'][task]['seeds'][condition]
            table = instructions[task][condition]
            if (len(seeds) != 100 or len(set(seeds)) != 100 or
                any(type(seed) is not int or seed < 0 for seed in seeds) or
                set(table) != {str(seed) for seed in seeds}):
                raise ValueError(f'invalid fixed seed/instruction coverage: {task}/{condition}')
            for index, seed in enumerate(seeds):
                instruction = table[str(seed)]['instruction']
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError('empty instruction')
                if task in tasks and condition in conditions and index < (episodes_per_condition or 100):
                    cases.append(dict(task=task, condition=condition, seed=seed, noise_id=0, instruction=instruction))
    return cases


def scope(cases):
    return 'supplied_panel_400' if len(cases) == 400 else 'smoke_subset'


def summarize(cases, outcomes):
    expected = {case_key(c): c for c in cases}
    observed = {}
    for row in outcomes:
        if type(row.get('seed')) is not int or type(row.get('noise_id')) is not int:
            raise ValueError('outcome seed/noise keys must be integers, not booleans')
        key = case_key(row)
        if key in observed or key not in expected:
            raise ValueError('duplicate or unexpected outcome key')
        if type(row.get('success')) is not bool:
            raise ValueError('policy success must be a JSON boolean')
        if row.get('status') != ('success' if row['success'] else 'failure'):
            raise ValueError('status/success mismatch or infrastructure failure')
        if row.get('instruction') != expected[key]['instruction']:
            raise ValueError('outcome instruction differs from the fixed manifest')
        if type(row.get('actions')) is not int or row['actions'] < 1:
            raise ValueError('outcome has no valid executed-action count')
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError('incomplete evaluation outcome coverage')
    groups = []
    for task in TASKS:
        for condition in CONDITIONS:
            rows = [r for r in outcomes if (r['task'], r['condition']) == (task, condition)]
            if rows:
                successes = sum(r['success'] for r in rows)
                groups.append(dict(task=task, condition=condition, successes=successes,
                                   trials=len(rows), success_rate=successes / len(rows)))
    successes = sum(r['success'] for r in outcomes)
    return dict(status='COMPLETE', scope=scope(cases), successes=successes, trials=len(outcomes),
                success_rate=successes / len(outcomes), groups=groups,
                infrastructure_failures=0, source='new_closed_loop_rollouts')


def _asset(path, item):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'missing required evaluation asset: {path}; prepare the required local bundle asset; no regeneration is allowed')
    if path.stat().st_size != item['size_bytes'] or sha(path) != item['sha256']:
        raise ValueError(f'evaluation asset size/SHA256 mismatch: {path}')


def _inventory(items):
    result = {}
    for item in items:
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts or item['path'] in result:
            raise ValueError('invalid or duplicate evaluation asset path')
        result[item['path']] = item
    return result


def validate_bundle_inventory(evaluation_root, manifest):
    """Require exact cache identities for every case of the supplied panel."""
    experts, texts = _inventory(manifest['expert_cache']), _inventory(manifest['text_cache'])
    all_cases = load_cases(evaluation_root)
    expected_experts = {f"{case['task']}/demo_{case['condition']}/{case['seed']}.json" for case in all_cases}
    expected_texts = {
        hashlib.sha256(PROMPT.format(task=case['instruction']).encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt'
        for case in all_cases
    }
    if set(experts) != expected_experts or set(texts) != expected_texts:
        raise ValueError('evaluation asset inventory must exactly match all 400 supplied cases and their unique text contexts')
    return experts, texts


def verify_checkpoint_identity(path, expected_sha256):
    """Reject a checkpoint replaced since preflight, before publishing results."""
    try:
        actual = sha(path)
    except OSError as error:
        raise ValueError(f'checkpoint identity changed or became unreadable: {path}') from error
    if actual != expected_sha256:
        raise ValueError(f'checkpoint identity changed since preflight: {path}')


def probe_runtime(args, cases, expected_dataset_stats=None):
    command = [sys.executable, '-m', 'wam_pwtr.evaluation_runtime.probe',
               '--fastwam-root', str(Path(args.fastwam_root).resolve()),
               '--robotwin-root', str(Path(args.robotwin_root).resolve()),
               '--dataset-stats', str(Path(args.dataset_stats).resolve()),
               '--tasks', *dict.fromkeys(c['task'] for c in cases)]
    if expected_dataset_stats is not None:
        command += ['--expected-dataset-stats', json.dumps(expected_dataset_stats)]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(args.release).resolve() / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    # No CUDA device is exposed to the read-only dependency/configuration probe.
    environment['CUDA_VISIBLE_DEVICES'] = ''
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise ValueError('zero-model runtime preflight timed out after 60 seconds') from error
    if result.returncode != 0:
        raise ValueError('zero-model runtime preflight failed; install/configure the required FastWAM and RoboTwin environment:\n' +
                         (result.stderr + '\n' + result.stdout)[-12000:])
    try:
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError('runtime probe did not return a valid completion receipt') from error
    if receipt.get('status') != 'RUNTIME_IMPORTS_PASSED' or receipt.get('cuda_initialized') is not False:
        raise ValueError('invalid zero-model runtime receipt')
    return receipt


def check_evaluation_sources(release, fastwam_root, robotwin_root=None):
    lock = read(Path(release) / 'provenance/evaluation_runtime.json')
    if lock.get('schema') != 'wam_pwtr_evaluation_sources_v1':
        raise ValueError('unsupported evaluation source manifest')
    for label, root in (('fastwam', fastwam_root), ('robotwin', robotwin_root)):
        if root is None:
            continue
        for name, expected in lock[label + '_source_hashes'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid evaluation source path')
            path = Path(root) / relative
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f'{label} evaluation source missing or changed: {name}')


def preflight(args, cases):
    """Read-only validation. Never load checkpoint tensors or create a scene."""
    validate_worker_timeout(args.worker_timeout)
    release = Path(args.release).resolve()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f'checkpoint is missing or empty: {checkpoint}')
    if Path(args.output).expanduser().resolve().exists():
        raise ValueError('output directory already exists; choose a new directory')
    if not args.gpu.strip() or ',' in args.gpu or args.gpu.strip().startswith('-'):
        raise ValueError('--gpu must select one GPU index or GPU UUID')
    if os.environ.get('FASTWAM_OPD_FP32_ROPE', '0') != '0':
        raise ValueError('the recorded protocol requires FASTWAM_OPD_FP32_ROPE=0')
    manifest = read_evaluation_input(evaluation_root, 'evaluation_assets.json')
    if manifest['schema'] != 'wam_pwtr_evaluation_assets_v1' or manifest['prompt_template'] != PROMPT:
        raise ValueError('unsupported evaluation asset manifest/prompt')
    experts, texts = validate_bundle_inventory(evaluation_root, manifest)
    _asset(args.dataset_stats, manifest['dataset_stats'])
    checked_texts = set()
    for case in cases:
        key = f"{case['task']}/demo_{case['condition']}/{case['seed']}.json"
        path = Path(args.expert_cache) / key
        _asset(path, experts[key])
        expert = read(path)
        if (expert.get('status') != 'valid' or expert.get('task_name') != case['task'] or
            expert.get('task_config') != 'demo_' + case['condition'] or expert.get('seed') != case['seed'] or
            not isinstance(expert.get('episode_info', {}).get('info'), dict)):
            raise ValueError(f'invalid expert cache identity: {path}')
        text_name = hashlib.sha256(PROMPT.format(task=case['instruction']).encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt'
        if text_name not in checked_texts:
            _asset(Path(args.text_cache) / text_name, texts[text_name])
            checked_texts.add(text_name)
    dependencies = ('torch', 'numpy', 'yaml', 'hydra', 'omegaconf', 'sapien', 'gymnasium',
                    'toppra', 'transforms3d', 'trimesh', 'imageio', 'cv2', 'PIL', 'einops',
                    'mplib', 'curobo')
    missing = [module for module in dependencies if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError('missing evaluation dependencies in this Python environment: ' + ', '.join(missing))
    verify_release(release)
    check_evaluation_sources(release, args.fastwam_root, args.robotwin_root)
    from .evaluation_runtime.runner import environment_arguments
    for task, condition in dict.fromkeys((c['task'], c['condition']) for c in cases):
        root = Path(args.robotwin_root)
        for name in ('envs/' + task + '.py', 'envs/_base_task.py', 'task_config/_eval_step_limit.yml'):
            if not (root / name).is_file():
                raise ValueError(f'missing RoboTwin runtime file: {root / name}')
        environment_arguments(release, root, task, condition, args.output)
    runtime = probe_runtime(args, cases, manifest['dataset_stats'])
    source_manifest = release / 'provenance/evaluation_runtime.json'
    runtime_files = sorted(read(source_manifest)['robotwin_source_hashes'])
    runtime_hashes = {name: sha(Path(args.robotwin_root) / name) for name in runtime_files}
    return dict(status='PREFLIGHT_PASSED', scope=scope(cases), planned_episodes=len(cases),
                robotwin_source_sha256=runtime_hashes,
                evaluation_sources_manifest_sha256=sha(source_manifest),
                evaluation_root=str(evaluation_root),
                evaluation_assets_manifest_sha256=sha(evaluation_root / 'evaluation_assets.json'),
                evaluation_asset_identity=dict(dataset_stats=manifest['dataset_stats'], text_cache=texts),
                model_instances_created=0, environment_instances_created=0, new_episodes=0,
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                scenes_sha256=sha(evaluation_root / 'evaluation_scenes.json'),
                instructions_sha256=sha(evaluation_root / 'evaluation_instructions.json'),
                expert_caches_verified=len(cases), text_caches_verified=len(checked_texts),
                runtime=runtime, gpu=args.gpu,
                note='No model inference or simulator rollout is performed by preflight; driver/rendering execution is tested only by an actual rollout.')


class WorkerFailure(RuntimeError):
    pass


def validate_worker_timeout(timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('--worker-timeout must be finite and positive')


@contextlib.contextmanager
def _defer_cleanup_signals():
    """Finish killing/reaping owned children before handling another cancel."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {number: signal.getsignal(number) for number in signals}
    exception_in_flight = sys.exc_info()[0] is not None
    pending = []
    def defer(number, frame):
        if not pending:
            pending.append((number, frame))
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        for number in signals:
            signal.signal(number, defer)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    try:
        yield
    finally:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        try:
            for number, handler in previous.items():
                signal.signal(number, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        # An existing cancellation/error continues unwinding unchanged. If the
        # first cancellation arrived during otherwise normal cleanup, replay it.
        if pending and not exception_in_flight:
            number, frame = pending[0]
            handler = previous[number]
            if callable(handler):
                handler(number, frame)
            elif handler == signal.SIG_DFL:
                signal.raise_signal(number)


def run_worker(command, log_path, *, timeout, env=None):
    """Own one child process group and stop it on timeout/interruption."""
    validate_worker_timeout(timeout)
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise WorkerFailure(f'worker timeout after {timeout} seconds; see {log_path}') from exc
            if code != 0:
                raise WorkerFailure(f'worker exit code {code}; see {log_path}')
        finally:
            # A crashed parent can leave a simulator/planner descendant alive.
            # Always clean our new process group, including after parent exit.
            with _defer_cleanup_signals():
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if child.poll() is None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)


def reserve_output(path):
    path = Path(path)
    if path.exists():
        raise ValueError('output directory already exists; no overwrite or automatic resume')
    path.mkdir(parents=True, exist_ok=False)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def evaluate(args, cases, receipt):
    validate_worker_timeout(args.worker_timeout)
    output = Path(args.output).expanduser().resolve()
    reserve_output(output)
    _write_json(output / 'preflight.json', receipt)
    _write_json(output / 'cases.json', cases)
    status = output / 'status.json'
    _write_json(status, dict(status='RUNNING', scope=scope(cases), planned_episodes=len(cases)))
    outcomes = []
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.gpu
    environment['PYTHONPATH'] = str(Path(args.release).resolve() / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    try:
        for task in TASKS:
            for condition in CONDITIONS:
                group = [c for c in cases if (c['task'], c['condition']) == (task, condition)]
                if not group:
                    continue
                folder = output / task / condition
                folder.mkdir(parents=True)
                request = {name: str(Path(getattr(args, name)).expanduser().resolve()) for name in
                           ('release', 'checkpoint', 'fastwam_root', 'robotwin_root', 'dataset_stats', 'text_cache')}
                request.update(cases=group, output=str(folder),
                               expected_checkpoint_sha256=receipt['checkpoint_sha256'],
                               expected_dataset_stats=receipt['evaluation_asset_identity']['dataset_stats'],
                               expected_text_cache=receipt['evaluation_asset_identity']['text_cache'])
                request_path = folder / 'request.json'
                _write_json(request_path, request)
                command = [sys.executable, '-u', '-m', 'wam_pwtr.evaluation_runtime.runner', '--request', str(request_path)]
                run_worker(command, folder / 'worker.log', timeout=args.worker_timeout, env=environment)
                rows = [json.loads(line) for line in (folder / 'outcomes.jsonl').read_text().splitlines()]
                summarize(group, rows)
                outcomes.extend(rows)
        verify_checkpoint_identity(receipt['checkpoint'], receipt['checkpoint_sha256'])
        summary = summarize(cases, outcomes)
        summary.update(checkpoint_sha256=receipt['checkpoint_sha256'], checkpoint=receipt['checkpoint'],
                       completed_at_unix=time.time())
        with (output / 'outcomes.jsonl').open('x') as handle:
            for row in outcomes:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        _write_json(output / 'summary.json', summary)
        with (output / 'summary.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['task', 'condition', 'successes', 'trials', 'success_rate'])
            writer.writeheader()
            writer.writerows(summary['groups'])
            writer.writerow(dict(task='ALL', condition=scope(cases), **{k: summary[k] for k in ('successes', 'trials', 'success_rate')}))
        _write_json(status, summary)
        return summary
    except BaseException as error:
        _write_json(status, dict(status='FAILED', scope=scope(cases), error=repr(error),
                                accepted_completed_episodes=len(outcomes), planned_episodes=len(cases),
                                note='Partial worker files are retained; no automatic retry or resampling.'))
        raise


def parser():
    release = Path(__file__).resolve().parents[2]
    current = Path.cwd()
    if (current / 'configs/evaluation/protocol.json').is_file() and (current / 'SHA256SUMS').is_file():
        release = current
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', default=str(release), help='release repository root')
    p.add_argument('--evaluation-root', required=True,
                   help='local evaluation bundle directory; scene data, expert caches, statistics and text features are not included')
    p.add_argument('--checkpoint', required=True, help='full model checkpoint to evaluate')
    p.add_argument('--fastwam-root', required=True)
    p.add_argument('--robotwin-root', required=True)
    p.add_argument('--text-cache', required=True)
    p.add_argument('--expert-cache', help='defaults to <evaluation-root>/expert_cache')
    p.add_argument('--dataset-stats', help='defaults to <evaluation-root>/evaluation_dataset_stats.json')
    p.add_argument('--output', required=True, help='new output directory; existing paths are rejected')
    p.add_argument('--task', choices=TASKS, nargs='+', dest='tasks')
    p.add_argument('--condition', choices=CONDITIONS, nargs='+', dest='conditions')
    p.add_argument('--gpu', default='0', help='one physical GPU index or UUID exposed to the owned worker')
    p.add_argument('--check-only', action='store_true', help='verify inputs and dependencies without creating model or simulator')
    p.add_argument('--episodes-per-condition', type=int, help='first N supplied scenes in each selected condition; labelled smoke_subset')
    p.add_argument('--worker-timeout', type=float, default=43200, help='seconds per task/condition, then stop without retry')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.expert_cache = args.expert_cache or str(Path(args.evaluation_root).expanduser() / 'expert_cache')
    args.dataset_stats = args.dataset_stats or str(Path(args.evaluation_root).expanduser() / 'evaluation_dataset_stats.json')
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise WorkerFailure('evaluation interrupted by SIGTERM')
    signal.signal(signal.SIGTERM, terminated)
    try:
        cases = load_cases(args.evaluation_root, tasks=args.tasks, conditions=args.conditions,
                           episodes_per_condition=args.episodes_per_condition)
        receipt = preflight(args, cases)
        result = receipt if args.check_only else evaluate(args, cases, receipt)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps(dict(status='FAILED', error=str(error)), ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == '__main__':
    raise SystemExit(main())
