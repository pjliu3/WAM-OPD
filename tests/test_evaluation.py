"""CPU contract tests for rollout orchestration, outcome validation and process cleanup."""
import json
import hashlib
import os
import signal
import subprocess
import time
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

from wam_pwtr import evaluate
from wam_pwtr.evaluation_runtime.runner import run_case

ROOT = Path(__file__).resolve().parents[1]


def test_panel_has_exact_400_cases_and_subset_preserves_order(synthetic_evaluation_root):
    full = evaluate.load_cases(synthetic_evaluation_root)
    small = evaluate.load_cases(synthetic_evaluation_root, episodes_per_condition=2)
    assert len(full) == 400
    assert len({evaluate.case_key(c) for c in full}) == 400
    assert len(small) == 8
    for task in evaluate.TASKS:
        for condition in evaluate.CONDITIONS:
            assert [c for c in small if (c['task'], c['condition']) == (task, condition)] == [
                c for c in full if (c['task'], c['condition']) == (task, condition)][:2]
    assert evaluate.scope(full) == 'supplied_panel_400'
    assert evaluate.scope(small) == 'smoke_subset'


@pytest.mark.parametrize('n', [0, -1, 101, True])
def test_invalid_subset_rejected(n, synthetic_evaluation_root):
    with pytest.raises(ValueError):
        evaluate.load_cases(synthetic_evaluation_root, episodes_per_condition=n)


def test_summary_requires_exact_keys_and_boolean_success(synthetic_evaluation_root):
    cases = evaluate.load_cases(synthetic_evaluation_root, tasks=['open_microwave'], episodes_per_condition=1)
    rows = [dict(c, status='success' if i == 0 else 'failure', success=i == 0, actions=24)
            for i, c in enumerate(cases)]
    result = evaluate.summarize(cases, rows)
    assert (result['successes'], result['trials'], result['success_rate']) == (1, 2, .5)
    assert result['scope'] == 'smoke_subset'
    for bad in [rows[:1], rows + rows[:1], [dict(rows[0], seed=99), rows[1]],
                [dict(rows[0], success=1), rows[1]], [dict(rows[0], status='failure'), rows[1]],
                [dict(rows[0], noise_id=False), rows[1]]]:
        with pytest.raises(ValueError):
            evaluate.summarize(cases, bad)


class Array:
    """Tiny action matrix to keep orchestration tests independent of NumPy."""
    shape = (32, 14)

    def __getitem__(self, key):
        return [[float(i)] * 14 for i in range(32)][key]


class Env:
    def __init__(self, success_at=None):
        self.take_action_cnt = 0
        self.step_lim = 50
        self.eval_success = False
        self.success_at = success_at
        self.observations = []
        self.actions = []
        self.closed = False

    def setup_demo(self, **kw):
        self.setup = kw

    def set_instruction(self, *, instruction):
        self.instruction = instruction

    def get_obs(self):
        self.observations.append(self.take_action_cnt)
        return {'step': self.take_action_cnt}

    def take_action(self, action, action_type):
        assert action_type == 'qpos'
        self.actions.append(action[0])
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.success_at

    def close_env(self, clear_cache=False):
        self.closed = True


class Policy:
    def reset(self, environment_seed, instruction):
        self.reset_args = (environment_seed, instruction)

    def infer(self, observation):
        return Array()


def test_closed_loop_reobserves_after_24_actions_and_stops_at_limit(synthetic_evaluation_root):
    case = evaluate.load_cases(synthetic_evaluation_root, episodes_per_condition=1)[0]
    env, policy = Env(), Policy()
    row = run_case(env, policy, case, {'clear_cache_freq': 5}, episode_index=0)
    assert env.observations == [0, 24, 48]
    assert env.actions == list(map(float, range(24))) * 2 + [0., 1.]
    assert policy.reset_args == (case['seed'], case['instruction'])
    assert env.setup['seed'] == case['seed'] and env.instruction == case['instruction']
    assert row['status'] == 'failure' and row['success'] is False
    assert row['actions'] == 50 and env.closed


def test_closed_loop_early_success_and_exception_are_different(synthetic_evaluation_root):
    case = evaluate.load_cases(synthetic_evaluation_root, episodes_per_condition=1)[0]
    env = Env(success_at=3)
    result = run_case(env, Policy(), case, {'clear_cache_freq': 5}, episode_index=0)
    assert result['success'] is True and len(env.actions) == 3 and env.closed
    class BrokenPolicy(Policy):
        def infer(self, observation):
            raise RuntimeError('model unavailable')
    env = Env()
    with pytest.raises(RuntimeError, match='model unavailable'):
        run_case(env, BrokenPolicy(), case, {'clear_cache_freq': 5}, episode_index=0)
    assert env.closed and not env.actions


def test_child_nonzero_with_results_is_still_failure(tmp_path):
    result = tmp_path / 'outcomes.jsonl'
    code = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}\\n'); sys.exit(7)"
    with pytest.raises(evaluate.WorkerFailure, match='exit code 7'):
        evaluate.run_worker([sys.executable, '-c', code, str(result)], tmp_path / 'log', timeout=5)
    assert result.exists()


def test_child_timeout_stops_owned_process(tmp_path):
    with pytest.raises(evaluate.WorkerFailure, match='timeout'):
        evaluate.run_worker([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path / 'log', timeout=.05)


@pytest.mark.parametrize('cancel_signal', ['SIGTERM', 'SIGINT'])
def test_repeated_cancellation_cannot_interrupt_worker_cleanup(tmp_path, cancel_signal):
    pidfile, marker = tmp_path / 'child.pid', tmp_path / 'cleanup.started'
    child_code = '''import os,signal,sys,time
from pathlib import Path
def resist(signum,frame): Path(sys.argv[2]).write_text('received termination')
signal.signal(signal.SIGTERM,resist)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)
'''
    parent_code = '''import json,signal,sys
from pathlib import Path
from wam_pwtr.evaluate import run_worker,WorkerFailure
def stop(signum,frame): raise WorkerFailure('first SIGTERM cancellation')
signal.signal(signal.SIGTERM,stop)
original={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
try:
 run_worker([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],Path(sys.argv[4]),timeout=60)
except BaseException as error:
 print(json.dumps({'exception':type(error).__name__,'message':str(error),
                   'handlers_restored':all(signal.getsignal(s)==h for s,h in original.items())}),flush=True)
'''
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    parent = subprocess.Popen([sys.executable, '-B', '-c', parent_code, child_code,
                               str(pidfile), str(marker), str(tmp_path / 'worker.log')],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    child_pid = None
    def wait_for(path):
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert path.exists(), parent.poll()
    try:
        wait_for(pidfile)
        child_pid = int(pidfile.read_text())
        number = getattr(signal, cancel_signal)
        os.kill(parent.pid, number)
        wait_for(marker)  # The first cancellation has entered the cleanup wait.
        os.kill(parent.pid, number)
        stdout, stderr = parent.communicate(timeout=15)
        result = json.loads(stdout)
        assert result['handlers_restored'], stderr
        assert result['exception'] == ('WorkerFailure' if cancel_signal == 'SIGTERM' else 'KeyboardInterrupt')
        if cancel_signal == 'SIGTERM':
            assert result['message'] == 'first SIGTERM cancellation'
        stat = Path(f'/proc/{child_pid}/stat')
        assert not stat.exists() or stat.read_text().split()[2] == 'Z', 'owned child survived repeated cancellation'
    finally:
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if parent.poll() is None:
            parent.kill()
            parent.wait()


@pytest.mark.parametrize('number', [signal.SIGTERM, signal.SIGINT])
def test_first_cancellation_during_cleanup_is_replayed_and_handler_restored(number):
    previous = signal.getsignal(number)
    finished = []
    class FirstCancellation(Exception):
        pass
    def cancel(signum, frame):
        raise FirstCancellation('first cancellation')
    signal.signal(number, cancel)
    try:
        with pytest.raises(FirstCancellation, match='first cancellation'):
            with evaluate._defer_cleanup_signals():
                signal.raise_signal(number)
                finished.append(True)
        assert finished == [True]
        assert signal.getsignal(number) is cancel
    finally:
        signal.signal(number, previous)


def test_failed_worker_cleans_its_remaining_process_group(tmp_path):
    pidfile = tmp_path / 'child.pid'
    code = ("import subprocess,sys; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); sys.exit(7)")
    pid = None
    try:
        with pytest.raises(evaluate.WorkerFailure):
            evaluate.run_worker([sys.executable, '-c', code, str(pidfile)], tmp_path / 'log', timeout=5)
        pid = int(pidfile.read_text())
        # An orphan awaiting init's reaping is already stopped if it is a zombie.
        stat = Path(f'/proc/{pid}/stat')
        for _ in range(50):
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_output_directory_is_never_reused(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        evaluate.reserve_output(output)


def test_cli_help_is_cpu_only():
    parser = evaluate.parser()
    args = parser.parse_args(['--evaluation-root', '/local-evaluation-bundle', '--checkpoint', '/model.pt', '--fastwam-root', '/fastwam',
                              '--robotwin-root', '/robotwin', '--text-cache', '/texts',
                              '--output', '/output', '--check-only', '--episodes-per-condition', '2'])
    assert args.check_only and args.episodes_per_condition == 2
    assert args.gpu == '0'


@pytest.mark.parametrize('timeout', [float('nan'), float('inf'), -float('inf'), 0., -1.])
def test_invalid_timeout_is_rejected_before_output_or_child(tmp_path, timeout):
    args = SimpleNamespace(output=str(tmp_path / 'output'), worker_timeout=timeout)
    with pytest.raises(ValueError, match='finite and positive'):
        evaluate.evaluate(args, [], {})
    assert not Path(args.output).exists()
    with pytest.raises(ValueError, match='finite and positive'):
        evaluate.run_worker([sys.executable, '-c', 'raise AssertionError("must not run")'],
                            tmp_path / 'worker.log', timeout=timeout)
    assert not (tmp_path / 'worker.log').exists()


def test_missing_and_corrupt_cache_are_rejected_without_regeneration(tmp_path):
    path = tmp_path / 'cache.pt'
    expected = dict(size_bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
    with pytest.raises(ValueError, match='missing required evaluation asset'):
        evaluate._asset(path, expected)
    assert not path.exists()
    path.write_bytes(b'xyz')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        evaluate._asset(path, expected)
    assert path.read_bytes() == b'xyz'
    path.write_bytes(b'abc')
    evaluate._asset(path, expected)


def test_manifest_rejects_paths_outside_cache_and_duplicate_entries():
    for entries in [[{'path': '../escape.pt'}], [{'path': '/escape.pt'}],
                    [{'path': 'cache.pt'}, {'path': 'cache.pt'}]]:
        with pytest.raises(ValueError):
            evaluate._inventory(entries)


def test_bundle_inventory_matches_supplied_cases_and_texts(synthetic_evaluation_root):
    cases = evaluate.load_cases(synthetic_evaluation_root)
    manifest = dict(
        expert_cache=[dict(path=f"{c['task']}/demo_{c['condition']}/{c['seed']}.json") for c in cases],
        text_cache=[dict(path=hashlib.sha256(evaluate.PROMPT.format(task=c['instruction']).encode()).hexdigest()
                        + '.t5_len128.wan22ti2v5b.pt') for c in cases])
    experts, texts = evaluate.validate_bundle_inventory(synthetic_evaluation_root, manifest)
    assert len(experts) == len(texts) == 400
    for field in ('expert_cache', 'text_cache'):
        missing = dict(manifest, **{field: manifest[field][1:]})
        extra = dict(manifest, **{field: manifest[field] + [dict(path='unrequested-artifact')]})
        for invalid in (missing, extra):
            with pytest.raises(ValueError, match='exactly match'):
                evaluate.validate_bundle_inventory(synthetic_evaluation_root, invalid)


def test_bundle_inventory_accepts_shared_synthetic_instruction(tmp_path, synthetic_evaluation_root):
    root = tmp_path / 'shared-instruction-panel'
    root.mkdir()
    (root / 'evaluation_scenes.json').write_bytes((synthetic_evaluation_root / 'evaluation_scenes.json').read_bytes())
    instructions = json.loads((synthetic_evaluation_root / 'evaluation_instructions.json').read_text())
    for task in instructions.values():
        for condition in task.values():
            for entry in condition.values():
                entry['instruction'] = 'One synthetic test instruction shared across all cases.'
    (root / 'evaluation_instructions.json').write_text(json.dumps(instructions))
    cases = evaluate.load_cases(root)
    name = hashlib.sha256(evaluate.PROMPT.format(task=cases[0]['instruction']).encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt'
    manifest = dict(expert_cache=[dict(path=f"{c['task']}/demo_{c['condition']}/{c['seed']}.json") for c in cases],
                    text_cache=[dict(path=name)])
    experts, texts = evaluate.validate_bundle_inventory(root, manifest)
    assert len(experts) == 400 and list(texts) == [name]


def test_instruction_rng_matches_legacy_50_description_draw():
    np = pytest.importorskip('numpy')
    from wam_pwtr.evaluation_runtime.runner import advance_instruction_rng
    np.random.seed(731)
    np.random.choice(['description ' + str(i) for i in range(50)])
    expected = np.random.get_state()
    np.random.seed(731)
    advance_instruction_rng()
    actual = np.random.get_state()
    assert expected[0] == actual[0]
    assert np.array_equal(expected[1], actual[1])
    assert expected[2:] == actual[2:]


def test_clean_randomized_configs_keep_recorded_camera_and_randomization(tmp_path):
    yaml = pytest.importorskip('yaml')
    from wam_pwtr.evaluation_runtime.runner import environment_arguments
    configs = tmp_path / 'task_config'
    configs.mkdir()
    (configs / '_camera_config.yml').write_text('D435: {h: 240, w: 320}\n')
    (configs / '_embodiment_config.yml').write_text('aloha-agilex: {file_path: ./assets/robot}\n')
    robot = tmp_path / 'assets/robot'
    robot.mkdir(parents=True)
    (robot / 'config.yml').write_text('arm_joints_name: [[a], [b]]\n')
    clean = environment_arguments(ROOT, tmp_path, 'move_stapler_pad', 'clean', tmp_path / 'out')
    rand = environment_arguments(ROOT, tmp_path, 'move_stapler_pad', 'randomized', tmp_path / 'out')
    assert clean['left_robot_file'] == str(robot)
    assert clean['head_camera_h'] == 240 and clean['head_camera_w'] == 320
    assert clean['camera'] == rand['camera']
    assert clean['domain_randomization']['random_light'] is False
    assert rand['domain_randomization']['random_light'] is True
    assert rand['domain_randomization']['crazy_random_light_rate'] == .02
    assert clean['eval_mode'] is True and rand['defer_eval_render'] is True
    assert clean['eval_video_log'] is False and rand['render_freq'] == 0


def test_worker_rejects_replaced_checkpoint_before_loading(tmp_path, monkeypatch, synthetic_evaluation_root):
    from wam_pwtr.evaluation_runtime import runner
    from wam_pwtr import robotwin_policy
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'original weights')
    expected = evaluate.sha(checkpoint)
    checkpoint.write_bytes(b'replaced weights')
    request = dict(release=str(ROOT), robotwin_root=str(tmp_path), output=str(tmp_path),
                   expected_dataset_stats={}, expected_text_cache={},
                   checkpoint=str(checkpoint), expected_checkpoint_sha256=expected,
                   fastwam_root=str(tmp_path), dataset_stats=str(tmp_path), text_cache=str(tmp_path),
                   cases=evaluate.load_cases(synthetic_evaluation_root, tasks=['move_stapler_pad'], conditions=['clean'], episodes_per_condition=1))
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'environment_arguments', lambda *args: {'clear_cache_freq': 5})
    def forbidden_load(**kwargs):
        raise AssertionError('changed checkpoint reached the model loader')
    monkeypatch.setattr(robotwin_policy, 'create_policy', forbidden_load)
    with pytest.raises(ValueError, match='checkpoint identity changed'):
        runner.main(['--request', str(request_path)])
    assert not (tmp_path / 'outcomes.jsonl').exists()


def test_worker_rechecks_checkpoint_after_condition(tmp_path, monkeypatch, synthetic_evaluation_root):
    from wam_pwtr.evaluation_runtime import runner
    from wam_pwtr import robotwin_policy
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'original weights')
    request = dict(release=str(ROOT), robotwin_root=str(tmp_path), output=str(tmp_path),
                   expected_dataset_stats={}, expected_text_cache={},
                   checkpoint=str(checkpoint), expected_checkpoint_sha256=evaluate.sha(checkpoint),
                   fastwam_root=str(tmp_path), dataset_stats=str(tmp_path), text_cache=str(tmp_path),
                   cases=evaluate.load_cases(synthetic_evaluation_root, tasks=['move_stapler_pad'], conditions=['clean'], episodes_per_condition=1))
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    class ReplacingEnv(Env):
        def __init__(self):
            super().__init__(success_at=1)
        def take_action(self, action, action_type):
            super().take_action(action, action_type)
            checkpoint.write_bytes(b'new training checkpoint')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'environment_arguments', lambda *args: {'clear_cache_freq': 5})
    monkeypatch.setattr(runner, 'advance_instruction_rng', lambda: None)
    monkeypatch.setattr(robotwin_policy, 'create_policy', lambda **kwargs: Policy())
    monkeypatch.setitem(sys.modules, 'envs.move_stapler_pad', SimpleNamespace(move_stapler_pad=ReplacingEnv))
    with pytest.raises(ValueError, match='checkpoint identity changed'):
        runner.main(['--request', str(request_path)])
    assert (tmp_path / 'outcomes.jsonl').exists()


def test_parent_rechecks_checkpoint_before_publishing_summary(tmp_path, monkeypatch, synthetic_evaluation_root):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'original weights')
    digest = evaluate.sha(checkpoint)
    cases = evaluate.load_cases(synthetic_evaluation_root, tasks=['move_stapler_pad'], conditions=['clean'], episodes_per_condition=1)
    args = SimpleNamespace(output=str(tmp_path / 'output'), release=str(ROOT), gpu='0',
                           checkpoint=str(checkpoint), fastwam_root=str(tmp_path), robotwin_root=str(tmp_path),
                           dataset_stats=str(tmp_path), text_cache=str(tmp_path), worker_timeout=5)
    def replaced_at_worker_return(command, log_path, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        assert request['expected_checkpoint_sha256'] == digest
        # This is the process boundary; use a synthetic CPU outcome, not a model.
        rows = [dict(c, status='success', success=True, actions=1) for c in request['cases']]
        (Path(log_path).parent / 'outcomes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        checkpoint.write_bytes(b'replaced after worker exit')
    monkeypatch.setattr(evaluate, 'run_worker', replaced_at_worker_return)
    with pytest.raises(ValueError, match='checkpoint identity changed'):
        evaluate.evaluate(args, cases, dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                                           evaluation_asset_identity=dict(dataset_stats={}, text_cache={})))
    output = Path(args.output)
    assert json.loads((output / 'status.json').read_text())['status'] == 'FAILED'
    assert not (output / 'summary.json').exists()
    assert not (output / 'summary.csv').exists()


def test_evaluation_uses_its_recorded_runtime_sources(tmp_path):
    release, fastwam, robotwin = (tmp_path / name for name in ('release', 'fastwam', 'robotwin'))
    (release / 'provenance').mkdir(parents=True)
    for root in (fastwam, robotwin):
        root.mkdir()
        (root / 'runtime.py').write_bytes(b'recorded evaluation source')
    expected = hashlib.sha256(b'recorded evaluation source').hexdigest()
    (release / 'provenance/evaluation_runtime.json').write_text(json.dumps(dict(
        schema='wam_pwtr_evaluation_sources_v1',
        fastwam_source_hashes={'runtime.py': expected},
        robotwin_source_hashes={'runtime.py': expected})))
    evaluate.check_evaluation_sources(release, fastwam, robotwin)
    (robotwin / 'runtime.py').write_bytes(b'changed task implementation')
    with pytest.raises(ValueError, match='robotwin evaluation source missing or changed'):
        evaluate.check_evaluation_sources(release, fastwam, robotwin)
