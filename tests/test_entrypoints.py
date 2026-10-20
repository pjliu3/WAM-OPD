"""Public entry-point and runtime-inventory contracts."""
import hashlib
import json
from pathlib import Path

import pytest

from wam_pwtr import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_installed_entry_uses_current_release_checkout(tmp_path, monkeypatch):
    (tmp_path / 'configs/evaluation').mkdir(parents=True)
    (tmp_path / 'configs/evaluation/protocol.json').write_text('{}')
    (tmp_path / 'SHA256SUMS').write_text('')
    monkeypatch.chdir(tmp_path)
    args = evaluate.parser().parse_args([
        '--evaluation-root', 'local-bundle',
        '--checkpoint', 'model.pt', '--fastwam-root', 'fastwam',
        '--robotwin-root', 'robotwin', '--text-cache', 'text', '--output', 'out'])
    assert Path(args.release).resolve() == tmp_path


def test_entry_requires_explicit_local_evaluation_bundle(capsys):
    with pytest.raises(SystemExit) as error:
        evaluate.parser().parse_args(['--checkpoint', 'model.pt', '--fastwam-root', 'fastwam',
                                     '--robotwin-root', 'robotwin', '--text-cache', 'text', '--output', 'out'])
    assert error.value.code == 2
    assert '--evaluation-root' in capsys.readouterr().err


def test_missing_bundle_reports_external_inputs_without_creating_output(tmp_path, capsys):
    output = tmp_path / 'output'
    result = evaluate.main(['--evaluation-root', str(tmp_path / 'absent-bundle'),
                            '--checkpoint', 'model.pt', '--fastwam-root', 'fastwam',
                            '--robotwin-root', 'robotwin', '--text-cache', 'text',
                            '--output', str(output), '--check-only'])
    assert result == 2 and not output.exists()
    error = capsys.readouterr().err
    assert 'local evaluation bundle input' in error
    assert 'not included in this code release' in error


def test_external_bundle_defaults_do_not_read_release_data(tmp_path, monkeypatch, synthetic_evaluation_root):
    release = tmp_path / 'code-only-release'
    release.mkdir()
    observed = {}
    def inspect(args, cases):
        observed.update(evaluation_root=Path(args.evaluation_root),
                        expert_cache=Path(args.expert_cache),
                        dataset_stats=Path(args.dataset_stats), cases=len(cases))
        return {'status': 'PREFLIGHT_PASSED', 'new_episodes': 0}
    monkeypatch.setattr(evaluate, 'preflight', inspect)
    result = evaluate.main(['--release', str(release), '--evaluation-root', str(synthetic_evaluation_root),
                            '--checkpoint', 'model.pt', '--fastwam-root', 'fastwam',
                            '--robotwin-root', 'robotwin', '--text-cache', 'text',
                            '--output', str(tmp_path / 'out'), '--check-only'])
    assert result == 0
    assert observed == dict(evaluation_root=synthetic_evaluation_root,
                            expert_cache=synthetic_evaluation_root / 'expert_cache',
                            dataset_stats=synthetic_evaluation_root / 'evaluation_dataset_stats.json', cases=400)
    assert not (release / 'data').exists()


def test_infer_record_checks_evaluation_sources_before_model_loading(tmp_path, monkeypatch):
    pytest.importorskip('torch', minversion='2.7')
    from wam_pwtr import inference
    from fastwam_opd import model_loader

    release, fastwam, assets = (tmp_path / name for name in ('release', 'fastwam', 'assets'))
    (release / 'provenance').mkdir(parents=True)
    fastwam.mkdir()
    assets.mkdir()
    (fastwam / 'runtime.py').write_bytes(b'evaluation runtime')
    source_hash = hashlib.sha256(b'evaluation runtime').hexdigest()
    (release / 'provenance/evaluation_runtime.json').write_text(json.dumps({
        'schema': 'wam_pwtr_evaluation_sources_v1',
        'fastwam_source_hashes': {'runtime.py': source_hash},
        'robotwin_source_hashes': {'unused_simulator.py': '0' * 64}}))
    checkpoint, record = assets / 'model.pt', assets / 'record.pt'
    checkpoint.write_bytes(b'checkpoint')
    record.write_bytes(b'record')
    row = {'record_id': 'one', 'path': str(record),
           'sha256': hashlib.sha256(b'record').hexdigest()}
    index = {'models': {'pwtr_u96': {'path': 'model.pt',
              'sha256': hashlib.sha256(b'checkpoint').hexdigest()}}}
    monkeypatch.setattr(inference, 'dataset', lambda *args: ({'episodes': [{'records': [row]}]}, index))

    class ReachedModelLoad(Exception):
        pass

    def load(**kwargs):
        assert kwargs['strict_full_checkpoint'] is True
        assert kwargs['checkpoint'] == checkpoint
        raise ReachedModelLoad

    monkeypatch.setattr(model_loader, 'load_fastwam_model', load)
    with pytest.raises(ReachedModelLoad):
        inference.infer_record(release, tmp_path / 'local-training-bundle', assets, fastwam, 'one', tmp_path / 'out.pt')
    assert not (tmp_path / 'out.pt').exists()


def test_runtime_lock_covers_robot_configuration_and_observation_code():
    lock = json.loads((ROOT / 'provenance/evaluation_runtime.json').read_text())
    required = {'assets/embodiments/aloha-agilex/config.yml',
                'envs/camera/camera.py', 'envs/utils/create_actor.py',
                'envs/utils/rand_create_cluttered_actor.py'}
    assert required <= set(lock['robotwin_source_hashes'])


def test_changed_robot_configuration_is_rejected(tmp_path):
    release, fastwam, robotwin = (tmp_path / name for name in ('release', 'fastwam', 'robotwin'))
    (release / 'provenance').mkdir(parents=True)
    relative = 'assets/embodiments/aloha-agilex/config.yml'
    config = robotwin / relative
    config.parent.mkdir(parents=True)
    original = b'joint_stiffness: 1000\n'
    config.write_bytes(original)
    (release / 'provenance/evaluation_runtime.json').write_text(json.dumps({
        'schema': 'wam_pwtr_evaluation_sources_v1',
        'fastwam_source_hashes': {},
        'robotwin_source_hashes': {relative: hashlib.sha256(original).hexdigest()}}))
    evaluate.check_evaluation_sources(release, fastwam, robotwin)
    config.write_bytes(b'joint_stiffness: 1\n')
    with pytest.raises(ValueError, match='robotwin evaluation source missing or changed'):
        evaluate.check_evaluation_sources(release, fastwam, robotwin)
