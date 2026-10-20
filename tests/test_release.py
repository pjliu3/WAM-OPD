"""Synthetic contract tests; these do not reproduce experimental snapshots."""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from synthetic_training import manifest, references, write_bundle
from wam_pwtr import cli
from wam_pwtr.core import indices, sampled_view, schedule, reweight
from wam_pwtr.io import dataset, resolve, verify_assets

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(path.read_text())


@pytest.mark.parametrize("length", [1, 3, 15, 16, 17, 31, 1000])
def test_lengths(length):
    selected = indices(length)
    assert len(selected) == min(length, 16) and len(set(selected)) == len(selected)
    assert selected[0] == 0 and selected[-1] == length - 1
    if length < 16:
        assert selected == list(range(length))


def test_synthetic_schedule_and_sampling_probabilities():
    view, q = sampled_view(manifest())
    rows = schedule(view)
    assert len(rows) == 96 and len(q) == 96
    assert sum(row["decision_count"] for row in rows) == 798
    assert [row["trajectory_id"] for row in rows[:8]] == [
        f"synthetic-trajectory-{i:02d}" for i in (6, 6, 0, 4, 8, 7, 6, 4)
    ]
    assert math.fsum(q.values()) == 1.
    for episode in view["episodes"]:
        assert all(q[row["record_id"]] == 1 / 12 / len(episode["records"])
                   for row in episode["records"])
    assert [r["decision_index"] for r in view["episodes"][1]["records"]] == list(range(15)) + [16]


def scored_rows(current, initial=0., teacher=0.):
    rows = []
    for tid, positions in (("long", (0, 5, 10)), ("short", (0,))):
        for rank, position in enumerate(positions):
            rows.append(dict(record_id=f"{tid}-{rank}", trajectory_id=tid, decision_index=position,
                             trajectory_length=11 if tid == "long" else 1,
                             task_name="synthetic-task", condition="clean", behavior_version="current_base",
                             phi_current=current, phi_initial=initial, phi_teacher=teacher))
    return rows, {"long-0": 1 / 6, "long-1": 1 / 6, "long-2": 1 / 6, "short-0": 1 / 2}


def test_exclusive_sampled_prefix_matches_analytic_dual_reference_mixture():
    rows, q = scored_rows(math.log(2), teacher=math.log(4))
    result = reweight(rows, q)
    raw = [1., (2 / (2 / 3 + 4 / 3)) ** .1,
           (4 / (2 / 3 + 16 / 3)) ** .1, 1.]
    normalizer = math.fsum(q[row["record_id"]] * value for row, value in zip(rows, raw))
    assert [r["weight"] for r in result] == pytest.approx([value / normalizer for value in raw])
    assert [r["Phi_current"] for r in result] == pytest.approx([0., math.log(2), 2 * math.log(2), 0.])
    assert [r["decision_index"] for r in result] == [0, 5, 10, 0]
    assert [r["trajectory_length"] for r in result] == [11, 11, 11, 1]


def test_q_normalization_clipping_and_second_normalization():
    rows, q = scored_rows(100.)
    result = reweight(rows, q)
    raw = [1., math.exp(10), math.exp(20), 1.]
    norm = math.fsum(q[row["record_id"]] * value for row, value in zip(rows, raw))
    clipped = [min(value / norm, 2.) for value in raw]
    after = math.fsum(q[row["record_id"]] * value for row, value in zip(rows, clipped))
    assert [r["weight"] for r in result] == pytest.approx([value / after for value in clipped])
    assert math.fsum(q[r["record_id"]] * r["weight"] for r in result) == pytest.approx(1.)
    assert result[2]["weight"] > 2.


def test_external_training_bundle_does_not_need_release_data(tmp_path):
    training = write_bundle(tmp_path / "training")
    assets = tmp_path / "assets"
    pool, _ = dataset(training, assets)
    assert pool["episodes"][0]["records"][0]["path"].startswith(str(assets))
    result = verify_assets(training, assets)
    assert result["status"] == "MISSING_OR_INVALID_ASSETS" and result["missing"]
    with pytest.raises(ValueError):
        resolve(assets, "../../escape")


@pytest.mark.parametrize("name", ["pool.json", "reference_scores.json"])
def test_external_metadata_hash_guard_is_preserved(tmp_path, name):
    training = write_bundle(tmp_path / "training")
    with (training / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="modified.*data"):
        dataset(training, tmp_path / "assets")


def test_preview_uses_explicit_training_bundle(tmp_path, monkeypatch, capsys):
    training = write_bundle(tmp_path / "training")
    monkeypatch.setattr(sys, "argv", ["wam-pwtr", "--release-root", str(tmp_path / "code"),
                                    "preview", "--training-root", str(training)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result == dict(trajectories=12, selected_records=96, updates=96, teacher_queries=798, model_queries=0)


def test_preview_rejects_changed_external_schedule(tmp_path, monkeypatch):
    training = write_bundle(tmp_path / "training")
    (training / "schedule.json").write_text("[]")
    monkeypatch.setattr(sys, "argv", ["wam-pwtr", "preview", "--training-root", str(training)])
    with pytest.raises(ValueError, match="schedule"):
        cli.main()


@pytest.mark.parametrize("command", ["preview", "doctor", "train", "infer-record"])
def test_data_commands_require_explicit_training_root(command, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wam-pwtr", command])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_reference_metrics_command_is_not_available(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wam-pwtr", "metrics"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2


def test_source_inventory():
    for item in read(ROOT / "provenance/implementation.json")["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_full_runtime_save_export_and_synthetic_budget(tmp_path):
    import pickle
    from wam_pwtr.runtime import Runner
    from wam_pwtr.io import digest
    pool = manifest()
    view, q = sampled_view(pool)
    rows = schedule(view)
    refs = references(pool)
    subset = {role: {rid: refs[role][rid] for rid in q} for role in ("initial", "teacher")}
    subset["provenance"] = {"synthetic_fixture": True}

    class Backend:
        identities = {"checkpoints": {"initial": {"sha256": "a" * 64}}}
        def __init__(self):
            self.steps = 0
            self.calls = []
            self.queries = 0
        def prime_student(self):
            pass
        def score_records(self, role, records, version):
            self.calls.append((role, len(records), version))
            return {r["record_id"]: 0. for r in records}
        def begin_update(self, expected_decisions):
            self.expected = expected_decisions
            self.actual = 0
        def backward_decision(self, row, *, weight, divisor, seed, query_observer):
            assert divisor == self.expected and weight == pytest.approx(1.)
            query_observer("started")
            self.actual += 1
            self.queries += 1
            query_observer("completed")
            return {"action_mse": 1.}
        def finish_update(self):
            assert self.actual == self.expected
            self.steps += 1
            return {"grad_norm": 1.}
        def state_dict(self):
            return {"steps": self.steps}
        def activity_inventory(self):
            return {"completed_updates": self.steps, "parameter_updates": {"x": self.steps}, "cuda_rng_states": 2}
        def checkpoint_inventory(self, state):
            return digest(state)
        def validate_checkpoint_state(self, state, activity, done):
            assert state["steps"] == done == activity["parameter_updates"]["x"]
        def save_checkpoint(self, path, state):
            Path(path).write_bytes(pickle.dumps(state))
        def load_checkpoint(self, path):
            return pickle.loads(Path(path).read_bytes())
        def export_model(self, path, completed_updates):
            Path(path).write_text(str(completed_updates))

    backend = Backend()
    runner = Runner(view, rows, backend, tmp_path, probabilities=q, references=subset, experiment_id="a" * 64)
    runner.run()
    assert backend.steps == 96 and backend.queries == 798
    assert backend.calls == [("student", 96, version) for version in (0, 24, 48, 72)]
    assert runner.query_accounting()["committed_queries"] == 798
    assert (tmp_path / "checkpoint.pt").stat().st_nlink == 2
    assert read(tmp_path / "exports/update_000096/model_weights.receipt.json")["schema"] == "wam_pwtr_release_export_v1"
