"""Small-file asset preparation tests; no model tensors or trajectory payloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from wam_pwtr import cli
from wam_pwtr.io import verify_assets


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pwtr_stage_assets_test", ROOT / "tools/stage_assets.py")
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def assets(tmp_path):
    release = tmp_path / "training"
    data = release
    data.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    records = []
    source_records = []
    for number, raw in enumerate((b"record-zero", b"record-one")):
        original = source / f"record-{number}.pt"
        original.write_bytes(raw)
        item = dict(record_id=f"record-{number}", path=f"records/{digest(raw)}.pt",
                    sha256=digest(raw), size_bytes=len(raw))
        records.append(item)
        source_records.append(dict(item, path=str(original)))
    models = {}
    paths = {}
    for role in ("initial", "move_stapler_pad", "open_microwave", "pwtr_u96"):
        raw = ("fixture-" + role).encode()
        original = source / (role + ".pt")
        original.write_bytes(raw)
        paths[role] = original
        models[role] = dict(path=f"models/{role}.pt", size_bytes=len(raw), sha256=digest(raw))
    pool = dict(initial_checkpoint=dict(models["initial"]),
                teacher_checkpoints={role: dict(models[role]) for role in ("move_stapler_pad", "open_microwave")},
                episodes=[dict(behavior_checkpoint=dict(models["initial"]), records=records)])
    source_pool = source / "pool.json"
    source_pool.write_text(json.dumps(dict(episodes=[dict(records=source_records)])))
    references = data / "reference_scores.json"
    references.write_text('{"initial": {}, "teacher": {}}')

    def save_metadata():
        (data / "pool.json").write_text(json.dumps(pool))
        index = dict(models=models, pool_manifest_sha256=digest((data / "pool.json").read_bytes()),
                     reference_scores_sha256=digest(references.read_bytes()))
        (data / "assets.json").write_text(json.dumps(index))

    save_metadata()
    destination = tmp_path / "staged"

    def run_stage(monkeypatch, *, include_reference=False):
        argv = ["stage_assets.py", "--training-root", str(release), "--source-pool", str(source_pool),
                "--destination", str(destination), "--initial", str(paths["initial"]),
                "--stapler-teacher", str(paths["move_stapler_pad"]),
                "--microwave-teacher", str(paths["open_microwave"])]
        if include_reference:
            argv += ["--pwtr-u96", str(paths["pwtr_u96"])]
        monkeypatch.setattr(sys, "argv", argv)
        STAGE.main()

    return dict(release=release, source=source, pool=pool, source_pool=source_pool,
                source_records=source_records, models=models, paths=paths,
                destination=destination, save=save_metadata, run=run_stage)


def test_stage_training_inputs_does_not_require_trained_checkpoint(assets, monkeypatch):
    before = {p: p.read_bytes() for p in assets["source"].iterdir() if p.is_file()}
    assets["run"](monkeypatch)
    result = verify_assets(assets["release"], assets["destination"], hash_files=True, purpose="train")
    assert result["status"] == "READY" and result["expected_files"] == 5
    assert not (assets["destination"] / "models/pwtr_u96.pt").exists()
    assert before == {p: p.read_bytes() for p in before}


def test_default_all_still_requires_reference_checkpoint(assets, monkeypatch):
    assets["run"](monkeypatch)
    result = verify_assets(assets["release"], assets["destination"])
    assert result["status"] == "MISSING_OR_INVALID_ASSETS"
    assert result["expected_files"] == 6
    assert result["missing"] == [str(assets["destination"] / "models/pwtr_u96.pt")]


def test_reference_inference_needs_only_records_and_reference_model(assets, monkeypatch):
    assets["run"](monkeypatch, include_reference=True)
    for role in ("initial", "move_stapler_pad", "open_microwave"):
        (assets["destination"] / assets["models"][role]["path"]).unlink()
    result = verify_assets(assets["release"], assets["destination"], hash_files=True, purpose="infer")
    assert result["status"] == "READY" and result["expected_files"] == 3


def test_optional_reference_is_staged_when_supplied(assets, monkeypatch):
    assets["run"](monkeypatch, include_reference=True)
    result = verify_assets(assets["release"], assets["destination"], hash_files=True)
    assert result["status"] == "READY" and result["expected_files"] == 6


def test_doctor_train_flag_selects_training_inputs(assets, monkeypatch, capsys):
    assets["run"](monkeypatch)
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["wam-pwtr", "doctor", "--training-root", str(assets["release"]),
                                     "--asset-root", str(assets["destination"]),
                                     "--purpose", "train", "--hash-files"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY"


def test_unknown_asset_purpose_is_rejected(assets):
    with pytest.raises(ValueError, match="purpose"):
        verify_assets(assets["release"], assets["destination"], purpose="other")


@pytest.mark.parametrize("path", ["../escaped.pt", "ABSOLUTE", "records/../../escaped.pt"])
def test_unsafe_target_record_path_rejected_before_destination(assets, monkeypatch, path):
    if path == "ABSOLUTE":
        path = str(assets["destination"].parent / "escaped-absolute.pt")
    assets["pool"]["episodes"][0]["records"][-1]["path"] = path
    assets["save"]()
    with pytest.raises(ValueError, match="path"):
        assets["run"](monkeypatch, include_reference=True)
    assert not assets["destination"].exists()


def test_unused_reference_target_path_is_also_validated(assets, monkeypatch):
    assets["models"]["pwtr_u96"]["path"] = "../escaped-reference.pt"
    assets["save"]()
    with pytest.raises(ValueError, match="path"):
        assets["run"](monkeypatch)
    assert not assets["destination"].exists()


def test_duplicate_source_record_id_is_not_silently_overwritten(assets, monkeypatch):
    records = assets["source_records"]
    assets["source_pool"].write_text(json.dumps(dict(episodes=[dict(records=records + records[:1])])))
    with pytest.raises(ValueError, match="duplicate"):
        assets["run"](monkeypatch, include_reference=True)
    assert not assets["destination"].exists()


def test_duplicate_destination_is_rejected_before_any_copy(assets, monkeypatch):
    records = assets["pool"]["episodes"][0]["records"]
    records[1]["path"] = records[0]["path"]
    assets["save"]()
    with pytest.raises(ValueError, match="duplicate"):
        assets["run"](monkeypatch, include_reference=True)
    assert not assets["destination"].exists()


def test_late_bad_metadata_is_rejected_before_any_copy(assets, monkeypatch):
    assets["models"]["open_microwave"]["size_bytes"] = "not-an-integer"
    assets["save"]()
    with pytest.raises(ValueError):
        assets["run"](monkeypatch, include_reference=True)
    assert not assets["destination"].exists()


def test_corrupt_source_is_rejected_before_destination_creation(assets, monkeypatch):
    assets["paths"]["open_microwave"].write_bytes(b"broken")
    with pytest.raises(ValueError):
        assets["run"](monkeypatch, include_reference=True)
    assert not assets["destination"].exists()


def test_existing_destination_is_not_modified(assets, monkeypatch):
    assets["destination"].mkdir()
    sentinel = assets["destination"] / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new staging"):
        assets["run"](monkeypatch, include_reference=True)
    assert list(assets["destination"].iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"keep"
