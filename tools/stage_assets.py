"""Stage verified trajectory and model artifacts in the PWTR asset layout.

Supply the external training metadata directory and run on a host where the source pool's record paths resolve. The reference
PWTR-u96 checkpoint is optional; training only needs the initial model and
its two task teachers. No source file or existing destination is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil


MODEL_ROLES = {"initial", "move_stapler_pad", "open_microwave", "pwtr_u96"}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("asset sha256 must contain 64 hexadecimal characters")
    return value.lower()


def _target_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("asset metadata must be a mapping")
    value = spec.get("path")
    if (not isinstance(value, str) or not value or "\x00" in value or "\\" in value
            or Path(value).is_absolute() or PureWindowsPath(value).drive
            or ".." in Path(value).parts or not Path(value).parts):
        raise ValueError("asset target path must be a nonempty contained relative path")
    size = spec.get("size_bytes")
    if type(size) is not int or size < 0:
        raise ValueError("asset size_bytes must be a nonnegative integer")
    return dict(spec, path=Path(value).as_posix(), sha256=_digest(spec.get("sha256")))


def _records(pool):
    if not isinstance(pool, dict) or not isinstance(pool.get("episodes"), list):
        raise ValueError("pool must contain an episode list")
    for episode in pool["episodes"]:
        if not isinstance(episode, dict) or not isinstance(episode.get("records"), list):
            raise ValueError("episode must contain a record list")
        for record in episode["records"]:
            if not isinstance(record, dict) or not isinstance(record.get("record_id"), str) or not record["record_id"]:
                raise ValueError("record_id must be a nonempty string")
            yield record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True,
                        help="external directory containing pool.json and assets.json")
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--stapler-teacher", required=True, type=Path)
    parser.add_argument("--microwave-teacher", required=True, type=Path)
    parser.add_argument("--pwtr-u96", type=Path, help="optional reference checkpoint for inference")
    args = parser.parse_args(argv)
    if args.destination.exists() or args.destination.is_symlink():
        raise ValueError("use a new staging directory")
    read = lambda path: json.loads(path.read_text())
    target = read(args.training_root / "pool.json")
    source = read(args.source_pool)
    assets = read(args.training_root / "assets.json")
    sources = {}
    for record in _records(source):
        rid = record["record_id"]
        if rid in sources:
            raise ValueError("duplicate source record_id: " + rid)
        value = record.get("path")
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("source record path must be a nonempty path")
        sources[rid] = dict(record, sha256=_digest(record.get("sha256")))
    work, target_ids, target_paths = [], set(), set()

    def reserve_spec(spec):
        checked = _target_spec(spec)
        if checked["path"] in target_paths:
            raise ValueError("duplicate target path: " + checked["path"])
        target_paths.add(checked["path"])
        return checked

    for record in _records(target):
        rid = record["record_id"]
        if rid in target_ids:
            raise ValueError("duplicate target record_id: " + rid)
        target_ids.add(rid)
        spec = reserve_spec(record)
        if rid not in sources or sources[rid]["sha256"] != spec["sha256"]:
            raise ValueError("source record identity mismatch: " + rid)
        work.append((Path(sources[rid]["path"]), spec))
    if not isinstance(assets.get("models"), dict) or set(assets["models"]) != MODEL_ROLES:
        raise ValueError("model metadata must identify the initial, two teacher and reference models")
    # Check every declared destination, including an omitted reference model.
    model_specs = {role: reserve_spec(spec) for role, spec in assets["models"].items()}
    models = {"initial": args.initial, "move_stapler_pad": args.stapler_teacher,
              "open_microwave": args.microwave_teacher, "pwtr_u96": args.pwtr_u96}
    work.extend((models[role], spec) for role, spec in model_specs.items() if models[role] is not None)
    # Finish metadata and source validation before creating any destination.
    for source_path, spec in work:
        if (not source_path.is_file() or source_path.stat().st_size != spec["size_bytes"]
                or sha(source_path) != spec["sha256"]):
            raise ValueError("source hash/size mismatch: " + str(source_path))
    args.destination.mkdir(parents=True, exist_ok=False)
    for source_path, spec in work:
        destination = args.destination / spec["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copying keeps the staged artifacts independent of the originals.
        with source_path.open("rb") as source_stream, destination.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream, 8 << 20)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        if sha(destination) != spec["sha256"]:
            raise ValueError("staged file differs: " + str(destination))
    print(json.dumps(dict(status="STAGED", files=len(work), reference_checkpoint_included=args.pwtr_u96 is not None)))


if __name__ == "__main__":
    main()
