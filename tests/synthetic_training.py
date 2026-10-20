"""Invented metadata for CPU tests; contains no experimental records or scores."""
import hashlib
import json
from pathlib import Path


def manifest():
    def checkpoint(label):
        return dict(path=f"models/{label}.pt", sha256=hashlib.sha256(label.encode()).hexdigest(),
                    size_bytes=len(label))
    initial = checkpoint("synthetic-initial")
    teachers = {task: checkpoint("synthetic-" + task) for task in ("task_a", "task_b")}
    episodes = []
    for task in teachers:
        for condition in ("clean", "rand"):
            for source, length in (("current_base", 3), ("current_base", 17), ("teacher", 5)):
                number = len(episodes)
                tid = f"synthetic-trajectory-{number:02d}"
                behavior = initial if source == "current_base" else teachers[task]
                records = []
                for index in range(length):
                    rid = f"{tid}-decision-{index:02d}"
                    records.append(dict(
                        record_id=rid, trajectory_id=tid, decision_index=index,
                        trajectory_length=length, task_name=task, condition=condition,
                        behavior_version=source, behavior_checkpoint_sha256=behavior["sha256"],
                        path=f"records/{rid}.pt", size_bytes=len(rid),
                        sha256=hashlib.sha256(rid.encode()).hexdigest(),
                    ))
                episodes.append(dict(trajectory_id=tid, task_name=task, condition=condition,
                                     behavior_version=source, behavior_checkpoint=dict(behavior),
                                     records=records))
    return dict(initial_checkpoint=initial, teacher_checkpoints=teachers, episodes=episodes)


def references(pool):
    ids = [row["record_id"] for episode in pool["episodes"] for row in episode["records"]]
    return {role: {rid: 0. for rid in ids} for role in ("initial", "teacher")}


def write_bundle(root):
    """Write the four explicit external metadata files, without binary assets."""
    from wam_pwtr.core import sampled_view, schedule
    root = Path(root)
    root.mkdir(parents=True)
    pool = manifest()
    for name, value in (("pool.json", pool), ("reference_scores.json", references(pool)),
                        ("schedule.json", schedule(sampled_view(pool)[0]))):
        (root / name).write_text(json.dumps(value))
    models = dict(initial=pool["initial_checkpoint"], **pool["teacher_checkpoints"])
    models["pwtr_u96"] = dict(path="models/synthetic-trained.pt", size_bytes=5,
                              sha256=hashlib.sha256(b"model").hexdigest())
    index = dict(models=models,
                 pool_manifest_sha256=hashlib.sha256((root / "pool.json").read_bytes()).hexdigest(),
                 reference_scores_sha256=hashlib.sha256((root / "reference_scores.json").read_bytes()).hexdigest())
    (root / "assets.json").write_text(json.dumps(index))
    return root
