"""Training state, query accounting, and checkpoint management for PWTR."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked_scores(scores, records):
    expected = {r["record_id"] for r in records}
    if set(scores) != expected or any(not math.isfinite(float(v)) for v in scores.values()):
        raise ValueError("scores must cover the supplied record inventory exactly with finite values")
    return {str(k): float(v) for k, v in scores.items()}


def verify_reference_inventory(manifest, checkpoints, teachers):
    """Bind configured references and behavior collectors to the locked pool."""
    try:
        initial = manifest["initial_checkpoint"]["sha256"]
        fixed_teachers = manifest["teacher_checkpoints"]
        if checkpoints["initial"]["sha256"] != initial:
            raise ValueError("initial reference checkpoint differs from the pool manifest")
        for episode in manifest["episodes"]:
            task = episode["task_name"]
            specs = [spec for spec in teachers if task in spec["tasks"]]
            if len(specs) != 1 or checkpoints[specs[0]["name"]]["sha256"] != fixed_teachers[task]["sha256"]:
                raise ValueError("task teacher reference checkpoint differs from the pool manifest")
            source = episode["behavior_version"]
            if source not in ("current_base", "teacher"):
                raise ValueError("unknown behavior collector source")
            expected = initial if source == "current_base" else fixed_teachers[task]["sha256"]
            if episode["behavior_checkpoint"]["sha256"] != expected:
                raise ValueError("episode behavior collector checkpoint differs from its declared source")
            if any(row["behavior_checkpoint_sha256"] != expected for row in episode["records"]):
                raise ValueError("record behavior collector checkpoint differs from its trajectory")
    except KeyError as exc:
        raise ValueError(f"missing reference/behavior checkpoint identity: {exc}") from exc


class TrainingRunner:
    """One row = one trajectory, at most sixteen independent losses, one step.

    Only the initial and final training states are committed. Query attempts
    remain durably recorded even when an update fails. This release does not
    resume or replay interrupted training.
    """
    def __init__(self, manifest, schedule, backend, run_dir, *, seed=0,
                 probabilities=None, reweight=None, refresh_interval=24, checkpoint_every=96, strength=.1):
        if (type(seed) is not int or seed != 0 or strength != .1
                or refresh_interval != 24 or checkpoint_every != 96 or len(schedule) != 96):
            raise ValueError("PWTR requires seed0, strength=.1, 96 updates, refresh24 and final-only save")
        if probabilities is None or not callable(reweight):
            raise ValueError("PWTR requires an explicit sampled-view probability map and reweight function")
        if manifest.get("sampling_view") != "sampled_pwtr_v1_original_indices_preserved":
            raise ValueError("PWTR requires the fixed temporal sampled view")
        self.manifest, self.schedule, self.backend = manifest, schedule, backend
        self.records = [record for episode in manifest["episodes"] for record in episode["records"]]
        self.lookup = {row["record_id"]: row for row in self.records}
        if not self.records or len(self.lookup) != len(self.records):
            raise ValueError("pool requires nonempty unique record identities")
        lengths = {episode["trajectory_id"]: len(episode["records"]) for episode in manifest["episodes"]}
        selected_ids = {episode["trajectory_id"]: [r["record_id"] for r in episode["records"]]
                        for episode in manifest["episodes"]}
        expected_q = {r["record_id"]: 1.0 / len(manifest["episodes"]) / len(episode["records"])
                      for episode in manifest["episodes"] for r in episode["records"]}
        if probabilities != expected_q:
            raise ValueError("PWTR probabilities must be q=1/(N*K_tau) on the fixed selected points")
        for i, row in enumerate(schedule):
            expected_count = min(16, lengths.get(row["trajectory_id"], 0))
            if (row["update"] != i + 1 or not expected_count or
                    row.get("decision_count") != expected_count or len(row["record_ids"]) != expected_count):
                raise ValueError("schedule must use consecutive 1-based updates and min(16,T) decisions")
            if any(rid not in self.lookup for rid in row["record_ids"]):
                raise ValueError("schedule references an unknown pool record")
            if any(self.lookup[rid]["trajectory_id"] != row["trajectory_id"] for rid in row["record_ids"]):
                raise ValueError("one update must belong to exactly one trajectory")
            selected = [self.lookup[rid] for rid in row["record_ids"]]
            if row.get("decision_indices") != [r["decision_index"] for r in selected]:
                raise ValueError("schedule decision indices do not match record identities")
            if any(row.get(key) != r[key] for r in selected for key in
                   ("task_name", "condition", "behavior_version")):
                raise ValueError("schedule metadata differs from its trajectory")
            if row.get("replacement") is not False or len(set(row["record_ids"])) != expected_count:
                raise ValueError("schedule must use unique selected points without replacement")
            if row["record_ids"] != selected_ids[row["trajectory_id"]]:
                raise ValueError("schedule must process the fixed selected points in temporal order")
        self.probabilities, self.reweight = probabilities, reweight
        self.strength = float(strength)
        self.refresh_interval = refresh_interval
        self.checkpoint_every = checkpoint_every
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.training_seed = seed
        self.completed_updates = 0
        self.checkpoint_updates = 0
        self.references = None
        self.snapshot = None
        self._run_started = False
        self.identities = {"pool": canonical_digest(manifest),
                           "schedule": canonical_digest(schedule),
                           "references": backend.identities,
                           "recipe": canonical_digest({"seed": seed, "refresh": 24,
                               "decisions": "fixed_endpoint_inclusive_temporal_sample", "strength": self.strength, "eta": 2 / 3, "cap": 2,
                               "probabilities": probabilities})}
        self.ledger_path = self.run_dir / "query_ledger.jsonl"
        header = {"event": "identity", "sha256": canonical_digest(self.identities)}
        self.ledger_events = []
        if self.ledger_path.exists():
            raise ValueError("PWTR requires a new query ledger; continuation is unsupported")
        self._append_ledger(header)
        self.query_attempts = {}
        self.logical_queries = {}

    def _append_ledger(self, event):
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.ledger_events.append(event)

    def _apply_query_event(self, event):
        attempt = event["attempt"]
        if event["event"] == "started":
            if attempt != len(self.query_attempts) + 1:
                raise ValueError("query ledger attempt sequence is invalid")
            key = (event["update"], event["decision_slot"])
            identity = (event["record_id"], event["query_seed"])
            if key in self.logical_queries:
                raise ValueError("teacher query already attempted; replay is unsupported")
            self.logical_queries[key] = identity
            self.query_attempts[attempt] = dict(event, completed=False)
        elif event["event"] == "completed":
            if attempt not in self.query_attempts or self.query_attempts[attempt]["completed"]:
                raise ValueError("query ledger contains an unmatched completion")
            self.query_attempts[attempt]["completed"] = True
        else:
            raise ValueError("unknown query ledger event")

    def _query_observer(self, update, slot, record_id, seed):
        state = {"attempt": None, "complete": False}

        def observe(phase):
            if phase == "started" and state["attempt"] is None:
                if (update, slot) in self.logical_queries:
                    raise ValueError("teacher query already attempted; replay is unsupported")
                state["attempt"] = len(self.query_attempts) + 1
                event = {"event": phase, "attempt": state["attempt"], "update": update,
                         "decision_slot": slot, "record_id": record_id, "query_seed": seed}
            elif phase == "completed" and state["attempt"] is not None and not state["complete"]:
                event = {"event": phase, "attempt": state["attempt"]}
                state["complete"] = True
            else:
                raise ValueError("backend must emit one started/completed query pair")
            self._append_ledger(event)
            self._apply_query_event(event)
        return observe, state

    def query_accounting(self):
        attempts = len(self.query_attempts)
        return {"attempted_queries": attempts,
                "finished_query_calls": sum(row["completed"] for row in self.query_attempts.values()),
                "replayed_queries": attempts - len(self.logical_queries),
                "unknown_query_attempts": sum(not row["completed"] for row in self.query_attempts.values()),
                "committed_queries": sum(row["decision_count"] for row in self.schedule[:self.checkpoint_updates]),
                "completed_update_queries": sum(row["decision_count"] for row in self.schedule[:self.completed_updates])}

    def prepare_references(self):
        raise NotImplementedError("the PWTR runner requires verified fixed reference scores")

    def refresh_weights(self):
        raise NotImplementedError("use the PWTR fresh runner for sampled-prefix refreshes")

    def checkpoint(self):
        raise NotImplementedError("use the PWTR complete-save implementation")

    def export_for_evaluation(self):
        raise NotImplementedError("the public PWTR runner owns the final hardlink export")

    def resume(self, path):
        raise ValueError("PWTR runs from the initial model; checkpoint resume is unsupported")

    def run(self, *, stop_after=None):
        if stop_after is not None:
            raise ValueError("PWTR executes the fixed 96-update schedule without partial-run exports")
        if self._run_started:
            raise ValueError("PWTR run already attempted; restart is unsupported")
        self._run_started = True
        target = len(self.schedule)
        while self.completed_updates < target:
            if self.snapshot is None or self.snapshot["version"] != self.completed_updates // 24 * 24:
                self.refresh_weights()
            required_version = self.completed_updates // 24 * 24
            if self.snapshot["version"] != required_version:
                raise ValueError("refusing stale student weight snapshot")
            # Preserve the initial state as evidence; this is not a resume entry.
            if self.completed_updates == 0 and not (self.run_dir / "checkpoint.pt").exists():
                self.checkpoint()
            row = self.schedule[self.completed_updates]
            decision_count = row["decision_count"]
            self.backend.begin_update(expected_decisions=decision_count)
            decisions = []
            for slot, rid in enumerate(row["record_ids"]):
                seed = self.rng.randrange(0, 2 ** 63)
                observer, query_state = self._query_observer(self.completed_updates + 1, slot, rid, seed)
                metrics = self.backend.backward_decision(self.lookup[rid],
                    weight=self.snapshot["weights"][rid], divisor=decision_count, seed=seed, query_observer=observer)
                if not query_state["complete"]:
                    raise ValueError("backend did not account for a completed teacher query")
                decisions.append(dict(record_id=rid, query_seed=seed,
                                      weight=self.snapshot["weights"][rid], **metrics))
            metrics = self.backend.finish_update()
            self.completed_updates += 1
            atomic_json(self.run_dir / "updates" / f"{self.completed_updates:06d}.json",
                        dict(update=self.completed_updates, trajectory_id=row["trajectory_id"],
                             weight_version=self.snapshot["version"], decisions=decisions, **metrics))
            if self.completed_updates % self.checkpoint_every == 0 or self.completed_updates == target:
                self.checkpoint()
            print(json.dumps({"completed_updates": self.completed_updates,
                              **self.query_accounting(), **metrics}), flush=True)
        return self.completed_updates
