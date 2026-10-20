"""FastWAM integration and full-MoT AdamW optimization for sampled-prefix PWTR.
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from .runtime import canonical_digest, file_digest, verify_reference_inventory


def capture_rng():
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


@contextlib.contextmanager
def isolated_evaluation(model=None):
    """Preserve RNG and every submodule flag, including frozen eval children."""
    rng = capture_rng()
    modes = [] if model is None else [(m, m.training) for m in model.modules()]
    if model is not None:
        model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(rng)


def full_action_loss(student, teacher):
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("student and teacher action velocities must share [1,H,D]")
    return 10.0 * (student.float() - teacher.detach().float()).square().mean()


def predict_action_with_checkpoint(adapter, condition_input, latents_action,
                                   timestep_action, *, enabled=True):
    """Recompute the actual action path, including its trainable video cache."""
    def forward(latents, timestep):
        condition = adapter.prepare_conditions([condition_input])
        return adapter.predict_velocity(latents_action=latents,
                                        timestep_action=timestep, condition=condition)
    if enabled and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(forward, latents_action, timestep_action,
                          use_reentrant=False, preserve_rng_state=True)
    return forward(latents_action, timestep_action)


class MasterAdamW:
    """FP32 parameter updates, with optional synchronous CPU master offload.

    CPU mode accumulates each microgradient into persistent FP32 CPU gradients
    and releases GPU gradients immediately. No compressed optimizer states,
    gradient sanitation, per-decision clipping or per-decision steps are used.
    """
    def __init__(self, parameters, *, cpu_offload=False):
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("no trainable parameters")
        self.cpu_offload = bool(cpu_offload)
        if cpu_offload:
            self.masters = tuple(torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.parameters)
        else:
            if any(p.dtype != torch.float32 for p in self.parameters):
                raise ValueError("standard AdamW requires FP32 trainable parameters")
            self.masters = self.parameters
        self.optimizer = torch.optim.AdamW(self.masters, lr=1e-6, betas=(.9, .95),
                                          weight_decay=0., foreach=False)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None

    def backward(self, loss):
        if loss.ndim or not bool(torch.isfinite(loss.detach())):
            raise ValueError("nonfinite or nonscalar decision loss")
        loss.backward()
        if self.cpu_offload:
            for parameter, master in zip(self.parameters, self.masters):
                if parameter.grad is not None:
                    grad = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
                    if master.grad is None:
                        master.grad = grad.clone() if grad.data_ptr() == parameter.grad.data_ptr() else grad
                    else:
                        master.grad.add_(grad)
                    parameter.grad = None

    def step(self):
        if not any(p.grad is not None for p in self.masters):
            raise RuntimeError("update produced no gradients")
        norm = torch.nn.utils.clip_grad_norm_(self.masters, 1., error_if_nonfinite=True, foreach=False)
        self.optimizer.step()
        if self.cpu_offload:
            with torch.no_grad():
                for parameter, master in zip(self.parameters, self.masters):
                    parameter.copy_(master)
        return {"grad_norm": float(norm)}

    def state_dict(self):
        return {"cpu_offload": self.cpu_offload,
                "masters": [p.detach().cpu() for p in self.masters] if self.cpu_offload else None,
                "adamw": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        raise ValueError("PWTR optimizer-state restore is unsupported")


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class FastWAMBackend:
    def __init__(self, config, manifest, *, seed=0):
        if (type(seed) is not int or seed != 0 or config.get("strength") != .1
                or type(config.get("score_batch_size")) is not int or config["score_batch_size"] != 1
                or config.get("mixed_precision") != "bf16"
                or config.get("cpu_optimizer_offload") is not True
                or config.get("activation_checkpointing") is not True):
            raise ValueError("PWTR requires seed0, strength=.1, B1, BF16, CPU masters and activation checkpointing")
        self.config = config
        legacy = Path(config["legacy_source"]).resolve()
        if not (legacy / "fastwam_opd" / "fastwam_adapter.py").is_file():
            raise ValueError("legacy_source must contain the fastwam_opd adapter package")
        sys.path.insert(0, str(legacy))
        import fastwam_opd.fastwam_adapter as adapter_module
        if Path(adapter_module.__file__).resolve().parent != legacy / "fastwam_opd":
            raise ValueError("another fastwam_opd source was already imported")
        from fastwam_opd.model_loader import load_fastwam_model
        from fastwam_opd.record_io import load_opd_record, load_opd_metadata
        from fastwam_opd.history_scoring import score_record_paths
        from fastwam_opd.query import sample_query_index
        self.adapter_module = adapter_module
        self.load_model = load_fastwam_model
        self.load_record, self.load_metadata = load_opd_record, load_opd_metadata
        self.score_paths, self.sample_query_index = score_record_paths, sample_query_index
        self.student_model = self.student_adapter = self.optimizer = None
        self.aux_model = self.aux_adapter = self.aux_key = None
        self.student_device = config.get("student_device", "cuda:0")
        self.teacher_device = config.get("teacher_device", "cuda:1")
        self.cpu_offload = bool(config.get("cpu_optimizer_offload", False))
        self.sampling = config["sampling"]
        if config.get("mixed_precision", "bf16") not in ("bf16", "no"):
            raise ValueError("this recipe supports BF16 or FP32, not unscaled FP16")
        if config.get("joint_video_action", False):
            raise ValueError("WAM-OPD v2 uses the deployed action-only flow")
        self.teacher_specs = config["teachers"]
        tasks = {r["task_name"] for e in manifest["episodes"] for r in e["records"]}
        if any(sum(task in spec["tasks"] for spec in self.teacher_specs) != 1 for task in tasks):
            raise ValueError("every pool task must have exactly one fixed teacher")
        self._verified_records = {}
        checkpoint_paths = {"initial": config["initial_checkpoint"],
                            **{spec["name"]: spec["checkpoint"] for spec in self.teacher_specs}}
        checkpoint_hashes = {name: {"path": str(Path(path).resolve()), "sha256": file_digest(path)}
                             for name, path in checkpoint_paths.items()}
        verify_reference_inventory(manifest, checkpoint_hashes, self.teacher_specs)
        code_files = {str(path.relative_to(legacy)): file_digest(path)
                      for path in sorted((legacy / "fastwam_opd").glob("*.py"))}
        root = Path(config["fastwam_root"])
        fastwam_files = {str(path.relative_to(root)): file_digest(path)
                         for subdir in ("src", "configs")
                         for path in sorted((root / subdir).rglob("*"))
                         if path.is_file() and path.suffix in (".py", ".yaml", ".yml")}
        self.identities = {"checkpoints": checkpoint_hashes, "legacy_source": code_files,
                           "fastwam_source": fastwam_files, "backend_config": config,
                           "runtime_source": {path.name: file_digest(path)
                               for path in sorted(Path(__file__).parent.glob("*.py"))},
                           "scorer": {"kappa": .025, "sigma_min": 1e-4,
                                      "residual_dtype": "float32", "reduction_dtype": "float64"}}
        # Content hashes, not mtimes alone, bind every pool file at process start.
        for episode in manifest["episodes"]:
            for record in episode["records"]:
                self.verify_record(record)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np
            np.random.seed(seed % (2 ** 32))
        except ImportError:
            pass

    def verify_record(self, row):
        path = Path(row["path"])
        stat = path.stat()
        marker = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, row["sha256"])
        if self._verified_records.get(row["record_id"]) != marker:
            if file_digest(path) != row["sha256"]:
                raise ValueError(f"pool content hash mismatch: {row['record_id']}")
            self._verified_records[row["record_id"]] = marker

    def _new_model(self, checkpoint, device):
        model = self.load_model(fastwam_root=self.config["fastwam_root"],
            config_name=self.config.get("model_config", "sim_robotwin.yaml"),
            checkpoint=checkpoint, device=device,
            mixed_precision=self.config.get("mixed_precision", "bf16"),
            load_text_encoder=bool(self.config.get("load_text_encoder", False)),
            action_lora_rank=0)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _student(self):
        if self.student_model is None:
            self.student_model = self._new_model(self.config["initial_checkpoint"], self.student_device)
            parameters = self.adapter_module.configure_trainable_parameters(self.student_model, "full_mot")
            self.optimizer = MasterAdamW(parameters, cpu_offload=self.cpu_offload)
            self.student_adapter = self.adapter_module.FastWAMActionAdapter(self.student_model, train_video_condition=True)
            self.parameter_names = [name for name, p in self.student_model.named_parameters() if p.requires_grad]
            print(json.dumps({"student_trainable_elements": sum(p.numel() for p in parameters),
                              "cpu_optimizer_offload": self.cpu_offload}), flush=True)
        return self.student_adapter

    def _auxiliary(self, checkpoint):
        if self.aux_key != checkpoint:
            self.aux_model = self.aux_adapter = self.aux_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.aux_model = self._new_model(checkpoint, self.teacher_device)
            self.aux_adapter = self.adapter_module.FastWAMActionAdapter(self.aux_model)
            self.aux_key = checkpoint
        return self.aux_adapter

    def _teacher(self, task):
        spec, = [spec for spec in self.teacher_specs if task in spec["tasks"]]
        return self._auxiliary(spec["checkpoint"])

    def _record(self, row):
        self.verify_record(row)
        record = self.load_record(row["path"])
        if record.condition.prompt is not None and not self.config.get("load_text_encoder", False):
            raise ValueError("pool requires cached text embeddings or load_text_encoder=true")
        if record.condition.action_horizon != self.sampling["action_horizon"]:
            raise ValueError("record action horizon differs from configured deployed horizon")
        return record

    def _prefix(self, row):
        value = row.get("executed_action_steps")
        if value is None:
            value = self.load_metadata(row["path"]).get("executed_action_steps")
        if type(value) is not int or value <= 0:
            raise ValueError("missing actual executed action prefix evidence")
        return value

    def score_records(self, role, records, version):
        if role not in ("initial", "teacher", "student"):
            raise ValueError("unknown scoring role")
        result = {}
        started = time.monotonic()
        total = len(records)
        print(json.dumps({"event": "scoring_started", "role": role, "version": version,
                          "scored": 0, "total": total, "elapsed_seconds": 0.}), flush=True)
        last_reported = 0
        # Model construction also consumes RNG; isolate the entire scoring phase.
        batch_size = self.config["score_batch_size"]
        if type(batch_size) is not int or batch_size != 1:
            raise ValueError("PWTR requires B1 path scoring")
        ordered = sorted(records, key=lambda r: (r["task_name"], r["record_id"]))
        batches = []
        for row in ordered:
            if not batches or len(batches[-1]) == batch_size or batches[-1][0]["task_name"] != row["task_name"]:
                batches.append([])
            batches[-1].append(row)
        with isolated_evaluation():
            for batch in batches:
                row = batch[0]
                adapter = (self._student() if role == "student" else
                           self._teacher(row["task_name"]) if role == "teacher" else
                           self._auxiliary(self.config["initial_checkpoint"]))
                loaded = [self._record(item) for item in batch]
                # Mixed schedule lengths are uncommon but must never be padded
                # into artificial transitions. Split just that batch if needed.
                compatible = len({tuple(record.trajectory.latents.shape) for record in loaded}) == 1
                with isolated_evaluation(adapter.model), self.adapter_module.autocast_for(adapter.model):
                    if compatible:
                        values = self.score_paths(adapter, loaded, [self._prefix(item) for item in batch],
                                                  sigma_scale=.025, min_sigma=1e-4)
                    else:
                        values = [self.score_paths(adapter, [record], [self._prefix(item)],
                                                   sigma_scale=.025, min_sigma=1e-4)[0]
                                  for item, record in zip(batch, loaded)]
                    result.update((item["record_id"], score) for item, score in zip(batch, values))
                # Do not retain the prior teacher while _auxiliary loads the
                # next task: its old adapter also owns the full model.
                del adapter, loaded
                if len(result) - last_reported >= 128:
                    print(json.dumps({"event": "scoring_progress", "role": role, "version": version,
                                      "scored": len(result), "total": total,
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
                    last_reported = len(result)
        print(json.dumps({"event": "scoring_completed", "role": role, "version": version,
                          "scored": len(result), "total": total,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        return result

    def begin_update(self, expected_decisions=16):
        if type(expected_decisions) is not int or not 1 <= expected_decisions <= 16:
            raise ValueError("update requires 1 through 16 unique decisions")
        self._student()
        self.optimizer.zero_grad()
        self.decision_count = 0
        self.expected_decisions = expected_decisions

    def backward_decision(self, row, *, weight, divisor, seed, query_observer=None):
        if divisor != self.expected_decisions or weight < 0 or not math.isfinite(weight):
            raise ValueError("each loss must use the actual decision count and a finite nonnegative weight")
        query_observer = query_observer or (lambda phase: None)
        student = self._student()
        record = self._record(row)
        index = self.sample_query_index(self.sampling["num_inference_steps"],
            distribution="low_noise", seed=seed, beta_alpha=5., beta_beta=2.)
        with isolated_evaluation(student.model), self.adapter_module.autocast_for(student.model):
            query = student.collect_online_queries([record.condition],
                num_inference_steps=self.sampling["num_inference_steps"],
                sigma_shift=self.sampling.get("sigma_shift"), seeds=[seed], query_indices=[index],
                rand_device=self.sampling.get("rand_device", "cpu"), storage_device="cpu")
        # Teacher loading and inference must not advance the student's RNG stream.
        with isolated_evaluation():
            teacher = self._teacher(row["task_name"])
            with isolated_evaluation(teacher.model), self.adapter_module.autocast_for(teacher.model):
                condition = teacher.prepare_conditions([record.condition])
                query_observer("started")
                target = teacher.predict_velocity(
                    latents_action=query.query_latents.to(teacher.device, teacher.dtype),
                    timestep_action=query.query_timesteps.to(teacher.device, teacher.dtype),
                    condition=condition).detach().to(student.device, torch.float32)
                query_observer("completed")
            del condition
        with self.adapter_module.autocast_for(student.model):
            prediction = predict_action_with_checkpoint(student, record.condition,
                query.query_latents.detach().to(student.device, student.dtype),
                query.query_timesteps.detach().to(student.device, student.dtype),
                enabled=bool(self.config.get("activation_checkpointing", True)))
            loss = full_action_loss(prediction, target)
        self.optimizer.backward(loss * weight / divisor)
        self.decision_count += 1
        return {"action_mse": float(loss.detach()) / 10., "query_index": index,
                "query_count": 1}

    def finish_update(self):
        if self.decision_count != self.expected_decisions:
            raise ValueError("optimizer step requires every decision from the scheduled trajectory subset")
        return self.optimizer.step()

    def state_dict(self):
        self._student()
        return {"model": {key: value.detach().cpu() for key, value in self.student_model.state_dict().items()},
                "optimizer": self.optimizer.state_dict(), "rng": capture_rng(),
                "trainable_names": self.parameter_names}

    def load_state_dict(self, state):
        raise ValueError("PWTR training-state restore is unsupported")

    def save_checkpoint(self, path, payload):
        torch.save(payload, path)

    def export_model(self, path, *, completed_updates):
        self._student()
        # Export exact student-forward parameters in FastWAM's deployment schema.
        self.student_model.save_checkpoint(str(path), optimizer=None,
            step=completed_updates, base_checkpoint=self.config["initial_checkpoint"])

    def load_checkpoint(self, path):
        return _torch_load(path)
