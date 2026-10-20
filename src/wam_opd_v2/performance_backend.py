"""Fixed B1 scoring with two resident task teachers for PWTR."""
from __future__ import annotations
import gc
from pathlib import Path

import torch

from .fastwam_backend import FastWAMBackend, isolated_evaluation


class PerformanceBackend(FastWAMBackend):
    resident_teacher_limit = 2

    def __init__(self, config, manifest, *, seed=0):
        if type(config.get("score_batch_size")) is not int or config["score_batch_size"] != 1:
            raise ValueError("PWTR requires explicit B1 scoring")
        teacher_paths = {str(Path(spec["checkpoint"]).resolve()) for spec in config["teachers"]}
        if len(teacher_paths) != self.resident_teacher_limit:
            raise ValueError("PWTR requires the two fixed task teachers")
        self._fixed_teacher_paths = teacher_paths
        self._resident_teacher_adapters = {}
        super().__init__(config, manifest, seed=seed)

    def score_records(self, role, records, version):
        if self.config["score_batch_size"] != 1:
            raise ValueError("PWTR scoring batch size cannot change")
        return super().score_records(role, records, version)

    def state_dict(self):
        # Preserve the B1 save inventory while rejecting all alternative modes.
        flags = dict(configured_batch_size=1, effective_batch_size=1,
                     validation_completed=False)
        return dict(super().state_dict(), performance_scoring=flags)

    def load_state_dict(self, state):
        raise ValueError("PWTR training-state restore is unsupported")

    @staticmethod
    def _release_unused_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _teacher(self, task):
        spec, = [spec for spec in self.teacher_specs if task in spec["tasks"]]
        return self._auxiliary(spec["checkpoint"])

    def _auxiliary(self, checkpoint):
        key = str(Path(checkpoint).resolve())
        if key not in self._fixed_teacher_paths:
            if key != str(Path(self.config["initial_checkpoint"]).resolve()):
                raise ValueError("auxiliary model must be the fixed initial model or a declared teacher")
            # Initial-reference scoring and resident teachers are mutually
            # exclusive, so initialization cannot leave a third owned model.
            if self._resident_teacher_adapters:
                self._resident_teacher_adapters.clear()
                self._release_unused_memory()
            with isolated_evaluation():
                return super()._auxiliary(checkpoint)
        if key in self._resident_teacher_adapters:
            return self._resident_teacher_adapters[key]
        if len(self._resident_teacher_adapters) >= self.resident_teacher_limit:
            raise RuntimeError("teacher cache exceeded its fixed model inventory")
        if self.aux_model is not None or self.aux_adapter is not None:
            self.aux_model = self.aux_adapter = self.aux_key = None
            self._release_unused_memory()
        # The loader may initialize temporary module parameters. Preserve the
        # caller RNG even when the first request is outside a scoring wrapper.
        with isolated_evaluation():
            model = self._new_model(checkpoint, self.teacher_device)
            adapter = self.adapter_module.FastWAMActionAdapter(model)
        self._resident_teacher_adapters[key] = adapter
        return adapter
