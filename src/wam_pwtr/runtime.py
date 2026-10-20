"""Sampled-prefix PWTR training and inference-checkpoint export."""
import os
from pathlib import Path
from wam_opd_v2.fresh_runtime import FreshTrainingRunner, durable_json
from wam_opd_v2.runtime import file_digest
from .core import SPEC, reweight


class Runner(FreshTrainingRunner):
    def __init__(self,view,schedule,backend,output,*,probabilities,references,experiment_id):
        super().__init__(view,schedule,backend,output,seed=0,probabilities=probabilities,
            reweight=reweight,refresh_interval=24,checkpoint_every=96,strength=.1,
            experiment_id=experiment_id,arm_id='sampled_prefix_s0',weighting_spec=SPEC,
            verified_references=references)

    def resume(self,path):
        raise ValueError('Training-state resume is unsupported by this entry')

    def checkpoint(self):
        if self.completed_updates not in (0,96):raise ValueError('only initial/final save permitted')
        super().checkpoint()
        if self.completed_updates==96:self.export_for_evaluation()

    def export_for_evaluation(self):
        if self.completed_updates!=96 or self.checkpoint_updates!=96:raise ValueError('final complete save required')
        source=self.run_dir/'checkpoint.pt';complete=self._complete_save(source)
        directory=self.run_dir/'exports/update_000096'
        directory.mkdir(parents=True,exist_ok=False)
        snapshot=directory/'training_checkpoint.pt'
        os.link(source,snapshot)  # Share checkpoint storage through a hard link.
        durable_json(directory/'training_checkpoint.receipt.json',complete)
        model=directory/'model_weights.pt';tmp=directory/'.model_weights.pt.tmp'
        self.backend.export_model(tmp,completed_updates=96)
        with tmp.open('r+b') as f:os.fsync(f.fileno())
        digest=file_digest(tmp);os.replace(tmp,model)
        durable_json(directory/'model_weights.receipt.json',dict(schema='wam_pwtr_release_export_v1',
            experiment_id=self.fresh_identity['experiment_id'],completed_updates=96,
            checkpoint=str(model),checkpoint_sha256=digest,source_checkpoint=str(snapshot),
            source_checkpoint_sha256=complete['checkpoint_sha256'],identities=self.identities,
            query_accounting=self.query_accounting()))
