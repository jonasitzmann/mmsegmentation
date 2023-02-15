from mmengine.runner import Runner
from mmengine.registry import RUNNERS
import os


@RUNNERS.register_module()
class SlurmRunner(Runner):
    @property
    def timestamp(self):
        return os.environ.get('SLURM_JOB_ID', super().timestamp)
