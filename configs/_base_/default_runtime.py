import pathlib
import platform
import sys
import os
import wandb

runner_type = 'SlurmRunner'
resume = True
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
project = 'mmseg'
is_local = platform.system() == 'Windows'
vis_backend = dict(type='LocalVisBackend')
if is_local:
    project += '_win'
else:
    vis_backend = dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='js0',
            project=project,
            reinit=True,
            name=pathlib.Path(sys.argv[1]).stem,
            notes=os.environ.get('SLURM_JOB_ID', ''),
            resume=True,
        ),
        watch_kwargs=dict(log_graph=True, log_freq=1),
    )

visualizer = dict(type='SegLocalVisualizer', vis_backends=[vis_backend], name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None