import torch
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
try:
    import wandb
    project = 'mmseg'
    if not torch.cuda.is_available():
        project += '_cpu'
    vis_backend = dict(
        type='WandbVisBackend',
        init_kwargs=dict(entity='js0', project=project, reinit=True),
        watch_kwargs=dict(log_graph=True, log_freq=1),
    )
except ImportError:
    print('not using wandb')
    vis_backend = dict(type='LocalVisBackend')

visualizer = dict(type='SegLocalVisualizer', vis_backends=[vis_backend], name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
