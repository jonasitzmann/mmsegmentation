default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
try:
    import wandb
    vis_backend = dict(type='WandbVisBackend', init_kwargs=dict(entity='js0', project='mmseg'))
except ImportError:
    print('not using wandb')
    vis_backend = dict(type='LocalVisBackend')

visualizer = dict(type='SegLocalVisualizer', vis_backends=[vis_backend], name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
