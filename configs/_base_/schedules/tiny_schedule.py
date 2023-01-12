# optimizer
max_iters = 1000
interval = 20
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam'),
)
# training schedule for 5k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU', save_last=True, interval=50),
)
