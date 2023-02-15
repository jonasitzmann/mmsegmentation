_base_ = [
    '../../configs/_base_/models/segformer_mit-b0.py', '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py', '../../configs/_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/mit_b0.pth',
    decode_head=dict(num_classes=150))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    optimizer=dict(type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

load_from = 'work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_160000.pth'
resume = False

train_dataloader = dict(batch_size=16, num_workers=16)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=500, val_begin=0)
