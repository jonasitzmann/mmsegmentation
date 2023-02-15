_base_ = [
    '../../_base_/models/segnext.py',
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/schedules/schedule_160k.py',
]
find_unused_parameters = True
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='pretrain/mscan_t.pth')),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
# 0.00006 is the lr for bs 16, should use 0.00006/8 as lr (need to test)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)

dataset_type = 'ADE20KDataset'
data_root = '../data/ade/ADEChallengeData2016'
