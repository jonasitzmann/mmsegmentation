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
    # backbone=dict(
    #     init_cfg=dict(type='Pretrained', checkpoint='/notebooks/mscan_t.pth')),
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
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

dataset_type = 'ADE20KDataset'
# data_root = '/notebooks/ADEChallengeData2016'
data_root = '../data/ade/ADEChallengeData2016'
