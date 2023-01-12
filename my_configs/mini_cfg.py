_base_ = [
    '../configs/_base_/datasets/ade20k.py', '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/tiny_schedule.py',
    '../configs/_base_/models/deeplabv3plus_r50-d8.py',
]

# model settings
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(type='MiniCnnBackbone'),
#     decode_head=dict(type='MiniCnnHead'),
#     test_cfg=dict(mode='whole')
#              )

model = dict(data_preprocessor=data_preprocessor,
             pretrained='open-mmlab://resnet18_v1c',
             backbone=dict(depth=18),
             decode_head=dict(
                 num_classes=150,
                 c1_in_channels=64,
                 c1_channels=12,
                 in_channels=512,
                 channels=128,
             ),
             auxiliary_head=dict(
                 num_classes=150,
                 in_channels=256,
                 channels=64
             ))

# pipeline = [
# dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='PackSegInputs')
# ]

dataloader = dict(dataset=dict(data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'), indices=2))
val_dataloader = dataloader
train_dataloader = dict(batch_size=2, **dataloader)
# vis_backend = dict(type='LocalVisBackend')
project = 'mmseg'
custom_hooks = [
    dict(type='MMSegWandbHook')
]
