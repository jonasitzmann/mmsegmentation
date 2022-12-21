_base_ = [
    'segnext_tiny_512x512_adamw_160k_ade20.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        # init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_s.pth')
    ),
    decode_head=dict(
        in_channels=[128, 320, 512],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
    ))