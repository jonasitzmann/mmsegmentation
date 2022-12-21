_base_ = [
    'segnext_tiny_512x512_adamw_160k_ade20.py'
]
# model settings
model = dict(
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        # init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_b.pth'),
        drop_path_rate=0.1),
    decode_head=dict(
        in_channels=[128, 320, 512],
        channels=512,
        ham_channels=512,
    ))