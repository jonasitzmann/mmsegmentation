_base_ = [
    'segnext_tiny_512x512_adamw_160k_ade20.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mscan_l.pth'),
        drop_path_rate=0.3),
    decode_head=dict(
        in_channels=[128, 320, 512],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=150
    )
)