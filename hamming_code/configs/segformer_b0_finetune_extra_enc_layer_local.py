_base_ = ['finetune_segformer_b0.py']
custom_imports = dict(imports=['hamming_code.data_preprocessor_enc', 'mmdet.models.losses'], allow_failed_imports=False)
optim_wrapper=dict(optimizer = dict(lr=1))
crop_size = (512, 512)
randomness=dict(seed=0)
num_classes = 150
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessorEncodedLabels',
    num_classes=num_classes
)
log_processor = dict(window_size=1)
model = dict(
    data_preprocessor=data_preprocessor,
    freeze_except_output_layer=True,
    decode_head=dict(
        # loss_decode=dict(
        #     type='MSELoss'
        # ),
        map_to_n_bits=8,
        num_classes=num_classes,
        encode_labels=True,
        encoding_conv_manual_weights=True
    ))
train_dataloader = dict(batch_size=4, num_workers=2)
val_dataloader=dict(dataset=dict(indices=[0]))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=10, val_begin=0)
default_hooks=dict(logger=dict(interval=1))
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={'decode_head.binary_encoding_conv': dict(lr_mult=100)}))

# default_hooks = dict(
#     _delete_=True,
#     logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
# )
#
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=2e-4))

# param_scheduler = []
