_base_ = ['segformer_b0_finetune_extra_enc_layer.py']
model = dict(
    freeze_except_output_layer=True,
    decode_head=dict(
        encoding_conv_manual_weights=True,
))
