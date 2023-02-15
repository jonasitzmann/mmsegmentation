_base_ = ['segformer_finetune_extra_enc_layer.py']
model = dict(decode_head=dict(encoding_conv_manual_weights=True))
