_base_ = ['segformer_b0_manual_extra_enc_layer.py']
model = dict(decode_head=dict(softmax_hardness=1e4))
