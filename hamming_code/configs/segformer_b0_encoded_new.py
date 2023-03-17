_base_ = [
    '../../configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py']
custom_imports = dict(imports='hamming_code.data_preprocessor_enc', allow_failed_imports=False)
num_classes = 150
data_preprocessor = dict(type='SegDataPreProcessorEncodedLabels', num_classes=num_classes)
model = dict(data_preprocessor=data_preprocessor, decode_head=dict(
        num_classes=num_classes,
        encode_labels=True
))
resume = False
train_cfg = dict(val_begin=0)
