_base_ = ['finetune_segformer_b0.py']
custom_imports = dict(imports='hamming_code.data_preprocessor_enc', allow_failed_imports=False)
crop_size = (512, 512)
num_classes = 150
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessorEncodedLabels',
    num_classes=num_classes
)
model = dict(
    data_preprocessor=data_preprocessor,
    freeze_except_output_layer=True,
    decode_head=dict(
        map_to_n_bits=8,
        encode_labels=True,
))