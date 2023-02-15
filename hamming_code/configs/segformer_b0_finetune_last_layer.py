_base_ = ['finetune_segformer']
model = dict(freeze_except_output_layer=True, remove_from_state_dict=['decode_head.conv_seg'])
