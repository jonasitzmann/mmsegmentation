from mmengine.registry import MODELS
from mmseg.models import EncoderDecoder
import numpy as np
from mmseg.utils import ForwardResults, OptSampleList
from torch import Tensor
from .encode import add_binary_encoding
from mmseg.utils import SampleList
import torch


@MODELS.register_module()
class EncoderDecoderCodedLabels(EncoderDecoder):
    def __init__(self, decode_head, *args, **kwargs):
        num_classes = decode_head.num_classes
        super().__init__(decode_head=decode_head, *args, **kwargs)
        self.num_classes = num_classes

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        return super().forward(inputs=inputs, data_samples=data_samples, mode=mode)