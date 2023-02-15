from mmengine.registry import MODELS
from mmseg.models import SegDataPreProcessor
from .encode import add_binary_encoding
from typing import Dict, Any
import numpy as np


@MODELS.register_module()
class SegDataPreProcessorEncodedLabels(SegDataPreProcessor):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_bits = int(np.ceil(np.log2(num_classes)))

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        preprocessed_data = super().forward(data, training)
        add_binary_encoding(preprocessed_data['data_samples'], self.num_bits)
        return preprocessed_data