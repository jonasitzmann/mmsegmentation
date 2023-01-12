from torch import nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class MiniCnnHead(BaseDecodeHead):
    def __init__(self, num_classes=150, channels=4, *args, **kwargs):
        super().__init__(
            in_channels=16,
            channels=channels,
            out_channels=num_classes,
            num_classes=num_classes,
            *args, **kwargs
        )
        self.first_part = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, channels, 3, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.cls_seg(x)
        return x
