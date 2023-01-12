from torch import nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class MiniCnnBackbone(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
