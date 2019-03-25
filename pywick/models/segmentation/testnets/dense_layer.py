from typing import Optional

from torch.nn import Sequential, BatchNorm2d, ReLU, Conv2d, Dropout2d

from .bottleneck import Bottleneck
from .utils import RichRepr


class DenseLayer(RichRepr, Sequential):
    r"""
    Dense Layer as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet

    Consists of:

    - Batch Normalization
    - ReLU
    - (Bottleneck)
    - 3x3 Convolution
    - (Dropout)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck_ratio: Optional[int] = None, dropout: float = 0.0):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))

        if bottleneck_ratio is not None:
            self.add_module('bottleneck', Bottleneck(in_channels, bottleneck_ratio * out_channels))
            in_channels = bottleneck_ratio * out_channels

        self.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        if dropout > 0:
            self.add_module('drop', Dropout2d(dropout, inplace=True))

    def __repr__(self):
        return super(DenseLayer, self).__repr__(self.in_channels, self.out_channels)
