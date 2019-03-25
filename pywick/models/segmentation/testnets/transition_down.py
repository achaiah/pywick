from math import ceil

from torch.nn import Sequential, BatchNorm2d, ReLU, Conv2d, Dropout2d, MaxPool2d

from .utils import RichRepr


class TransitionDown(RichRepr, Sequential):
    r"""
    Transition Down Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326),
    plus compression from [DenseNet](https://arxiv.org/abs/1608.06993)

    Consists of:
    - Batch Normalization
    - ReLU
    - 1x1 Convolution (with optional compression of the number of channels)
    - (Dropout)
    - 2x2 Max Pooling
    """

    def __init__(self, in_channels: int, compression: float = 1.0, dropout: float = 0.0):
        super(TransitionDown, self).__init__()

        if not 0.0 < compression <= 1.0:
            raise ValueError(f'Compression must be in (0, 1] range, got {compression}')

        self.in_channels = in_channels
        self.dropout = dropout
        self.compression = compression
        self.out_channels = int(ceil(compression * in_channels))

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False))

        if dropout > 0:
            self.add_module('drop', Dropout2d(dropout))

        self.add_module('pool', MaxPool2d(kernel_size=2, stride=2))

    def __repr__(self):
        return super(TransitionDown, self).__repr__(self.in_channels, self.out_channels, dropout=self.dropout)
