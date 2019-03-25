from typing import Optional

import torch
from torch.nn import Module, ConvTranspose2d

from .utils import RichRepr


class TransitionUp(RichRepr, Module):
    r"""
    Transition Up Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326)

    The block upsamples the feature map and concatenates it with the feature map coming from the skip connection.
    If the two maps don't overlap perfectly they are first aligened centrally and cropped to match.
    """

    def __init__(self, upsample_channels: int, skip_channels: Optional[int] = None):
        r"""
        :param upsample_channels: number of channels from the upsampling path
        :param skip_channels: number of channels from the skip connection, it is not required,
                              but if specified allows to statically compute the number of output channels
        """
        super(TransitionUp, self).__init__()

        self.upsample_channels = upsample_channels
        self.skip_channels = skip_channels
        self.out_channels = upsample_channels + skip_channels if skip_channels is not None else None

        self.add_module('upconv', ConvTranspose2d(self.upsample_channels, self.upsample_channels,
                                                  kernel_size=3, stride=2, padding=0, bias=True))
        self.add_module('concat', CenterCropConcat())

    def forward(self, upsample, skip):
        if self.skip_channels is not None and skip.shape[1] != self.skip_channels:
            raise ValueError(f'Number of channels in the skip connection input ({skip.shape[1]}) '
                             f'is different from the expected number of channels ({self.skip_channels})')
        res = self.upconv(upsample)
        res = self.concat(res, skip)
        return res

    def __repr__(self):
        skip_channels = self.skip_channels if self.skip_channels is not None else "?"
        out_channels = self.out_channels if self.out_channels is not None else "?"
        return super(TransitionUp, self).__repr__(f'[{self.upsample_channels}, {skip_channels}] -> {out_channels})')


class CenterCropConcat(Module):
    def forward(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y inputs contain a different number of samples')
        height = min(x.size(2), y.size(2))
        width = min(x.size(3), y.size(3))

        x = self.center_crop(x, height, width)
        y = self.center_crop(y, height, width)

        res = torch.cat([x, y], dim=1)
        return res

    @staticmethod
    def center_crop(x, target_height, target_width):
        current_height = x.size(2)
        current_width = x.size(3)
        min_h = (current_width - target_width) // 2
        min_w = (current_height - target_height) // 2
        return x[:, :, min_w:(min_w + target_height), min_h:(min_h + target_width)]
