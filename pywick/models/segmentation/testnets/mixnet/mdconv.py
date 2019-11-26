# https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_channels(total_filters, num_groups):
    """
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py#L33
    """
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


class MDConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dilatied=False, bias=False):
        super().__init__()

        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes]

        self.in_channels  = _split_channels(in_channels, len(kernel_sizes))

        self.convs = nn.ModuleList()
        for ch, k in zip(self.in_channels, kernel_sizes):
            dilation = 1
            if stride[0] == 1 and dilatied:
                dilation, stride = (k - 1) // 2, 3
                print("Use dilated conv with dilation rate = {}".format(dilation))
            pad = ((stride[0] - 1) + dilation * (k - 1)) // 2

            conv = nn.Conv2d(ch, ch, k, stride, pad, dilation,
                             groups=ch, bias=bias)
            self.convs.append(conv)

    def forward(self, x):
        xs = torch.split(x, self.in_channels, 1)
        return torch.cat([conv(x) for conv, x in zip(self.convs, xs)], 1)