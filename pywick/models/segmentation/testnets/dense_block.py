from typing import Optional

import torch
from torch.nn import Module

from .dense_layer import DenseLayer
from .utils import RichRepr


class DenseBlock(RichRepr, Module):
    r"""
    Dense Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet

    - Consists of several DenseLayer (possibly using a Bottleneck and Dropout) with the same output shape
    - The first DenseLayer is fed with the block input
    - Each subsequent DenseLayer is fed with a tensor obtained by concatenating the input and the output
      of the previous DenseLayer on the channel axis
    - The block output is the concatenation of the output of every DenseLayer, and optionally the block input,
      so it will have a channel depth of (growth_rate * num_layers) or (growth_rate * num_layers + in_channels)
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int,
                 concat_input: bool = False, dense_layer_params: Optional[dict] = None):
        super(DenseBlock, self).__init__()

        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = growth_rate * num_layers
        if self.concat_input:
            self.out_channels += self.in_channels

        if dense_layer_params is None:
            dense_layer_params = {}

        for i in range(num_layers):
            self.add_module(
                f'layer_{i}',
                DenseLayer(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, **dense_layer_params)
            )

    def forward(self, block_input):
        layer_input = block_input
        layer_output = torch.FloatTensor()  # empty

        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            # TODO check https://github.com/pytorch/pytorch/issues/5332
            # In version 0.3 pytorch can't concatenate empty variables, but is ok with empty tensors,
            # Check the issue to see how this develops
            if layer_output.dim() != 0:
                layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)

    def __repr__(self):
        concat_input = f'+{self.in_channels}' if self.concat_input else ''
        out_channels = f'{self.num_layers}*{self.growth_rate}{concat_input}={self.out_channels}'
        return super(DenseBlock, self).__repr__(self.in_channels, out_channels)
