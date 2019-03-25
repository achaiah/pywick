from typing import Optional, Sequence, Union

from torch.nn import Module, Conv2d, BatchNorm2d, Linear, init
from torch.nn import functional as F

from .transition_down import TransitionDown
from .transition_up import TransitionUp
from .dense_block import DenseBlock


class FCDenseNet(Module):
    r"""
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326

    In this paper, we extend DenseNets to deal with the problem of semantic segmentation. We achieve state-of-the-art
    results on urban scene benchmark datasets such as CamVid and Gatech, without any further post-processing module nor
    pretraining. Moreover, due to smart construction of the model, our approach has much less parameters than currently
    published best entries for these datasets.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1000,
                 initial_num_features: int = 48,
                 dropout: float = 0.2,

                 down_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 down_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 down_dense_num_layers: Union[int, Sequence[int]] = (4, 5, 7, 10, 12),
                 down_transition_compression_factors: Union[float, Sequence[float]] = 1.0,

                 middle_dense_growth_rate: int = 16,
                 middle_dense_bottleneck: Optional[int] = None,
                 middle_dense_num_layers: int = 15,

                 up_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 up_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 up_dense_num_layers: Union[int, Sequence[int]] = (12, 10, 7, 5, 4)):
        super(FCDenseNet, self).__init__()

        # region Parameters handling
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(down_dense_growth_rates) == int:
            down_dense_growth_rates = (down_dense_growth_rates,) * 5
        if down_dense_bottleneck_ratios is None or type(down_dense_bottleneck_ratios) == int:
            down_dense_bottleneck_ratios = (down_dense_bottleneck_ratios,) * 5
        if type(down_dense_num_layers) == int:
            down_dense_num_layers = (down_dense_num_layers,) * 5
        if type(down_transition_compression_factors) == float:
            down_transition_compression_factors = (down_transition_compression_factors,) * 5

        if type(up_dense_growth_rates) == int:
            up_dense_growth_rates = (up_dense_growth_rates,) * 5
        if up_dense_bottleneck_ratios is None or type(up_dense_bottleneck_ratios) == int:
            up_dense_bottleneck_ratios = (up_dense_bottleneck_ratios,) * 5
        if type(up_dense_num_layers) == int:
            up_dense_num_layers = (up_dense_num_layers,) * 5
        # endregion

        # region First convolution
        # The Lasagne implementation uses convolution with 'same' padding, the PyTorch equivalent is padding=1
        self.features = Conv2d(in_channels, initial_num_features, kernel_size=3, padding=1, bias=False)
        current_channels = self.features.out_channels
        # endregion

        # region Downward path
        # Pairs of Dense Blocks with input concatenation and TransitionDown layers
        down_dense_params = [
            {
                'concat_input': True,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(down_dense_growth_rates, down_dense_num_layers, down_dense_bottleneck_ratios)
        ]
        down_transition_params = [
            {
                'dropout': dropout,
                'compression': c
            } for c in down_transition_compression_factors
        ]
        skip_connections_channels = []

        self.down_dense = Module()
        self.down_trans = Module()
        down_pairs_params = zip(down_dense_params, down_transition_params)
        for i, (dense_params, transition_params) in enumerate(down_pairs_params):
            block = DenseBlock(current_channels, **dense_params)
            current_channels = block.out_channels
            self.down_dense.add_module(f'block_{i}', block)

            skip_connections_channels.append(block.out_channels)

            transition = TransitionDown(current_channels, **transition_params)
            current_channels = transition.out_channels
            self.down_trans.add_module(f'trans_{i}', transition)
        # endregion

        # region Middle block
        # Renamed from "bottleneck" in the paper, to avoid confusion with the Bottleneck of DenseLayers
        self.middle = DenseBlock(
            current_channels,
            middle_dense_growth_rate,
            middle_dense_num_layers,
            concat_input=True,
            dense_layer_params={
                'dropout': dropout,
                'bottleneck_ratio': middle_dense_bottleneck
            })
        current_channels = self.middle.out_channels
        # endregion

        # region Upward path
        # Pairs of TransitionUp layers and Dense Blocks without input concatenation
        up_transition_params = [
            {
                'skip_channels': sc,
            } for sc in reversed(skip_connections_channels)
        ]
        up_dense_params = [
            {
                'concat_input': False,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(up_dense_growth_rates, up_dense_num_layers, up_dense_bottleneck_ratios)
        ]

        self.up_dense = Module()
        self.up_trans = Module()
        up_pairs_params = zip(up_transition_params, up_dense_params)
        for i, (transition_params_up, dense_params_up) in enumerate(up_pairs_params):
            transition = TransitionUp(current_channels, **transition_params_up)
            current_channels = transition.out_channels
            self.up_trans.add_module(f'trans_{i}', transition)

            block = DenseBlock(current_channels, **dense_params_up)
            current_channels = block.out_channels
            self.up_dense.add_module(f'block_{i}', block)
        # endregion

        # region Final convolution
        self.final = Conv2d(current_channels, out_channels, kernel_size=1, bias=False)
        # endregion

        # region Weight initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform(module.weight)
                init.constant(module.bias, 0)
        # endregion

    def forward(self, x):
        res = self.features(x)

        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans.children()):
            res = dense(res)
            skip_tensors.append(res)
            res = trans(res)

        res = self.middle(res)

        for skip, trans, dense in zip(reversed(skip_tensors), self.up_trans.children(), self.up_dense.children()):
            res = trans(res, skip)
            res = dense(res)

        res = self.final(res)

        return res

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits)
