import torch.nn as nn

from .layers import Flatten
from .layers import SEModule
from .layers import Swish
from .mdconv import MDConv
from .utils import MixnetDecoder
from .utils import round_filters


class MixBlock(nn.Module):
    def __init__(self, dw_ksize, expand_ksize, project_ksize,
                 in_channels, out_channels, expand_ratio, id_skip,
                 strides, se_ratio, swish, dilated):
        super().__init__()

        self.id_skip = id_skip and all(s == 1 for s in strides) and in_channels == out_channels

        act_fn = lambda : Swish() if swish else nn.ReLU(True)

        layers = []
        expaned_ch = in_channels * expand_ratio
        if expand_ratio != 1:
            expand = nn.Sequential(
                nn.Conv2d(in_channels, expaned_ch, expand_ksize, bias=False),
                nn.BatchNorm2d(expaned_ch),
                act_fn(),
            )
            layers.append(expand)

        depthwise = nn.Sequential(
            MDConv(expaned_ch, dw_ksize, strides, bias=False),
            nn.BatchNorm2d(expaned_ch),
            act_fn(),
        )
        layers.append(depthwise)

        if se_ratio > 0:
            se = SEModule(expaned_ch, int(expaned_ch * se_ratio))
            layers.append(se)

        project = nn.Sequential(
            nn.Conv2d(expaned_ch, out_channels, project_ksize, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        layers.append(project)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.id_skip:
            out = out + x
        return out


class MixModule(nn.Module):
    def __init__(self, dw_ksize, expand_ksize, project_ksize, num_repeat,
                 in_channels, out_channels, expand_ratio, id_skip,
                 strides, se_ratio, swish, dilated):
        super().__init__()
        layers = [MixBlock(dw_ksize, expand_ksize, project_ksize,
                           in_channels, out_channels, expand_ratio, id_skip,
                           strides, se_ratio, swish, dilated)]

        for _ in range(num_repeat - 1):
            layers.append(MixBlock(dw_ksize, expand_ksize, project_ksize,
                                   in_channels, out_channels, expand_ratio, id_skip,
                                   [1, 1], se_ratio, swish, dilated))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MixNet(nn.Module):
    def __init__(self, stem, blocks_args, head, dropout_rate, num_classes=1000, **kwargs):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(True)
        )

        self.blocks = nn.Sequential(*[MixModule(*args) for args in blocks_args])

        self.classifier = nn.Sequential(
            nn.Conv2d(blocks_args[-1].out_channels, head, 1, bias=False),
            nn.BatchNorm2d(head),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(head, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # print("Input : ", x.shape)
        stem = self.stem(x)
        # print("Stem : ", x.shape)
        feature = self.blocks(stem)
        # print("feature : ", feature.shape)
        out = self.classifier(feature)
        return out


def mixnet_s(depth_multiplier=1, depth_divisor=8, min_depth=None, num_classes=1000):
    """
    Creates mixnet-s model.

    Args:
        depth_multiplier: depth_multiplier to number of filters per layer.
    """
    stem = round_filters(16,   depth_multiplier, depth_divisor, min_depth)
    head = round_filters(1536, depth_multiplier, depth_divisor, min_depth)
    dropout = 0.2

    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i16_o16',
        'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
        'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

        'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
        'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
        'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]

    blocks_args = MixnetDecoder.decode(blocks_args, depth_multiplier, depth_divisor, min_depth)
    print("-----------")
    print("Mixnet S")
    for a in blocks_args:
        print(a)
    print("-----------")
    return MixNet(stem, blocks_args, head, dropout, num_classes=num_classes)


if __name__ == "__main__":
    mixnet_s()
