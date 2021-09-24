import torch
import torch.nn as nn


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):
    def __init__(self, ch, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, squeeze_ch, 1, 1, 0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, ch, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))
