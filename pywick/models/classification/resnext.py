# Source: https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/pretrainedmodels/models/resnext.py (License: BSD-3-Clause)
# Pretrained: Yes

"""
Implementation of paper: `Aggregated Residual Transformations for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .resnext_features import resnext50_32x4d_features
from .resnext_features import resnext101_32x4d_features
from .resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt50_32x4d', 'resnext50_32x4d',
           'ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']

pretrained_settings = {
    'resnext50_32x4d': {
        'imagenet': {
            'url': 'https://github.com/barrh/pretrained-models.pytorch/releases/download/v0.7.4.1/resnext50_32x4d-b86d1c04b9.pt',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_32x4d-29e315fa.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'resnext101_64x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}

class ResNeXt50_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt50_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext50_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input_):
        x = self.avg_pool(input_)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input_):
        x = self.features(input_)
        x = self.logits(x)
        return x

class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input_):
        x = self.avg_pool(input_)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input_):
        x = self.features(input_)
        x = self.logits(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def logits(self, input_):
        x = self.avg_pool(input_)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input_):
        x = self.features(input_)
        x = self.logits(x)
        return x

def resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    """Pretrained Resnext50_32x4d model"""
    model = ResNeXt50_32x4d(num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['resnext50_32x4d'][pretrained]
        if num_classes != settings['num_classes']:
            raise AssertionError("num_classes should be {}, but is {}".format(settings['num_classes'], num_classes))
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


def resnext101_32x4d(pretrained='imagenet'):
    """Pretrained Resnext101_32x4d model"""
    model = ResNeXt101_32x4d(num_classes=1000)
    if pretrained:
        settings = pretrained_settings['resnext101_32x4d'][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def resnext101_64x4d(pretrained='imagenet'):
    """Pretrained ResNeXt101_64x4d model"""
    model = ResNeXt101_64x4d(num_classes=1000)
    if pretrained:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model