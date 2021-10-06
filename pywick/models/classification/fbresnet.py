"""`Facebook implementation <https://github.com/facebook/fb.resnet.torch>`_ of ResNet"""

# Source: https://github.com/Cadene/pretrained-models.pytorch/blob/0819c4f43a70fcd40234b03ff02f87599cd8ace6/pretrainedmodels/models/fbresnet.py
# Note this is the version with adaptive capabilities so it can accept differently-sized images

import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['FBResNet', 'FBResNet18', 'FBResNet34', 'FBResNet50', 'FBResNet101', 'fbresnet152']

pretrained_settings = {
    'fbresnet152': {
        'imagenet': {
            # 'url': 'http://data.lip6.fr/cadene/pretrainedmodels/fbresnet152-2e20f6b4.pth',        # old version?
            'url': 'http://pretorched-x.csail.mit.edu/models/fbresnet152-3ade0e00.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input_):
        x = self.conv1(input_)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input_):
        x = self.features(input_)
        x = self.logits(x)
        return x


def FBResNet18(num_classes=1000):
    """Constructs a ResNet-18 model.

    Args:
        num_classes
    """
    model = FBResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def FBResNet34(num_classes=1000):
    """Constructs a ResNet-34 model.

    Args:
        num_classes
    """
    model = FBResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def FBResNet50(num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        num_classes
    """
    model = FBResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


def FBResNet101(num_classes=1000):
    """Constructs a ResNet-101 model.

    Args:
        num_classes
    """
    model = FBResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model


def fbresnet152(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['fbresnet152'][pretrained]
        if num_classes != settings['num_classes']:
            raise AssertionError("num_classes should be {}, but is {}".format(settings['num_classes'], num_classes))
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


