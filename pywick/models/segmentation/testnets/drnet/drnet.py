# Source: https://github.com/cyj5030/DRC-Release

import torch.nn.functional as F
from .utils import *
from .backbone import *

__all__ = ['DRNet', 'DRCLoss']


class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    # nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x

class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel)
        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        if self.factor > 1:
            padding, out_padding = get_padding(x1.size(), x2.size(), self.factor)
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=padding,
                                    output_padding=out_padding)
        return x1 + x2

class Refine_block3_1(nn.Module):
    def __init__(self, in_channel, out_channel, require_grad=False):
        super(Refine_block3_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel)
        self.pre_conv3 = adap_conv(in_channel[2], out_channel)
        self.deconv_weight1 = nn.Parameter(bilinear_upsample_weights(2, out_channel), requires_grad=require_grad)
        self.deconv_weight2 = nn.Parameter(bilinear_upsample_weights(4, out_channel), requires_grad=require_grad)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x3 = self.pre_conv3(input[2])

        padding, out_padding = get_padding(x1.size(), x2.size(), 2)
        x2 = F.conv_transpose2d(x2, self.deconv_weight1, stride=2, padding=padding,
                                output_padding=out_padding)

        padding, out_padding = get_padding(x1.size(), x3.size(), 4)
        x3 = F.conv_transpose2d(x3, self.deconv_weight2, stride=4, padding=padding,
                                output_padding=out_padding)
        return x1 + x2 + x3


class decode(nn.Module):
    def __init__(self, backbone, dataset, **_):
        super(decode, self).__init__()
        if 'vgg' in backbone:
            ch = {'L1_1': [(64, 256), 64, 4],
                'L1_2': [(128, 512), 128, 4],
                'L1_3': [(256, 512), 256, 4]}
        else:
            if 'BSDS' in dataset:
                ch = {'L1_1': [(64, 512), 64, 4],
                    'L1_2': [(256, 1024), 128, 2],
                    'L1_3': [(512, 2048), 256, 1]}
            elif 'NYUD' in dataset:
                ch = {'L1_1': [(64, 512), 64, 4],
                    'L1_2': [(256, 1024), 128, 4],
                    'L1_3': [(512, 2048), 256, 2]}

        self.level1_1 = Refine_block2_1(ch['L1_1'][0], ch['L1_1'][1], ch['L1_1'][2], require_grad=False)
        self.level1_2 = Refine_block2_1(ch['L1_2'][0], ch['L1_2'][1], ch['L1_2'][2], require_grad=False)
        self.level1_3 = Refine_block2_1(ch['L1_3'][0], ch['L1_3'][1], ch['L1_3'][2], require_grad=False)

        self.level2 = Refine_block3_1((64, 128, 256), 64, require_grad=False)
        self.level3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *x):
        level1 = []
        level1 += [self.level1_1(x[0], x[2])]
        level1 += [self.level1_2(x[1], x[3])]
        level1 += [self.level1_3(x[2], x[4])]
        level2 = self.level2(*level1)

        return self.level3(level2)


class DRNet(nn.Module):
    """
    @article{cao2020learning,
            author = {Cao, Yi-Jun and Lin, Chuan and Li, Yong-Jie},
            year = {2020},
            title = {Learning Crisp Boundaries Using Deep Refinement Network and Adaptive Weighting Loss},
            journal = {IEEE Transactions on Multimedia},
            doi = {10.1109/TMM.2020.2987685}
    }

    dataset_name options: {'BSDS', 'NYUD'}
    """
    def __init__(self, backbone_name='resnext101', dataset_name='BSDS', **_):            # dataset_name options: {'BSDS', 'NYUD'}
        super(DRNet, self).__init__()

        self.encode = backbone(backbone_name, dataset=dataset_name)
        self.decode = decode(backbone=backbone_name, dataset=dataset_name)

    def forward(self, x):
        end_points = self.encode(x)
        end_points = [end_points['feat1'], end_points['feat2'], end_points['feat3'], end_points['feat4'], end_points['feat5']]
        x = self.decode(*end_points)
        return x


class DRCLoss(nn.Module):
    def __init__(self, cfg_c=0.1, **_):
        super(DRCLoss, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]))
        self.weight2 = nn.Parameter(torch.Tensor([1.]))
        self.cfg_c = cfg_c

    def forward(self, pred, labels):
        pred = pred.sigmoid()
        total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
                 self.weight2.pow(-2) * self.cfg_c * dice_loss_per_image(pred, labels) + \
                 (1 + self.weight1 * self.weight2).log()

        return total_loss


def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss

def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_with_weight(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6 # 1e-6 is the good choise if smaller than 1e-6, it may appear NaN
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def get_weight(src, mask, threshold, weight):
    count_pos = src[mask >= threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))