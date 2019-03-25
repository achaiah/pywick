from .drn import *
import torch.nn as nn
import math
import torch.nn.functional as F

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()

        if model_name == 'DRN_C_42':
            model = drn_c_42(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_C_58':
            model = drn_c_58(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_38':
            model = drn_d_38(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_54':
            model = drn_d_54(pretrained=pretrained, num_classes=1000)
        elif model_name == 'DRN_D_105':
            model = drn_d_105(pretrained=pretrained, num_classes=1000)

        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        self.use_torch_up = use_torch_up

        up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                output_padding=0, groups=classes,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

    def forward(self, x):
        base = self.base(x)
        final = self.seg(base)
        if self.use_torch_up:
            return F.interpolate(final, x.size()[2:], mode='bilinear')
        else:
            return self.up(final)

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]