import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class ConvBNReLU(nn.Module):
    def __init__(self, in_size, out_size, norm, kernel_size=3, stride=1, padding=1, act=nn.ReLU):
        super(ConvBNReLU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                   norm(out_size),
                                   act(inplace=True),)
        # initialise the blocks
        for m in self.children():
            m.apply(weights_init_kaiming)
        
    def forward(self, inputs):
        return self.conv1(inputs)

class UnetConv2D(nn.Module):
    def __init__(self, in_size, out_size, norm, kernel_size=3, stride=1, padding=1, act=nn.ReLU):
        super(UnetConv2D, self).__init__()
        self.conv1 = ConvBNReLU(in_size, out_size, norm, kernel_size, stride, padding, act)
        self.conv2 = ConvBNReLU(out_size, out_size, norm, kernel_size, 1, padding, act)


    def forward(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)

class UnetUpConv2D(nn.Module):
    def __init__(self, in_size, out_size, norm, is_deconv=True, act=nn.ReLU):
        super(UnetUpConv2D, self).__init__()

        self.conv = UnetConv2D(in_size, out_size, norm, act=act)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2D') != -1: 
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset  = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output  = torch.cat([output1, output2], 1)
        return self.conv(output)
