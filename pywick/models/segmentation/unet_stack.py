# Source: https://github.com/doodledood/carvana-image-masking-challenge/models (MIT)

"""
Implementation of stacked `U-net: Convolutional networks for biomedical image segmentation <https://arxiv.org/pdf/1505.04597>`_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReluStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ConvBNReluStack, self).__init__()

        in_dim = int(in_dim)
        out_dim = int(out_dim)

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        # nn.init.xavier_normal(self.conv.weight.data)

        self.bn = nn.BatchNorm2d(out_dim)
        self.activation = nn.PReLU()  # nn.LeakyReLU(0.2)

    def forward(self, inputs_):
        x = self.conv(inputs_)
        x = self.bn(x)
        x = self.activation(x)

        return x


class UNetDownStack(nn.Module):
    def __init__(self, input_dim, filters, pool=True):
        super(UNetDownStack, self).__init__()

        self.stack1 = ConvBNReluStack(input_dim, filters, 1, stride=1, padding=0)
        self.stack3 = ConvBNReluStack(input_dim, filters, 3, stride=1, padding=1)
        self.stack5 = ConvBNReluStack(input_dim, filters, 5, stride=1, padding=2)
        self.stack_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.reducer = ConvBNReluStack(filters * 3 + input_dim, filters, kernel_size=1, stride=1, padding=0)

        # self.pool = ConvBNReluStack(filters, filters, kernel_size, stride=2, padding=1) if pool else None
        self.pool = nn.MaxPool2d(2, stride=2) if pool else None
        # ConvBNReluStack(filters, filters, kernel_size, stride=2, padding=1) if pool else None
        # nn.MaxPool2d(2, stride=2) if pool else None

    def forward(self, inputs_):
        x1 = self.stack1(inputs_)
        x3 = self.stack3(inputs_)
        x5 = self.stack5(inputs_)
        x_pool = self.stack_pool(inputs_)

        x = torch.cat([x1, x3, x5, x_pool], dim=1)
        x = self.reducer(x)

        if self.pool:
            return x, self.pool(x)

        return x


class UNetUpStack(nn.Module):
    def __init__(self, input_dim, filters, kernel_size=3):
        super(UNetUpStack, self).__init__()

        self.scale_factor = 2
        self.stack1 = ConvBNReluStack(input_dim, filters, 1, stride=1, padding=0)
        self.stack3 = ConvBNReluStack(input_dim, filters, 3, stride=1, padding=1)
        self.stack5 = ConvBNReluStack(input_dim, filters, 5, stride=1, padding=2)
        self.stack_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.reducer = ConvBNReluStack(filters * 3 + input_dim, filters, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs_, down):
        x = F.interpolate(inputs_, scale_factor=self.scale_factor)
        x = torch.cat([x, down], dim=1)

        x1 = self.stack1(x)
        x3 = self.stack3(x)
        x5 = self.stack5(x)
        x_pool = self.stack_pool(x)

        x = torch.cat([x1, x3, x5, x_pool], dim=1)
        x = self.reducer(x)

        return x


class UNet_stack(nn.Module):
    def get_n_stacks(self, input_size):
        n_stacks = 0
        width, height = input_size
        while width % 2 == 0 and height % 2 == 0:
            n_stacks += 1
            width = width // 2
            height = height // 2

        return n_stacks

    def __init__(self, input_size, filters, kernel_size=3, max_stacks=6):
        super(UNet_stack, self).__init__()

        self.n_stacks = min(self.get_n_stacks(input_size), max_stacks)

        # dynamically create stacks
        self.down1 = UNetDownStack(3, filters)
        prev_filters = filters
        for i in range(2, self.n_stacks + 1):
            n = i
            layer = UNetDownStack(prev_filters, prev_filters * 2)
            layer_name = 'down' + str(n)
            setattr(self, layer_name, layer)
            prev_filters *= 2

        self.center = UNetDownStack(prev_filters, prev_filters * 2, pool=False)

        prev_filters = prev_filters * 3
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            layer = UNetUpStack(prev_filters, prev_filters // 3, kernel_size)
            layer_name = 'up' + str(n)
            setattr(self, layer_name, layer)
            prev_filters = prev_filters // 2

        self.classify = nn.Conv2d(prev_filters * 2 // 3, 1, kernel_size, stride=1, padding=1)
        # nn.init.xavier_normal(self.classify.weight.data)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)

        downs = [down1]

        # execute down nodes
        prev_down_pool = down1_pool
        for i in range(2, self.n_stacks + 1):
            layer_name = 'down' + str(i)
            layer = getattr(self, layer_name)
            down, prev_down_pool = layer(prev_down_pool)
            downs.append(down)

        center = self.center(prev_down_pool)

        # excute up nodes
        prev = center
        for i in range(self.n_stacks):
            n = self.n_stacks - i
            matching_down = downs.pop()
            layer_name = 'up' + str(n)
            layer = getattr(self, layer_name)
            prev = layer(prev, matching_down)

        x = self.classify(prev)

        return x


class UNet960(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super(UNet960, self).__init__()

        # 960
        self.down1 = UNetDownStack(3, filters)
        # 480
        self.down2 = UNetDownStack(filters, filters * 2)
        # 240
        self.down3 = UNetDownStack(filters * 2, filters * 4)
        # 120
        self.down4 = UNetDownStack(filters * 4, filters * 8)
        # 60
        self.down5 = UNetDownStack(filters * 8, filters * 16)
        # 30
        self.down6 = UNetDownStack(filters * 16, filters * 32)
        # 15
        self.center = UNetDownStack(filters * 32, filters * 64, pool=False)
        # 15
        self.up6 = UNetUpStack(filters * 96, filters * 32, kernel_size)
        # 30
        self.up5 = UNetUpStack(filters * 48, filters * 16, kernel_size)
        # 60
        self.up4 = UNetUpStack(filters * 24, filters * 8, kernel_size)
        # 120
        self.up3 = UNetUpStack(filters * 12, filters * 4, kernel_size)
        # 240
        self.up2 = UNetUpStack(filters * 6, filters * 2, kernel_size)
        # 480
        self.up1 = UNetUpStack(filters * 3, filters, kernel_size)
        # 960
        self.classify = nn.Conv2d(filters, 1, kernel_size, stride=1, padding=1)

    def forward(self, inputs_):
        down1, down1_pool = self.down1(inputs_)
        down2, down2_pool = self.down2(down1_pool)
        down3, down3_pool = self.down3(down2_pool)
        down4, down4_pool = self.down4(down3_pool)
        down5, down5_pool = self.down5(down4_pool)
        down6, down6_pool = self.down6(down5_pool)

        center = self.center(down6_pool)

        up6 = self.up6(center, down6)
        up5 = self.up5(up6, down5)
        up4 = self.up4(up5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        x = self.classify(up1)

        return x
