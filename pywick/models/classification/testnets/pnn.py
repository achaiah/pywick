# Source: https://github.com/juefeix/pnn.pytorch.update/blob/master/models.py (License: Apache 2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

def act_fn(act):
    if act == 'relu':
        act_ = nn.ReLU(inplace=False)
    elif act == 'lrelu':
        act_ = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_ = nn.PReLU()
    elif act == 'rrelu':
        act_ = nn.RReLU(inplace=True)
    elif act == 'elu':
        act_ = nn.ELU(inplace=True)
    elif act == 'selu':
        act_ = nn.SELU(inplace=True)
    elif act == 'tanh':
        act_ = nn.Tanh()
    elif act == 'sigmoid':
        act_ = nn.Sigmoid()
    else:
        print('\n\nActivation function {} is not supported/understood\n\n'.format(act))
        act_ = None
    return act_

""" ****************** Modified (Michael Klachko) PNN Implementation ******************* """


class PerturbLayerFirst(nn.Module):  # (2) Felix added this
    def __init__(self, in_channels=None, out_channels=None, nmasks=None, level=None, filter_size=None,
                 debug=False, use_act=False, stride=1, act=None, unique_masks=False, mix_maps=None,
                 train_masks=False, noise_type='uniform', input_size=None):
        super(PerturbLayerFirst, self).__init__()
        # self.nmasks = nmasks              #per input channel
        self.nmasks = nmasks
        self.unique_masks = unique_masks  # same set or different sets of nmasks per input channel
        self.train_masks = train_masks  # whether to treat noise masks as regular trainable parameters of the model
        self.level = level  # noise magnitude
        self.filter_size = filter_size  # if filter_size=0, layers=(perturb, conv_1x1) else layers=(conv_NxN), N=filter_size
        self.use_act = use_act  # whether to use activation immediately after perturbing input (set it to False for the first layer)
        # self.act = act_fn(act)            #relu, prelu, rrelu, elu, selu, tanh, sigmoid (see utils)
        self.act = act_fn('sigmoid')  # (6) Felix modified this, to sigmoid, only first layer, all-layer noise
        self.debug = debug  # print input, mask, output values for each batch
        self.noise_type = noise_type  # normal or uniform
        self.in_channels = in_channels
        self.input_size = input_size  # input image resolution (28 for MNIST, 32 for CIFAR), needed to construct masks
        self.mix_maps = mix_maps  # whether to apply second 1x1 convolution after perturbation, to mix output feature maps

        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False

        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                self.act
            )
        else:  # Felix: noise layer definitions here
            noise_channels = in_channels if self.unique_masks else 1
            shape = (1, noise_channels, self.nmasks, input_size, input_size)  # can't dynamically reshape masks in forward if we want to train them

            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=self.train_masks)
            if noise_type == "uniform":
                self.noise.data.uniform_(-1, 1)
                # Felix added: Gaussian blue the noise masks
                # self.noise = gaussian_filter(self.noise, sigma=2)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))

            if nmasks != 1:
                if out_channels % in_channels != 0:
                    print('\n\n\nnfilters must be divisible by 3 if using multiple noise masks per input channel\n\n\n')
                groups = in_channels
            else:
                groups = 1

            self.layers = nn.Sequential(
                nn.BatchNorm2d(in_channels * self.nmasks),  # (1) Felix modified this
                self.act,
                nn.Conv2d(in_channels * self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),
                nn.BatchNorm2d(out_channels),
                self.act,
            )
            if self.mix_maps:
                self.mix_layers = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=1),
                    nn.BatchNorm2d(out_channels),
                    self.act,
                )

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)  # image, conv, batchnorm, relu
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)
            # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)

            y = y.view(-1, self.in_channels * self.nmasks, self.input_size, self.input_size)
            y = self.layers(y)

            if self.mix_maps:
                y = self.mix_layers(y)

            return y  # image, perturb, (relu?), conv1x1, batchnorm, relu + mix_maps (conv1x1, batchnorm relu)


class PerturbLayer(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, nmasks=None, level=None, filter_size=None,
                 debug=False, use_act=False, stride=1, act=None, unique_masks=False, mix_maps=None,
                 train_masks=False, noise_type='uniform', input_size=None):
        super(PerturbLayer, self).__init__()
        self.nmasks = nmasks  # per input channel
        self.unique_masks = unique_masks  # same set or different sets of nmasks per input channel
        self.train_masks = train_masks  # whether to treat noise masks as regular trainable parameters of the model
        self.level = level  # noise magnitude
        self.filter_size = filter_size  # if filter_size=0, layers=(perturb, conv_1x1) else layers=(conv_NxN), N=filter_size
        self.use_act = use_act  # whether to use activation immediately after perturbing input (set it to False for the first layer)
        self.act = act_fn(act)  # relu, prelu, rrelu, elu, selu, tanh, sigmoid (see utils)
        self.debug = debug  # print input, mask, output values for each batch
        self.noise_type = noise_type  # normal or uniform
        self.in_channels = in_channels
        self.input_size = input_size  # input image resolution (28 for MNIST, 32 for CIFAR), needed to construct masks
        self.mix_maps = mix_maps  # whether to apply second 1x1 convolution after perturbation, to mix output feature maps

        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False

        if self.filter_size > 0:
            self.noise = None
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                self.act
            )
        else:
            noise_channels = in_channels if self.unique_masks else 1
            shape = (1, noise_channels, self.nmasks, input_size, input_size)  # can't dynamically reshape masks in forward if we want to train them
            self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=self.train_masks)
            if noise_type == "uniform":
                self.noise.data.uniform_(-1, 1)
            elif self.noise_type == 'normal':
                self.noise.data.normal_()
            else:
                print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))

            if nmasks != 1:
                if out_channels % in_channels != 0:
                    print('\n\n\nnfilters must be divisible by 3 if using multiple noise masks per input channel\n\n\n')
                groups = in_channels
            else:
                groups = 1

            self.layers = nn.Sequential(
                # self.act,      #TODO orig code uses ReLU here
                # nn.BatchNorm2d(out_channels), #TODO: orig code uses BN here
                nn.Conv2d(in_channels * self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),
                nn.BatchNorm2d(out_channels),
                self.act,
            )
            if self.mix_maps:
                self.mix_layers = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=1),
                    nn.BatchNorm2d(out_channels),
                    self.act,
                )

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)  # image, conv, batchnorm, relu
        else:
            y = torch.add(x.unsqueeze(2), self.noise * self.level)  # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)

            if self.use_act:
                y = self.act(y)

            y = y.view(-1, self.in_channels * self.nmasks, self.input_size, self.input_size)
            y = self.layers(y)

            if self.mix_maps:
                y = self.mix_layers(y)

            return y  # image, perturb, (relu?), conv1x1, batchnorm, relu + mix_maps (conv1x1, batchnorm relu)


class PerturbBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels=None, out_channels=None, stride=1, shortcut=None, nmasks=None, train_masks=False,
                 level=None, use_act=False, filter_size=None, act=None, unique_masks=False, noise_type=None,
                 input_size=None, pool_type=None, mix_maps=None):
        super(PerturbBasicBlock, self).__init__()
        self.shortcut = shortcut
        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            print('\n\nPool Type {} is not supported/understood\n\n'.format(pool_type))
            return
        self.layers = nn.Sequential(
            PerturbLayer(in_channels=in_channels, out_channels=out_channels, nmasks=nmasks, input_size=input_size,
                         level=level, filter_size=filter_size, use_act=use_act, train_masks=train_masks,
                         act=act, unique_masks=unique_masks, noise_type=noise_type, mix_maps=mix_maps),
            pool(stride, stride),
            PerturbLayer(in_channels=out_channels, out_channels=out_channels, nmasks=nmasks, input_size=input_size // stride,
                         level=level, filter_size=filter_size, use_act=use_act, train_masks=train_masks,
                         act=act, unique_masks=unique_masks, noise_type=noise_type, mix_maps=mix_maps),
        )

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class PerturbResNet(nn.Module):
    def __init__(self, block, nblocks=None, avgpool=None, nfilters=None, nclasses=None, nmasks=None, input_size=32,
                 level=None, filter_size=None, first_filter_size=None, use_act=False, train_masks=False, mix_maps=None,
                 act=None, scale_noise=1, unique_masks=False, debug=False, noise_type=None, pool_type=None):
        super(PerturbResNet, self).__init__()
        self.nfilters = nfilters
        self.unique_masks = unique_masks
        self.noise_type = noise_type
        self.train_masks = train_masks
        self.pool_type = pool_type
        self.mix_maps = mix_maps
        self.act = act_fn(act)  # (7) Felix added this

        # layers = [PerturbLayer(in_channels=3, out_channels=nfilters, nmasks=nmasks, level=level*scale_noise,
        # debug=debug, filter_size=first_filter_size, use_act=use_act, train_masks=train_masks, input_size=input_size,
        # act=act, unique_masks=self.unique_masks, noise_type=self.noise_type, mix_maps=mix_maps)]

        layers = [PerturbLayerFirst(in_channels=3, out_channels=3 * nfilters, nmasks=nfilters * 5, level=level * scale_noise * 20,
                                    debug=debug, filter_size=first_filter_size, use_act=use_act, train_masks=train_masks, input_size=input_size,
                                    act=act, unique_masks=self.unique_masks, noise_type=self.noise_type, mix_maps=mix_maps)]  # scale noise 20x at 1st layer
        # 256x3  X (5) = 3840
        # 128x3  X (5) = 1920

        # (3) Felix modified this

        if first_filter_size == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # self.pre_layers = nn.Sequential(*layers)
        self.pre_layers = nn.Sequential(*layers,  # (4) Felix modified this, not really necessary.
                                        nn.Conv2d(self.nfilters * 3 * 1, self.nfilters, kernel_size=1, stride=1, bias=False),
                                        # mapping 10*nfilters back to nfilters with 1x1 conv
                                        nn.BatchNorm2d(self.nfilters),
                                        self.act
                                        )

        self.layer1 = self._make_layer(block, 1 * nfilters, nblocks[0], stride=1, level=level, nmasks=nmasks, use_act=True,
                                       filter_size=filter_size, act=act, input_size=input_size)
        self.layer2 = self._make_layer(block, 2 * nfilters, nblocks[1], stride=2, level=level, nmasks=nmasks, use_act=True,
                                       filter_size=filter_size, act=act, input_size=input_size)
        self.layer3 = self._make_layer(block, 4 * nfilters, nblocks[2], stride=2, level=level, nmasks=nmasks, use_act=True,
                                       filter_size=filter_size, act=act, input_size=input_size // 2)
        self.layer4 = self._make_layer(block, 8 * nfilters, nblocks[3], stride=2, level=level, nmasks=nmasks, use_act=True,
                                       filter_size=filter_size, act=act, input_size=input_size // 4)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8 * nfilters * block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2, nmasks=None, use_act=False,
                    filter_size=None, act=None, input_size=None):
        shortcut = None
        if stride != 1 or self.nfilters != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.nfilters, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.nfilters, out_channels, stride, shortcut, level=level, nmasks=nmasks, use_act=use_act,
                            filter_size=filter_size, act=act, unique_masks=self.unique_masks, noise_type=self.noise_type,
                            train_masks=self.train_masks, input_size=input_size, pool_type=self.pool_type, mix_maps=self.mix_maps))
        self.nfilters = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.nfilters, out_channels, level=level, nmasks=nmasks, use_act=use_act,
                                train_masks=self.train_masks, filter_size=filter_size, act=act, unique_masks=self.unique_masks,
                                noise_type=self.noise_type, input_size=input_size // stride, pool_type=self.pool_type, mix_maps=self.mix_maps))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class LeNet(nn.Module):
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None, linear=128, input_size=28,
                 debug=False, scale_noise=1, act='relu', use_act=False, first_filter_size=None, pool_type=None,
                 dropout=None, unique_masks=False, train_masks=False, noise_type='uniform', mix_maps=None):
        super(LeNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4

        if input_size == 32:
            first_channels = 3
        elif input_size == 28:
            first_channels = 1

        if pool_type == 'max':
            pool = nn.MaxPool2d
        elif pool_type == 'avg':
            pool = nn.AvgPool2d
        else:
            print('\n\nPool Type {} is not supported/understood\n\n'.format(pool_type))
            return

        self.linear1 = nn.Linear(nfilters * n * n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=first_channels, out_channels=nfilters, nmasks=nmasks, level=level * scale_noise,
                         filter_size=first_filter_size, use_act=use_act, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=input_size, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),

            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks, debug=debug, train_masks=train_masks,
                         noise_type=noise_type, input_size=input_size // 2, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),

            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks, train_masks=train_masks, noise_type=noise_type,
                         input_size=input_size // 4, mix_maps=mix_maps),
            pool(kernel_size=3, stride=2, padding=1),
        )

        self.last_layers = nn.Sequential(
            self.dropout,
            self.linear1,
            self.batch_norm,
            self.act,
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        x = self.first_layers(x)
        x = x.view(x.size(0), -1)
        x = self.last_layers(x)
        return x



def perturb_resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=0,
                     pool_type=None, input_size=None, scale_noise=1, act='relu', use_act=True, dropout=0.5,
                     unique_masks=False, debug=False, noise_type='uniform', train_masks=False, mix_maps=None):
    return PerturbResNet(PerturbBasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses, pool_type=pool_type,
                         scale_noise=scale_noise, nmasks=nmasks, level=level, filter_size=filter_size, train_masks=train_masks,
                         first_filter_size=first_filter_size, act=act, use_act=use_act, unique_masks=unique_masks,
                         debug=debug, noise_type=noise_type, input_size=input_size, mix_maps=mix_maps)
