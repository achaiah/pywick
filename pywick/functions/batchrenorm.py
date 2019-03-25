# Source: https://pastebin.com/31J3JD7r

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class BatchReNorm1d(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, rmax=3.0, dmax=5.0):
        super(BatchReNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.rmax = rmax
        self.dmax = dmax

        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('r', torch.ones(num_features))
        self.register_buffer('d', torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.r.fill_(1)
        self.d.zero_()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input):
        self._check_input_dim(input)
        n = input.size()[0]

        if self.training:
            mean = torch.mean(input, dim=0)

            sum = torch.sum((input - mean.expand_as(input)) ** 2, dim=0)
            if sum == 0 and self.eps == 0:
                invstd = 0.0
            else:
                invstd = 1. / torch.sqrt(sum / n + self.eps)
            unbiased_var = sum / (n - 1)

            self.r = torch.clamp(torch.sqrt(unbiased_var).data / torch.sqrt(self.running_var),
                                 1. / self.rmax, self.rmax)
            self.d = torch.clamp((mean.data - self.running_mean) / torch.sqrt(self.running_var),
                                 -self.dmax, self.dmax)

            r = self.r.expand_as(input)
            d = self.d.expand_as(input)

            input_normalized = (input - mean.expand_as(input)) * invstd.expand_as(input)

            input_normalized = input_normalized * r + d

            self.running_mean += self.momentum * (mean.data - self.running_mean)
            self.running_var += self.momentum * (unbiased_var.data - self.running_var)

            if not self.affine:
                return input_normalized

            output = input_normalized * self.weight.expand_as(input)
            output += self.bias.unsqueeze(0).expand_as(input)

            return output

        else:
            mean = self.running_mean.expand_as(input)
            invstd = 1. / torch.sqrt(self.running_var.expand_as(input) + self.eps)

            input_normalized = (input - mean.expand_as(input)) * invstd.expand_as(input)

            if not self.affine:
                return input_normalized

            output = input_normalized * self.weight.expand_as(input)
            output += self.bias.unsqueeze(0).expand_as(input)

            return output

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                'affine={affine}, rmax={rmax}, dmax={dmax})'
                .format(name=self.__class__.__name__, **self.__dict__))