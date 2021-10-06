"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pad
import numpy as np


def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input_, kernel, groups, bias=None, stride=1, padding=0, dilation=1):
    n, c, h, w = input_.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    input_ = F.pad(input_, (pw_l, pw_r, ph_t, ph_b))
    result = F.conv2d(input_, kernel, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    if result.shape != input_.shape:
        raise AssertionError
    return result


def gradient_central_diff(input_):
    return input_, input_


def compute_single_sided_diferences(o_x, o_y, input_):
    # n,c,h,w
    #input = input.clone()
    o_y[:, :, 0, :] = input_[:, :, 1, :].clone() - input_[:, :, 0, :].clone()
    o_x[:, :, :, 0] = input_[:, :, :, 1].clone() - input_[:, :, :, 0].clone()
    # --
    o_y[:, :, -1, :] = input_[:, :, -1, :].clone() - input_[:, :, -2, :].clone()
    o_x[:, :, :, -1] = input_[:, :, :, -1].clone() - input_[:, :, :, -2].clone()
    return o_x, o_y


def numerical_gradients_2d(input_, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input_: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input_.shape
    if not (h > 1 and w > 1):
        raise AssertionError
    x, y = gradient_central_diff(input_)
    return x, y


def convTri(input_, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input_:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input_.shape
    return input_


def compute_normal(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        # import ipdb;
        # ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        # import ipdb;
        # ipdb.set_trace()

    return O


def compute_normal_2(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        # import ipdb;
        # ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        # import ipdb;
        # ipdb.set_trace()

    return O, (Oyy, Oxx)


def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox,Ox) + torch.mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max();

    return mag
