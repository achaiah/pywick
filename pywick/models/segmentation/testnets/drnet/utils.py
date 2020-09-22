import numpy as np
import torch

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)

def get_padding(output_size, input_size, factor):
    TH = output_size[2] - ((input_size[2]-1)*factor) - (factor*2)
    TW = output_size[3] - ((input_size[3]-1)*factor) - (factor*2)
    padding_H = int(np.ceil(TH / (-2)))
    out_padding_H = TH - padding_H*(-2)

    padding_W = int(np.ceil(TW / (-2)))
    out_padding_W = TW - padding_W*(-2)
    return (padding_H, padding_W), (out_padding_H, out_padding_W)

def cfgs2name(cfgs):
    name = '%s_%s_%s(%s,%s,%s)' % \
            (cfgs['dataset'], cfgs['backbone'], cfgs['loss'], cfgs['a'], cfgs['b'],cfgs['c'])
    if 'MultiCue' in cfgs['dataset']:
        name = name + '_' + str(cfgs['multicue_seq'])
    return name