# Source: https://github.com/rwightman/pytorch-dpn-pretrained (License: Apache 2.0)
# Pretrained: Yes

import os
import argparse
import torch
from .model_factory import create_model

try:
    import mxnet
    has_mxnet = True
except ImportError:
    has_mxnet = False


def _convert_bn(k):
    aux = False
    if k == 'bias':
        add = 'beta'
    elif k == 'weight':
        add = 'gamma'
    elif k == 'running_mean':
        aux = True
        add = 'moving_mean'
    elif k == 'running_var':
        aux = True
        add = 'moving_var'
    else:
        assert False, 'Unknown key: %s' % k
    return aux, add


def convert_from_mxnet(model, checkpoint_prefix, debug=False):
    _, mxnet_weights, mxnet_aux = mxnet.model.load_checkpoint(checkpoint_prefix, 0)
    remapped_state = {}
    for state_key in model.state_dict().keys():
        k = state_key.split('.')
        aux = False
        mxnet_key = ''
        if k[-1] == 'num_batches_tracked':
            continue
        if k[0] == 'features':
            if k[1] == 'conv1_1':
                # input block
                mxnet_key += 'conv1_x_1__'
                if k[2] == 'bn':
                    mxnet_key += 'relu-sp__bn_'
                    aux, key_add = _convert_bn(k[3])
                    mxnet_key += key_add
                else:
                    assert k[3] == 'weight'
                    mxnet_key += 'conv_' + k[3]
            elif k[1] == 'conv5_bn_ac':
                # bn + ac at end of features block
                mxnet_key += 'conv5_x_x__relu-sp__bn_'
                assert k[2] == 'bn'
                aux, key_add = _convert_bn(k[3])
                mxnet_key += key_add
            else:
                # middle blocks
                if model.b and 'c1x1_c' in k[2]:
                    bc_block = True  # b-variant split c-block special treatment
                else:
                    bc_block = False
                ck = k[1].split('_')
                mxnet_key += ck[0] + '_x__' + ck[1] + '_'
                ck = k[2].split('_')
                mxnet_key += ck[0] + '-' + ck[1]
                if ck[1] == 'w' and len(ck) > 2:
                    mxnet_key += '(s/2)' if ck[2] == 's2' else '(s/1)'
                mxnet_key += '__'
                if k[3] == 'bn':
                    mxnet_key += 'bn_' if bc_block else 'bn__bn_'
                    aux, key_add = _convert_bn(k[4])
                    mxnet_key += key_add
                else:
                    ki = 3 if bc_block else 4
                    assert k[ki] == 'weight'
                    mxnet_key += 'conv_' + k[ki]
        elif k[0] == 'classifier':
            if 'fc6-1k_weight' in mxnet_weights:
                mxnet_key += 'fc6-1k_'
            else:
                mxnet_key += 'fc6_'
            mxnet_key += k[1]
        else:
            assert False, 'Unexpected token'

        if debug:
            print(mxnet_key, '=> ', state_key, end=' ')

        mxnet_array = mxnet_aux[mxnet_key] if aux else mxnet_weights[mxnet_key]
        torch_tensor = torch.from_numpy(mxnet_array.asnumpy())
        if k[0] == 'classifier' and k[1] == 'weight':
            torch_tensor = torch_tensor.view(torch_tensor.size() + (1, 1))
        remapped_state[state_key] = torch_tensor

        if debug:
            print(list(torch_tensor.size()), torch_tensor.mean(), torch_tensor.std())

    model.load_state_dict(remapped_state)

    return model

parser = argparse.ArgumentParser(description='MXNet to PyTorch DPN conversion')
parser.add_argument('checkpoint_path', metavar='DIR', help='path to mxnet checkpoints')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')


def main():
    args = parser.parse_args()
    if 'dpn' not in args.model:
        print('Error: Can only convert DPN models.')
        exit(1)
    if not has_mxnet:
        print('Error: Cannot import MXNet module. Please install.')
        exit(1)

    model = create_model(args.model, num_classes=1000, pretrained=False)

    model_prefix = args.model
    if model_prefix in ['dpn107', 'dpn68b', 'dpn92']:
        model_prefix += '-extra'
    checkpoint_base = os.path.join(args.checkpoint_path, model_prefix)
    convert_from_mxnet(model, checkpoint_base)

    output_checkpoint = os.path.join(args.checkpoint_path, model_prefix + '.pth')
    torch.save(model.state_dict(), output_checkpoint)


if __name__ == '__main__':
    main()
