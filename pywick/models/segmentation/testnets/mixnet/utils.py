import re
from collections import namedtuple

BlockArgs = namedtuple('BlockArgs', [
    'dw_ksize', 'expand_ksize', 'project_ksize', 'num_repeat',
    'in_channels', 'out_channels', 'expand_ratio', 'id_skip',
    'strides', 'se_ratio', 'swish', 'dilated',
])


def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth depth_multiplier.
    TODO : ref link
    """
    if not depth_multiplier:
        return filters

    filters *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return new_filters


class MixnetDecoder:
    """A class of Mixnet decoder to get model configuration."""

    @staticmethod
    def _decode_block_string(block_string, depth_multiplier, depth_divisor, min_depth):
        """Gets a mixnet block through a string notation of arguments.

        E.g. r2_k3_a1_p1_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
        k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
        o - output filters, se - squeeze/excitation ratio

        Args:
        block_string: a string, a string representation of block arguments.

        Returns:
        A BlockArgs instance.
        Raises:
        ValueError: if the strides option is not correctly specified.
        """
        if not isinstance(block_string, str):
            raise AssertionError

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        def _parse_ksize(ss):
            ks = [int(k) for k in ss.split('.')]
            return ks if len(ks) > 1 else ks[0]

        return BlockArgs(num_repeat=int(options['r']),
                         dw_ksize=_parse_ksize(options['k']),
                         expand_ksize=_parse_ksize(options['a']),
                         project_ksize=_parse_ksize(options['p']),
                         strides=[int(options['s'][0]), int(options['s'][1])],
                         expand_ratio=int(options['e']),
                         in_channels=round_filters(int(options['i']), depth_multiplier, depth_divisor, min_depth),
                         out_channels=round_filters(int(options['o']), depth_multiplier, depth_divisor, min_depth),
                         id_skip=('noskip' not in block_string),
                         se_ratio=float(options['se']) if 'se' in options else 0,
                         swish=('sw' in block_string),
                         dilated=('dilated' in block_string)
                         )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a Mixnet block to a string."""

        def _encode_ksize(arr):
            return '.'.join([str(k) for k in arr])

        args = [
            'r%d' % block.num_repeat,
            'k%s' % _encode_ksize(block.dw_ksize),
            'a%s' % _encode_ksize(block.expand_ksize),
            'p%s' % _encode_ksize(block.project_ksize),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.in_channels,
            'o%d' % block.out_channels
        ]

        if (block.se_ratio is not None and block.se_ratio > 0 and block.se_ratio <= 1):
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        if block.swish:
            args.append('sw')
        if block.dilated:
            args.append('dilated')
        return '_'.join(args)

    @staticmethod
    def decode(string_list, depth_multiplier, depth_divisor, min_depth):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
        string_list: a list of strings, each string is a notation of Mixnet
        block.build_model_base

        Returns:
        A list of namedtuples to represent Mixnet blocks arguments.
        """
        if not isinstance(string_list, list):
            raise AssertionError
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(MixnetDecoder._decode_block_string(block_string, depth_multiplier, depth_divisor, min_depth))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encodes a list of Mixnet Blocks to a list of strings.

        Args:
        blocks_args: A list of namedtuples to represent Mixnet blocks arguments.
        Returns:
        a list of strings, each string is a notation of Mixnet block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(MixnetDecoder._encode_block_string(block))
        return block_strings
