import re
from itertools import chain

from torch.nn import Module, Conv2d, ConvTranspose2d


def count_parameters(module: Module):
    """
    Counts the number of learnable parameters in a Module
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_conv2d(module: Module):
    """
    Counts the number of convolutions and transposed convolutions in a Module
    """
    return len([m for m in module.modules() if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d)])


class RichRepr(object):
    """
    Allows to modify the normal __repr__ output of a torch.nn.Module,
    adding info as positional and keyword arguments
    """
    def __repr__(self, *args, **kwargs):
        res = super(RichRepr, self).__repr__()
        args = filter(lambda s: len(s) > 0, map(str, args))
        kwargs = (f'{k}={v}' for k, v in kwargs.items())
        desc = ', '.join(chain(args, kwargs))
        return re.sub(rf'({self.__class__.__name__})', rf'\1({desc})', res, count=1)
