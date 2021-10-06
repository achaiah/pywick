# Source: https://github.com/pytorch/pytorch/pull/3740

import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD


class SGDW(Optimizer):
    r"""Implements stochastic gradient descent warm (optionally with momentum).

    It has been proposed in `Fixing Weight Decay Regularization in Adam <https://arxiv.org/abs/1711.05101>`_.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_.

    :param params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    :param lr: (float): learning rate
    :param momentum: (float, optional): momentum factor (default: 0)
    :param weight_decay: (float, optional): weight decay (L2 penalty) (default: 0)
    :param dampening: (float, optional): dampening for momentum (default: 0)
    :param nesterov: (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target).backward()
        >>> optimizer.step()

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.003, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        :param closure: (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if weight_decay != 0:
                    p.data.add_(-weight_decay, p.data)

        return loss
