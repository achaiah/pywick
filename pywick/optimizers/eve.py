# Source: https://github.com/moskomule/eve.pytorch

import math
from torch.optim import Optimizer


class Eve(Optimizer):
    """
    Implementation of `Eve:  A Gradient Based Optimization Method with Locally and Globally Adaptive Learning Rates <https://arxiv.org/pdf/1611.01505.pdf>`_
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), eps=1e-8, k=0.1, K=10, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        k=k, K=K, weight_decay=weight_decay)
        super(Eve, self).__init__(params, defaults)

    def step(self, closure):
        """
        :param closure: (closure). see http://pytorch.org/docs/optim.html#optimizer-step-closure
        :return: loss
        """
        loss = closure()
        _loss = loss.item()  # float

        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m_t'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['v_t'] = grad.new().resize_as_(grad).zero_()
                    # f hats, smoothly tracked objective functions
                    # \hat{f}_0 = f_0
                    state['ft_2'], state['ft_1'] = _loss, None
                    state['d'] = 1

                m_t, v_t = state['m_t'], state['v_t']
                beta1, beta2, beta3 = group['betas']
                k, K = group['k'], group['K']
                d = state['d']
                state['step'] += 1
                t = state['step']
                # initialization of \hat{f}_1
                if t == 1:
                    # \hat{f}_1 = f_1
                    state['ft_1'] = _loss
                # \hat{f_{t-1}}, \hat{f_{t-2}}
                ft_1, ft_2 = state['ft_1'], state['ft_2']
                # f(\theta_{t-1})
                f = _loss

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                m_t.mul_(beta1).add_(1 - beta1, grad)
                v_t.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                m_t_hat = m_t / (1 - beta1 ** t)
                v_t_hat = v_t / (1 - beta2 ** t)

                if t > 1:
                    if f >= state['ft_2']:
                        delta = k + 1
                        Delta = K + 1
                    else:
                        delta = 1 / (K + 1)
                        Delta = 1 / (k + 1)

                    c = min(max(delta, f / ft_2), Delta)
                    r = abs(c - 1) / min(c, 1)
                    state['ft_1'], state['ft_2'] = c * ft_2, ft_1
                    state['d'] = beta3 * d + (1 - beta3) * r

                # update parameters
                p.data.addcdiv_(-group['lr'] / state['d'],
                                m_t_hat,
                                v_t_hat.sqrt().add_(group['eps']))

        return loss
