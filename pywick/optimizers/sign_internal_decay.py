# Source: https://github.com/cydonia999/AddSign_PowerSign_in_PyTorch/tree/master/torch/optim

import math

class _SignInternalDecay(object):
    """Base class for internal decays for PowerSign and AddSign optimizers.

    Arguments:
        T_max (int): the total number of training steps
            to be used to compute internal decays.
    """
    def __init__(self, T_max):
        if T_max < 1:
            raise ValueError('T_max should be >= 1.')

        self.T_max = T_max

    
class LinearInternalDecay(_SignInternalDecay):
    """Implements a linear decay used internally in PowerSign and AddSign optimizers.

    It has been proposed in `Neural Optimizer Search with Reinforcement Learning`_.

    Arguments:
        T_max (int): the total number of training steps
            to be used to compute internal decays.

    .. _Neural Optimizer Search with Reinforcement Learning:
        https://arxiv.org/abs/1709.07417
    """
    def __init__(self, T_max):
        super(LinearInternalDecay, self).__init__(T_max)

    def __call__(self, step):
        """Returns a linear decay at the current training step:
            1 - step / T_max

        Args:
          step: the current training step.
        """
        if step is None:
            raise ValueError("step is required for linear_decay.")
        if step < 0:
            raise ValueError("step should be >= 0.")
        step = min(step, self.T_max)
        decay = 1 - float(step) / float(self.T_max)
        return decay


class CosineInternalDecay(_SignInternalDecay):
    """Implements a cyclical decay used internally in PowerSign and AddSign optimizers.

    It has been proposed in `Neural Optimizer Search with Reinforcement Learning`_.

    Arguments:
        T_max (int): the total number of training steps
            to be used to compute internal decays
        num_periods: number of periods of cosine from 0 to T_max (default: 0.5)
        zero_after: if not None, number after which 0 is returned

    .. _Neural Optimizer Search with Reinforcement Learning:
        https://arxiv.org/abs/1709.07417
    """
    def __init__(self, T_max, num_periods=0.5, zero_after=None):
        super(CosineInternalDecay, self).__init__(T_max)
        if zero_after is not None and zero_after < 0:
            raise ValueError("zero_after should be >= 0.")
        self.num_periods = num_periods
        self.zero_after = zero_after

    def __call__(self, step):
        """Returns a cyclical decay at the current training step:
            0.5 * (1 + cos(2 * pi * num_periods * step / T_max))

        Args:
          step: the current training step.
        """
        if step is None:
            raise ValueError("step is required for cosine_decay.")
        if step < 0:
            raise ValueError("step should be >= 0.")
        step = min(step, self.T_max)
        frac = 2.0 * self.num_periods * step / float(self.T_max)
        if self.zero_after is not None and frac >= 2 * self.zero_after:
            return 0.0
        decay = 0.5 * (1 + math.cos(math.pi * frac))
        return decay


class RestartCosineInternalDecay(_SignInternalDecay):
    """Implements a restart decay used internally in PowerSign and AddSign optimizers.

    It has been proposed in `Neural Optimizer Search with Reinforcement Learning`_.

    Arguments:
        T_max (int): the total number of training steps
            to be used to compute internal decays
        num_periods: number of half periods of cosine from 0 to T_max (default: 1)
        zero_after: if not None, number after which 0 is returned

    .. _Neural Optimizer Search with Reinforcement Learning:
        https://arxiv.org/abs/1709.07417
    """
    def __init__(self, T_max, num_periods=1, zero_after=None):
        super(RestartCosineInternalDecay, self).__init__(T_max)
        if zero_after is not None and zero_after < 0:
            raise ValueError("zero_after should be >= 0.")
        self.num_periods = num_periods
        self.zero_after = zero_after

    def __call__(self, step):
        """Returns a restart decay at the current training step:
            0.5 * (1 + cos(pi * (num_periods * step) % T_max / T_max))

        Args:
          step: the current training step.
        """
        if step is None:
            raise ValueError("step is required for cosine_decay.")
        if step < 0:
            raise ValueError("step should be >= 0.")
        step = min(step, self.T_max)
        frac = (self.num_periods * step) % self.T_max / float(self.T_max)
        if self.zero_after is not None and frac >= 2 * self.zero_after:
            return 0.0
        decay = 0.5 * (1 + math.cos(math.pi * frac))
        return decay
