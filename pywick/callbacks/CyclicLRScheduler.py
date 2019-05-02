# Source: https://github.com/anandsaha/pytorch.cyclic.learning.rate (MIT)
# Good description of how it functions is here: https://github.com/bckenstler/CLR

# This code is from https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py
# This code is under review at PyTorch and is to be merged eventually to make CLR available to all.
# Tested with pytorch 0.2.0

from torch.optim.optimizer import Optimizer
import numpy as np
from . import Callback
from ..misc import trun_n_d

widegap_scale_fn = lambda x: 1/(5**(x*0.0001))

class CyclicLRScheduler(Callback):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    :param optimizer: (Optimizer):
        Wrapped optimizer.
    :param base_lr: (float or list):
        Initial learning rate which is the
        lower boundary in the cycle for eachparam groups.
        Default: 0.001
    :param max_lr: (float or list): Upper boundaries in the cycle for
        each parameter group. Functionally,
        it defines the cycle amplitude (max_lr - base_lr).
        The lr at any cycle is the sum of base_lr
        and some scaling of the amplitude; therefore
        max_lr may not actually be reached depending on
        scaling function. Default: 0.006
    :param step_size: (int): Number of training iterations per
        half cycle. Authors suggest setting step_size
        2-8 x training iterations in epoch. Default: 2000
    :param mode: (str): One of {triangular, triangular2, exp_range}.
        Values correspond to policies detailed above.
        If scale_fn is not None, this argument is ignored.
        Default: 'triangular'
    :param gamma: (float): Constant in 'exp_range' scaling function:
        gamma**(cycle iterations)
        Default: 1.0
    :param scale_fn: (function): Custom scaling policy defined by a single
        argument lambda function, where
        0 <= scale_fn(x) <= 1 for all x >= 0.
        mode paramater is ignored
        Default: None
    :param scale_mode: (str): {'cycle', 'iterations'}.
        Defines whether scale_fn is evaluated on
        cycle number or cycle iterations (training
        iterations since start of cycle).
        Default: 'cycle'
    :param verbose: (bool): Whether to produce some output during initialization
        Default: True

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', verbose=True):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if verbose:
            print('CyclicLRScheduler params:')
            print('\tstep_size: {}'.format(step_size))
            print('\tmode: {}'.format(mode))
            print('\tbase_lr: {}'.format(base_lr))
            print('\tmax_lr: {}'.format(max_lr))

        if mode not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.last_batch_iteration = 0
        self.epoch_count = 0

        self.optimizer_name = optimizer.__class__.__name__.lower()

    def on_batch_end(self, batch, logs=None):
        if 'yellowfin' in self.optimizer_name:
            computed_lr = [self.optimizer._optimizer.param_groups[0]['lr']]  # this is because trainer history expects a list
        else:
            computed_lr = self.get_lr()                 # returns a list
            self.last_batch_iteration = self.last_batch_iteration + 1               # global iteration counter
            for param_group, lr in zip(self.optimizer.param_groups, computed_lr):
                param_group['lr'] = lr

        if self.trainer.history is not None:
            for i,lr in enumerate(computed_lr):
                computed_lr[i] = trun_n_d(lr.item(), 5)         # .item() is a numpy way of obtaining a float

            self.trainer.history.lrs = computed_lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))       # cycle number is based on global batch counter
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
