import warnings

from . import Callback


class LRScheduler(Callback):
    """
    Schedule the learning rate according to some function of the
    current epoch index, current learning rate, and current train/val loss.

    :param schedule: (callable):
        should return a number of learning rates equal to the number
        of optimizer.param_groups. It should take the epoch index and
        **kwargs (or logs) as argument. **kwargs (or logs) will return
        the epoch logs such as mean training and validation loss from
        the epoch
    """

    def __init__(self, schedule):
        if isinstance(schedule, dict):
            schedule = self.schedule_from_dict
            self.schedule_dict = schedule
            if any([k < 1.0 for k in schedule.keys()]):
                self.fractional_bounds = False
            else:
                self.fractional_bounds = True
        self.schedule = schedule
        super(LRScheduler, self).__init__()

    def schedule_from_dict(self, epoch, logs=None):
        for epoch_bound, learn_rate in self.schedule_dict.items():
            # epoch_bound is in units of "epochs"
            if not self.fractional_bounds:
                if epoch_bound < epoch:
                    return learn_rate
            # epoch_bound is in units of "cumulative percent of epochs"
            else:
                if epoch <= epoch_bound * logs['num_epoch']:
                    return learn_rate
        warnings.warn('Check the keys in the schedule dict.. Returning last value')
        return learn_rate

    def on_epoch_begin(self, epoch, logs=None):
        """
            WARNING: Do NOT use this callback with self-adjusting learners like Yellowfin
        """
        current_lrs = [p['lr'] for p in self.trainer._optimizer.param_groups]
        lr_list = self.schedule(epoch, current_lrs, **logs)
        if not isinstance(lr_list, list):
            lr_list = [lr_list]

        for param_group, lr_change in zip(self.trainer._optimizer.param_groups, lr_list):
            param_group['lr'] = lr_change