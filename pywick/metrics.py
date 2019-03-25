import torch
from .utils import th_matrixcorr
from .meters.averagemeter import AverageMeter
from .callbacks import Callback
from .losses import lovaszloss, hingeloss, dice_coeff

def is_iterable(x):
    return isinstance(x, (tuple, list))

class MetricContainer(object):
    def __init__(self, metrics, prefix=''):
        self.metrics = metrics
        self.helper = None
        self.prefix = prefix

    def set_helper(self, helper):
        self.helper = helper

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __call__(self, input_batch, output_batch, target_batch, is_val=False):
        logs = {}
        for metric in self.metrics:
            # logs[self.prefix+metric._name] = self.helper.calculate_loss(output_batch, target_batch, metric)
            metric_out = metric(input_batch, output_batch, target_batch, is_val)
            if metric_out is not None:
                logs[self.prefix + metric._name] = metric_out
        return logs

class Metric(object):

    def __call__(self, inputs, y_pred, y_true, is_val):
        '''

        :param y_pred: Predictions from doing the forward pass
        :param y_true: Ground Truth
        :param is_val: Whether this is a validation pass (otherwise assumed training pass)

        :return:
        '''
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')


class MetricCallback(Callback):

    def __init__(self, container):
        self.container = container
    def on_epoch_begin(self, epoch_idx, logs):
        self.container.reset()

class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self._name = 'top_'+str(top_k)+':acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, inputs, y_pred, y_true, is_val=False):
        top_k = y_pred.topk(self.top_k,1)[1]
        true_k = y_true.view(len(y_true),1).expand_as(top_k)
        self.correct_count += top_k.eq(true_k).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class CategoricalAccuracySingleInput(CategoricalAccuracy):
    '''
    This class is a tiny modification of CategoricalAccuracy to handle the issue when we desire a single output but
    the network outputs multiple y_pred (e.g. inception)
    '''
    def __init__(self, top_k=1):
        super().__init__(top_k)

    def __call__(self, inputs, y_pred, y_true, is_val=False):
        if is_iterable(y_pred):
            return super().__call__(inputs, y_pred[0], y_true, is_val=False)
        else:
            return super().__call__(inputs, y_pred, y_true, is_val=False)


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, inputs, y_pred, y_true, is_val):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class ProjectionCorrelation(Metric):

    def __init__(self):
        self.corr_sum = 0.
        self.total_count = 0.

        self._name = 'corr_metric'

    def reset(self):
        self.corr_sum = 0.
        self.total_count = 0.

    def __call__(self, inputs, y_pred, y_true=None, is_val=False):
        """
        y_pred should be two projections
        """
        # covar_mat = torch.abs(th_matrixcorr(y_pred[0].data, y_pred[1].data))       # changed after pytorch 0.4
        covar_mat = torch.abs(th_matrixcorr(y_pred[0].detach(), y_pred[1].detach()))
        self.corr_sum += torch.trace(covar_mat)
        self.total_count += covar_mat.size(0)
        return self.corr_sum / self.total_count


class ProjectionAntiCorrelation(Metric):

    def __init__(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

        self._name = 'anticorr_metric'

    def reset(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

    def __call__(self, inputs, y_pred, y_true=None, is_val=False):
        """
        y_pred should be two projections
        """
        # covar_mat = torch.abs(th_matrixcorr(y_pred[0].data, y_pred[1].data))
        covar_mat = torch.abs(th_matrixcorr(y_pred[0].detach(), y_pred[1].detach()))       # changed after pytorch 0.4
        upper_sum = torch.sum(torch.triu(covar_mat,1))
        lower_sum = torch.sum(torch.tril(covar_mat,-1))
        self.anticorr_sum += upper_sum
        self.anticorr_sum += lower_sum
        self.total_count += covar_mat.size(0)*(covar_mat.size(1) - 1)
        return self.anticorr_sum / self.total_count


class DiceCoefficientMetric(Metric):
    '''
    Calculates the Dice Coefficient (typically used for image segmentation)
    '''
    def __init__(self, is_binary=True, run_on_val_only=False):
        '''
        :param is_binary: (default: True) Whether this is binary segmentation.
        :param run_on_val_only: (default: False) Whether we only want this to execute during evaluation loop
        '''
        self._name = 'dice_coeff'
        self.run_on_val_only = run_on_val_only
        self.dices = AverageMeter()
        self.is_binary = is_binary

    def reset(self):
        self.dices.reset()

    def __call__(self, inputs, y_pred, y_true, is_val):
        N = y_pred.size(0) * y_pred.size(2) * y_pred.size(3)
        if not self.run_on_val_only or (is_val and self.run_on_val_only):
            if self.is_binary:      # need to transpose into 0-1 range
                y_pred = torch.sigmoid(y_pred)
            # self.dices.update(dice_coeff(y_pred, y_true).data[0], N)
            self.dices.update(dice_coeff(y_pred, y_true).item(), N)     # changed after pytorch 0.4
            return self.dices.avg
        else:
            return -1337.0


class JaccardLossMetric(Metric):
    '''
    Calculates the Jaccard Loss (typically used for image segmentation)
    '''
    def __init__(self, run_on_val_only=False):
        '''
        :param is_binary: (default: True) Whether this is binary segmentation.
        :param run_on_val_only: (default: False) Whether we only want this to execute during evaluation loop
        '''
        self._name = 'jacc_loss'
        self.run_on_val_only = run_on_val_only
        self.jaccard = AverageMeter()

    def reset(self):
        self.jaccard.reset()

    def __call__(self, inputs, y_pred, y_true, is_val):
        N = y_pred.size(0) * y_pred.size(2) * y_pred.size(3)
        if not self.run_on_val_only or (is_val and self.run_on_val_only):
            # self.jaccard.update(lovaszloss(y_pred, y_true.data).data[0], N)     # changed after pytorch 0.4
            self.jaccard.update(lovaszloss(y_pred, y_true).item(), N)
            return self.jaccard.avg
        else:
            return -1337.0

class HingeLossMetric(Metric):
    '''
    Calculates the Hinge Loss (typically used for image segmentation)
    '''
    def __init__(self, run_on_val_only=False):
        '''
        :param is_binary: (default: True) Whether this is binary segmentation.
        :param run_on_val_only: (default: False) Whether we only want this to execute during evaluation loop
        '''
        self._name = 'hinge_loss'
        self.run_on_val_only = run_on_val_only
        self.hinge = AverageMeter()

    def reset(self):
        self.hinge.reset()

    def __call__(self, inputs, y_pred, y_true, is_val):
        N = y_pred.size(0) * y_pred.size(2) * y_pred.size(3)
        if not self.run_on_val_only or (is_val and self.run_on_val_only):
            # self.hinge.update(hingeloss(y_pred, y_true.data).data[0], N)    # changed after pytorch 0.4
            self.hinge.update(hingeloss(y_pred, y_true), N)
            return self.hinge.avg
        else:
            return -1337.0
