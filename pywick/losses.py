"""
Losses are critical to training a neural network well. The training can only make progress if you
provide a meaningful measure of loss for each training step. What the loss looks like usually depends
on your application. Pytorch has a number of `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions/>`_ that
you can use out of the box. However, some more advanced and cutting edge loss functions exist that are not (yet) part of
Pytorch. We include those below for your experimenting.\n
**Caution:** if you decide to use one of these, you will definitely want to peruse the source code first, as it has
many additional useful notes and references which will help you.

Keep in mind that losses are specific to the type of task. Classification losses are computed differently from Segmentation losses.
Within segmentation domain make sure to use BCE (Binary Cross Entropy) for any work involving binary masks (e.g. num_classes = 1)
Make sure to read the documentation and notes (in the code) for each loss to understand how it is applied.

`Read this blog post <https://gombru.github.io/2018/05/23/cross_entropy_loss/>`_

Note:
    Logit is the vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function.
    If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates
    a vector of (normalized) probabilities with one value for each possible class.

For example, BCEWithLogitsLoss is a BCE that accepts R((-inf, inf)) and automatically applies torch.sigmoid to convert it to ([0,1]) space.

However, if you use one-hot encoding or similar methods where you need to convert a tensor to pytorch from another source (e.g. numpy), you will need to
make sure to apply the correct type to the resulting tensor.  E.g. If y_hot is of type long and the BCE loss expects a Tensor of type float then you
can try converting y_hot with y_hot = y_hot.type_as(output).

To convert predictions into (0,1) range you will sometimes need to use either softmax or sigmoid.
Softmax is used for multi-classification in the Logistic Regression model, whereas Sigmoid is used for binary classification in the Logistic Regression model
"""

##  Various loss calculation functions  ##
# Sources:  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py (?)
#           https://github.com/doodledood/carvana-image-masking-challenge/blob/master/losses.py (MIT)
#           https://github.com/atlab/attorch/blob/master/attorch/losses.py (MIT)
#           https://github.com/EKami/carvana-challenge (MIT)
#           https://github.com/DingKe/pytorch_workplace (MIT)


import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch import Tensor
from typing import Iterable, Set


__all__ = ['ActiveContourLoss', 'ActiveContourLossAlt', 'AngularPenaltySMLoss', 'AsymLoss', 'BCELoss2d', 'BCEDiceLoss',
           'BCEWithLogitsViewLoss', 'BCEDiceTL1Loss', 'BCEDicePenalizeBorderLoss', 'BCEDiceFocalLoss', 'BinaryFocalLoss',
           'ComboBCEDiceLoss', 'ComboSemsegLossWeighted', 'EncNetLoss', 'FocalLoss', 'FocalLoss2',
           'HausdorffERLoss', 'HausdorffDTLoss', 'LovaszSoftmax', 'mIoULoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'MSE3D', 'OhemCELoss', 'OhemCrossEntropy2d', 'OhemBCEDicePenalizeBorderLoss', 'PoissonLoss',
           'PoissonLoss3d', 'RecallLoss', 'RMILoss', 'RMILossAlt', 'RMIBCEDicePenalizeBorderLoss', 'SoftInvDiceLoss',
           'SoftDiceLoss', 'StableBCELoss', 'TverskyLoss', 'ThresholdedL1Loss', 'WeightedSoftDiceLoss', 'WeightedBCELoss2d',
           'BDLoss', 'L1Loss3d', 'WingLoss', 'BoundaryLoss']

VOID_LABEL = 255
N_CLASSES = 1


class StableBCELoss(nn.Module):
    def __init__(self, **_):
        super(StableBCELoss, self).__init__()

    @staticmethod
    def forward(input_, target, **_):
        neg_abs = - input_.abs()
        loss = input_.clamp(min=0) - input_ * target + (1 + neg_abs.exp()).log()
        return loss.mean()


# WARN: Only applicable to Binary Segmentation!
def binaryXloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, label)
    target = label.contiguous().view(-1)[mask]
    logits = logits.contiguous().view(-1)[mask]
    # loss = F.binary_cross_entropy(logits, target.float())
    loss = StableBCELoss()(logits, target.float())
    return loss


def naive_single(logit, label):
    # single images
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logit.sum() * 0.
    target = label.contiguous().view(-1)[mask].float()
    logit = logit.contiguous().view(-1)[mask]
    prob = torch.sigmoid(logit)
    intersect = target * prob
    union = target + prob - intersect
    loss = (1. - intersect / union).sum()
    return loss


# WARN: Only applicable to Binary Segmentation!
def hingeloss(logits, label):
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum().item()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logits.sum().item() * 0.
    target = label.contiguous().view(-1)[mask]
    target = 2. * target.float() - 1.  # [target == 0] = -1
    logits = logits.contiguous().view(-1)[mask]
    hinge = 1. / num_preds * F.relu(1. - logits * target).sum().item()
    return hinge


def gamma_fast(gt, permutation):
    p = len(permutation)
    gt = gt.gather(0, permutation)
    gts = gt.sum()

    intersection = gts - gt.float().cumsum(0)
    union = gts + (1 - gt).float().cumsum(0)
    jaccard = 1. - intersection / union

    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# WARN: Only applicable to Binary Segmentation right now (zip function needs to be replaced)!
def lovaszloss(logits, labels, prox=False, max_steps=20, debug=None):
    """
    `The Lovasz-Softmax loss <https://arxiv.org/abs/1705.08790>`_

    :param logits:
    :param labels:
    :param prox:
    :param max_steps:
    :param debug:
    :return:
    """
    if debug is None:
        debug = {}

    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = lovasz_single(logits.squeeze(0), labels.squeeze(0), prox, max_steps, debug)
    else:
        losses = []
        # assert len(logits[0]) == len(labels[0])
        for logit, label in zip(logits, labels):
            loss = lovasz_single(logit, label, prox, max_steps, debug)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss


def naiveloss(logits, labels):
    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = naive_single(logits.squeeze(0), labels.squeeze(0))
    else:
        losses = []
        for logit, label in zip(logits, labels):
            loss = naive_single(logit, label)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss


def iouloss(pred, gt):
    # works for one binary pred and associated target
    # make byte tensors
    pred = (pred == 1)
    mask = (gt != 255)
    gt = (gt == 1)
    union = (gt | pred)[mask].long().sum()
    if not union:
        return 0.
    else:
        intersection = (gt & pred)[mask].long().sum()
        return 1. - intersection / union


def compute_step_length(x, grad, active, eps=1e-6):
    # compute next intersection with an edge in the direction grad
    # OR next intersection with a 0 - border
    # returns: delta in ind such that:
    # after a step delta in the direction grad, x[ind] and x[ind+1] will be equal
    delta = np.inf
    ind = -1
    if active > 0:
        numerator = (x[:active] - x[1:active + 1])  # always positive (because x is sorted)
        denominator = (grad[:active] - grad[1:active + 1])
        # indices corresponding to negative denominator won't intersect
        # also, we are not interested in indices in x that are *already equal*
        valid = (denominator > eps) & (numerator > eps)
        valid_indices = valid.nonzero()
        intersection_times = numerator[valid] / denominator[valid]
        if intersection_times.size():
            delta, ind = intersection_times.min(0)
            ind = valid_indices[ind]
            delta, ind = delta[0], ind[0, 0]
    if grad[active] > 0:
        intersect_zero = x[active] / grad[active]
        if intersect_zero > 0. and intersect_zero < delta:
            return intersect_zero, -1
    return delta, ind


def project(gam, active, members):
    tovisit = set(range(active + 1))
    while tovisit:
        v = tovisit.pop()
        if len(members[v]) > 1:
            avg = 0.
            for k in members[v]:
                if k != v: tovisit.remove(k)
                avg += gam[k] / len(members[v])
            for k in members[v]:
                gam[k] = avg
    if active + 1 < len(gam):
        gam[active + 1:] = 0.


def find_proximal(x0, gam, lam, eps=1e-6, max_steps=20, debug=None):
    if debug is None:
        debug = {}
    # x0: sorted margins data
    # gam: initial gamma_fast(target, perm)
    # regularisation parameter lam
    x = x0.clone()
    act = (x >= eps).nonzero()
    finished = False
    if not act.size():
        finished = True
    else:
        active = act[-1, 0]
        members = {i: {i} for i in range(active + 1)}
        if active > 0:
            equal = (x[:active] - x[1:active + 1]) < eps
            for i, e in enumerate(equal):
                if e:
                    members[i].update(members[i + 1])
                    members[i + 1] = members[i]
            project(gam, active, members)
    step = 0
    while not finished and step < max_steps and active > -1:
        step += 1
        res = compute_step_length(x, gam, active, eps)
        delta, ind = res

        if ind == -1:
            active = active - len(members[active])

        stop = torch.dot(x - x0, gam) / torch.dot(gam, gam) + 1. / lam
        if 0 <= stop < delta:
            delta = stop
            finished = True

        x = x - delta * gam
        if not finished:
            if ind >= 0:
                repr = min(members[ind])
                members[repr].update(members[ind + 1])
                for m in members[ind]:
                    if m != repr:
                        members[m] = members[repr]
            project(gam, active, members)
        if "path" in debug:
            debug["path"].append(x.numpy())

    if "step" in debug:
        debug["step"] = step
    if "finished" in debug:
        debug["finished"] = finished
    return x, gam


def lovasz_binary(margins, label, prox=False, max_steps=20, debug=None):
    if debug is None:
        debug = {}
    # 1d vector inputs
    # Workaround: can't sort Variable bug
    # prox: False or lambda regularization value
    _, perm = torch.sort(margins.detach(), dim=0, descending=True)
    margins_sorted = margins[perm]
    grad = gamma_fast(label, perm)
    loss = torch.dot(F.relu(margins_sorted), grad)
    if prox is not False:
        xp, gam = find_proximal(margins_sorted.detach(), grad, prox, max_steps=max_steps, eps=1e-6, debug=debug)
        hook = margins_sorted.register_hook(lambda grad: (margins_sorted.detach() - xp))
        return loss, hook, gam
    else:
        return loss


def lovasz_single(logit, label, prox=False, max_steps=20, debug=None):
    if debug is None:
        debug = {}
    # single images
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logit.sum() * 0.
    target = label.contiguous().view(-1)[mask]
    signs = 2. * target.float() - 1.
    logit = logit.contiguous().view(-1)[mask]
    margins = (1. - logit * signs)
    loss = lovasz_binary(margins, target, prox, max_steps, debug=debug)
    return loss


def dice_coefficient(logit, label, isCuda=True):
    '''
    WARNING THIS IS VERY SLOW FOR SOME REASON!!

    :param logit:   calculated guess   (expects torch.Tensor)
    :param label:   truth label        (expects torch.Tensor)
    :return:        dice coefficient
    '''
    A = label.view(-1)
    B = logit.view(-1)

    A = A.clone()
    B = B.clone()

    if len(A) != len(B):
        raise AssertionError

    for i in list(range(len(A))):
        if A[i] > 0.5:
            A[i] = 1.0
        else:
            A[i] = 0.0

        if B[i] > 0.5:
            B[i] = 1.0
        else:
            B[i] = 0.0

    if isCuda:
        A = A.type(torch.cuda.ByteTensor)
    else:
        A = A.type(torch.ByteTensor)

    dice = torch.masked_select(B, A).sum()*2.0 / (B.sum() + A.sum())
    return dice


# ==================================== #
# Source: https://github.com/EKami/carvana-challenge
class WeightedSoftDiceLoss(torch.nn.Module):
    def __init__(self, **_):
        super(WeightedSoftDiceLoss, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        probs = torch.sigmoid(logits)
        num   = labels.size(0)
        w     = weights.view(num,-1)
        w2    = w*w
        m1    = probs.view(num,-1)
        m2    = labels.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def dice_coeff_hard_np(y_true, y_pred):
    smooth = 1.
    y_true_f = np.flatten(y_true)
    y_pred_f = np.round(np.flatten(y_pred))
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    return score

# ==================================== #
# Source: https://github.com/doodledood/carvana-image-masking-challenge/blob/master/losses.py
# TODO Replace this with nn.BCEWithLogitsLoss??
class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, **_):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, labels, **_):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = labels.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **_):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels, **_):
        num = labels.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.

    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """
    def __init__(self, l=0.5, eps=1e-6, **_):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, labels, **_):
        labels = labels.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(labels * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - labels) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss


class ThresholdedL1Loss(nn.Module):
    def __init__(self, threshold=0.5, **_):
        super(ThresholdedL1Loss, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels, **_):
        labels = labels.view(-1)
        probs = torch.sigmoid(logits).view(-1)
        probs = (probs > self.threshold).float()

        losses = torch.abs(labels - probs)
        loss = torch.mean(losses)

        return loss


class BCEDiceTL1Loss(nn.Module):
    def __init__(self, threshold=0.5, **_):
        super(BCEDiceTL1Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.tl1 = ThresholdedL1Loss(threshold=threshold)

    def forward(self, logits, labels, **_):
        return self.bce(logits, labels) + self.dice(logits, labels) + self.tl1(logits, labels)


class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''
    def __init__(self, focal_param, weights=None, **kwargs):
        if weights is None:
            weights = [1.0,1.0,1.0]
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = BCEWithLogitsViewLoss(weight=None, size_average=True, **kwargs)
        self.dice = SoftDiceLoss(**kwargs)
        self.focal = FocalLoss(l=focal_param, **kwargs)
        self.weights = weights

    def forward(self, logits, labels, **_):
        return self.weights[0] * self.bce(logits, labels) + self.weights[1] * self.dice(logits, labels) + self.weights[2] * self.focal(logits.unsqueeze(1), labels.unsqueeze(1))


class BCEDiceLoss(nn.Module):
    def __init__(self, **_):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, labels, **_):
        return self.bce(logits, labels) + self.dice(logits, labels)


class WeightedBCELoss2d(nn.Module):
    def __init__(self, **_):
        super(WeightedBCELoss2d, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        w = weights.view(-1)            # (-1 operation flattens all the dimensions)
        z = logits.view(-1)             # (-1 operation flattens all the dimensions)
        t = labels.view(-1)             # (-1 operation flattens all the dimensions)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss


class BCEDicePenalizeBorderLoss(nn.Module):
    def __init__(self, kernel_size=55, **_):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels, **_):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.ones(a.size()).to(device=logits.device)

        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        loss = self.bce(logits, labels, weights) + self.dice(logits, labels, weights)

        return loss


# ==== Focal Loss with extra parameters ==== #
# Source: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
# License: MIT
class FocalLoss2(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)

    Params:
        :param num_class:
        :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                        focus on hard misclassified example
        :param smooth: (float,double) smooth value when cross entropy
        :param balance_index: (int) balance class index, should be specific when alpha is float
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True, **_):
        super(FocalLoss2, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            if len(self.alpha) != self.num_class:
                raise AssertionError
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logits, labels, **_):

        # logits = F.softmax(logits, dim=1)

        if logits.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.permute(0, 2, 1).contiguous()
            logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha.to(logits.device)

        idx = labels.cpu().long()

        one_hot_key = torch.FloatTensor(labels.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(logits.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logits).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# -------- #
# Source: https://github.com/huaifeng1993/DFANet/blob/master/loss.py
class FocalLoss3(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.

        Params:
            :param alpha: (1D Tensor, Variable) - the scalar factor for this criterion
            :param gamma: (float, double) - gamma > 0
            :param size_average: (bool) - size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, **_):
        super(FocalLoss3, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num+1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, labels, **_):  # variables
        P = F.softmax(inputs)

        if len(inputs.size()) == 3:
            torch_out = torch.zeros(inputs.size())
        else:
            b,c,h,w = inputs.size()
            torch_out = torch.zeros([b,c+1,h,w])

        if inputs.is_cuda:
            torch_out = torch_out.cuda()

        class_mask = Variable(torch_out)
        class_mask.scatter_(1, labels.long(), 1.)
        class_mask = class_mask[:,:-1,:,:]

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # print('alpha',self.alpha.size())
        alpha = self.alpha[labels.data.view(-1)].view_as(labels)
        # print (alpha.size(),class_mask.size(),P.size())
        probs = (P * class_mask).sum(1)  # + 1e-6#.view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
# -------- #


# -------- #
# Source: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
class BinaryFocalLoss(nn.Module):
    '''
        Implementation of binary focal loss. For multi-class focal loss use one of the other implementations.

        gamma = 0 is equivalent to BinaryCrossEntropy Loss
    '''
    def __init__(self, gamma=1.333, eps=1e-6, alpha=1.0, **_):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha

    def forward(self, inputs, labels, **_):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, labels, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
# -------- #


# ==== Additional Losses === #
# Source: https://github.com/atlab/attorch/blob/master/attorch/losses.py
# License: MIT
class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12, **_):
        super().__init__()
        self.bias = bias

    def forward(self, output, labels, **_):
        # _assert_no_grad(target)
        with torch.no_grad:         # Pytorch 0.4.0 replacement (should be ok to use like this)
            return (output - labels * torch.log(output + self.bias)).mean()


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-12, **_):
        super().__init__()
        self.bias = bias

    def forward(self, output, target, **_):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :] * torch.log(output + self.bias)).mean()


class L1Loss3d(nn.Module):
    def __init__(self, bias=1e-12, **_):
        super().__init__()
        self.bias = bias

    @staticmethod
    def forward(output, target, **_):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).abs().mean()


class MSE3D(nn.Module):
    def __init__(self, **_):
        super().__init__()

    @staticmethod
    def forward(output, target, **_):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).pow(2).mean()


# ==== Custom ==== #
class BCEWithLogitsViewLoss(nn.BCEWithLogitsLoss):
    '''
    Silly wrapper of nn.BCEWithLogitsLoss because BCEWithLogitsLoss only takes a 1-D array
    '''
    def __init__(self, weight=None, size_average=True, **_):
        super().__init__(weight=weight, size_average=size_average)

    def forward(self, input_, target, **_):
        '''
        :param input_:
        :param target:
        :return:

        Simply passes along input.view(-1), target.view(-1)
        '''
        return super().forward(input_.view(-1), target.view(-1))


# ===================== #
# Source: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
# For calculating dice loss on images where multiple classes are present at the same time
def multi_class_dice_loss(output, target, weights=None, ignore_index=None):
    # output : NxCxHxW float tensor
    # target :  NxHxW long tensor
    # weights : C float tensor
    # ignore_index : int value to ignore from loss
    smooth = 1.
    loss = 0.

    output = output.exp()
    encoded_target = output.detach().clone().zero_()
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

    if weights is None:
        weights = torch.ones(output.size(1)).type_as(output.detach())

    intersection = output * encoded_target
    numerator = 2 * intersection.sum(3).sum(2).sum(0) + smooth
    denominator = (output + encoded_target).sum(3).sum(2).sum(0) + smooth
    loss_per_channel = weights * (1 - (numerator / denominator))

    return loss_per_channel.sum() / output.size(1)

# ====================== #
# Source: https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
# Calculation of soft-IOU loss
def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


# ====================== #
# Source: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
# Another calculation of dice loss over multiple classes. Input is numpy matrices.
def soft_multiclass_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean(numerator / (denominator + epsilon))  # average over classes and batch


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, num_classes=2, **_):
        super(mIoULoss, self).__init__()
        self.classes = num_classes

    def forward(self, inputs, target_oneHot, **_):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return -loss.mean()


# ====================== #
# Source: https://github.com/snakers4/mnasnet-pytorch/blob/master/src/models/semseg_loss.py
# Combination Loss from BCE and Dice
class ComboBCEDiceLoss(nn.Module):
    """
        Combination BinaryCrossEntropy (BCE) and Dice Loss with an optional running mean and loss weighing.
    """

    def __init__(self, use_running_mean=False, bce_weight=1, dice_weight=1, eps=1e-6, gamma=0.9, combined_loss_only=True, **_):
        """

        :param use_running_mean: - bool (default: False) Whether to accumulate a running mean and add it to the loss with (1-gamma)
        :param bce_weight: - float (default: 1.0) Weight multiplier for the BCE loss (relative to dice)
        :param dice_weight: - float (default: 1.0) Weight multiplier for the Dice loss (relative to BCE)
        :param eps: -
        :param gamma:
        :param combined_loss_only: - bool (default: True) whether to return a single combined loss or three separate losses
        """

        super().__init__()
        '''
        Note: BCEWithLogitsLoss already performs a torch.sigmoid(pred)
        before applying BCE!
        '''
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean is True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.bce_logits_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, labels, **_):
        # inputs and targets are assumed to be BxCxWxH (batch, color, width, height)
        outputs = outputs.squeeze()       # necessary in case we're dealing with binary segmentation (color dim of 1)
        if len(outputs.shape) != len(labels.shape):
            raise AssertionError
        # assert that B, W and H are the same
        if outputs.size(-0) != labels.size(-0):
            raise AssertionError
        if outputs.size(-1) != labels.size(-1):
            raise AssertionError
        if outputs.size(-2) != labels.size(-2):
            raise AssertionError

        bce_loss = self.bce_logits_loss(outputs, labels)

        dice_target = (labels == 1).float()
        dice_output = torch.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean is False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)

        loss = bce_loss * bmw + dice_loss * dmw

        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss


class ComboSemsegLossWeighted(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=1,
                 dice_weight=1,
                 eps=1e-6,
                 gamma=0.9,
                 use_weight_mask=False,
                 combined_loss_only=False, **_
                 ):
        super().__init__()

        self.use_weight_mask = use_weight_mask

        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma
        self.combined_loss_only = combined_loss_only

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean is True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.nll_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, logits, labels, weights, **_):
        # logits and labels are assumed to be BxCxWxH
        if len(logits.shape) != len(labels.shape):
            raise AssertionError
        # assert that B, W and H are the same
        if logits.size(0) != labels.size(0):
            raise AssertionError
        if logits.size(2) != labels.size(2):
            raise AssertionError
        if logits.size(3) != labels.size(3):
            raise AssertionError

        # weights are assumed to be BxWxH
        # assert that B, W and H are the are the same for target and mask
        if logits.size(0) != weights.size(0):
            raise AssertionError
        if logits.size(2) != weights.size(1):
            raise AssertionError
        if logits.size(3) != weights.size(2):
            raise AssertionError

        if self.use_weight_mask:
            bce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                          target=labels,
                                                          weight=weights)
        else:
            bce_loss = self.nll_loss(input=logits,
                                     target=labels)

        dice_target = (labels == 1).float()
        dice_output = torch.sigmoid(logits)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean is False:
            bmw = self.bce_weight
            dmw = self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
        else:
            self.running_bce_loss = self.running_bce_loss * self.gamma + bce_loss.data * (1 - self.gamma)
            self.running_dice_loss = self.running_dice_loss * self.gamma + dice_loss.data * (1 - self.gamma)

            bm = float(self.running_bce_loss)
            dm = float(self.running_dice_loss)

            bmw = 1 - bm / (bm + dm)
            dmw = 1 - dm / (bm + dm)

        loss = bce_loss * bmw + dice_loss * dmw

        if self.combined_loss_only:
            return loss
        else:
            return loss, bce_loss, dice_loss


# ====================== #
# Source: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/loss.py
# Description: http://www.erogol.com/online-hard-example-mining-pytorch/
# Online Hard Example Loss
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, thresh=0.6, min_kept=0, ignore_index=-100, is_binary=True, **kwargs):
        super().__init__()
        self.ignore_label = ignore_index
        self.is_binary = is_binary
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = BCEWithLogitsViewLoss(**kwargs)

    def forward(self, logits, labels, **_):
        """
            Args:
                predict:(n, c, h, w)
                labels:(n, h, w)
        """

        if self.is_binary:
            predict = torch.sigmoid(logits)
        else:
            predict = F.softmax(logits, dim=1)

        n, c, h, w = predict.size()
        input_label = labels.detach().cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.detach().cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label-1, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            #print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        #print(np.sum(input_label != self.ignore_label))
        labels = torch.from_numpy(input_label.reshape(labels.size())).type_as(predict).to(labels.device)

        predict = predict.squeeze()          # in case we're dealing with B/W images instead of RGB
        return self.criterion(predict, labels)


# ====================== #
# Source: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/utils/loss.py
# Loss used for EncNet
class EncNetLoss(nn.CrossEntropyLoss):
    """
    2D Cross Entropy Loss with SE Loss

    Specifically used for EncNet.
    se_loss is the Semantic Encoding Loss from the paper `Context Encoding for Semantic Segmentation <https://arxiv.org/pdf/1803.08904v1>`_.
    It computes probabilities of contexts appearing together.

    Without SE_loss and Aux_loss this class simply forwards inputs to Torch's Cross Entropy Loss (nn.CrossEntropyLoss)
    """

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **_):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs, **_):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    """
    Loss taking into consideration class and segmentation targets together, as well as, using OHEM
    """
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def to(self, device):
        super().to(device=device)
        self.bceloss.to(device=device)

    def _aux_forward(self, *inputs, **_):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **_):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds, target))


# ====================== #
# Source: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
# OHEM Segmentation Loss
class OHEMSegmentationLosses(OhemCrossEntropy2d):
    """
    2D Cross Entropy Loss with Auxiliary Loss

    """
    def __init__(self, se_loss=False, se_weight=0.2, num_classes=1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(OHEMSegmentationLosses, self).__init__(ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.num_classes = num_classes
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def to(self, device):
        super().to(device=device)
        self.bceloss.to(device=device)

    def forward(self, *inputs, **_):
        if not self.se_loss and not self.aux:
            return super(OHEMSegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.num_classes).type_as(pred)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.num_classes).type_as(pred1)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


# ====================== #
# Source: https://github.com/yinmh17/DNL-Semantic-Segmentation/blob/master/model/seg/loss/ohem_ce_loss.py
# OHEM CrossEntropy Loss

class OhemCELoss(nn.Module):
    def __init__(self, configer, is_binary=False):
        super(OhemCELoss, self).__init__()
        self.configer = configer
        weight = self.configer.get('loss.params.ohem_ce_loss.weight', default=None)
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = self.configer.get('loss.params.ohem_ce_loss.reduction', default='mean')
        self.ignore_index = self.configer.get('loss.params.ohem_ce_loss.ignore_index', default=-100)
        self.thresh = self.configer.get('loss.params.ohem_ce_loss.thresh', default=0.7)
        self.min_kept = max(1, self.configer.get('loss.params.ohem_ce_loss.minkeep', default=5))
        self.is_binary = is_binary

    def forward(self, logits, labels, **_):
        """
            Args:
                logits:(n, c, h, w)
                labels:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        batch_kept = self.min_kept * labels.size(0)
        labels = self._scale_target(labels, (logits.size(2), logits.size(3)))
        if self.is_binary:
            prob_out = torch.sigmoid(logits)
        else:
            prob_out = F.softmax(logits, dim=1)
        tmp_target = labels.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = labels.contiguous().view(-1, ) != self.ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()
        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        loss_matrix = F.cross_entropy(logits, labels,
                                      weight=self.weight.to(logits.device) if self.weight is not None else None,
                                      ignore_index=self.ignore_index, reduction='none')
        loss_matrix = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


# ===================== #
# Source: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean', **_):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    @staticmethod
    def prob_flatten(input, target):
        if input.dim() not in [4, 5]:
            raise AssertionError
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets, **_):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


# ===================== #
# Source: https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py (MIT)
class ActiveContourLoss(nn.Module):
    """
        `Learning Active Contour Models for Medical Image Segmentation <http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf>`_
        Note that is only works for B/W masks right now... which is kind of the point of this loss as contours in RGB should be cast to B/W
        before computing the loss.

        Params:
            :param mu:          (float, default=1.0) - Scales the inner region loss relative to outer region (less or more prominent)
            :param lambdaP:     (float, default=1.0) - Scales the combined region loss compared to the length loss (less or more prominent)
    """

    def __init__(self, lambdaP=5., mu=1., is_binary: bool = False, **_):
        super(ActiveContourLoss, self).__init__()
        self.lambdaP = lambdaP
        self.mu = mu
        self.is_binary = is_binary

    def forward(self, logits, labels, **_):
        if self.is_binary:
            logits = torch.sigmoid(logits)
        else:
            logits = F.softmax(logits, dim=1)

        if labels.shape != logits.shape:
            if logits.shape > labels.shape:
                labels.unsqueeze(dim=1)
            else:
                raise Exception(f'Non-matching shapes for logits ({logits.shape}) and labels ({labels.shape})')

        """
        length term
        """
        x = logits[:,:,1:,:] - logits[:,:,:-1,:]    # horizontal gradient (B, C, H-1, W)
        y = logits[:,:,:,1:] - logits[:,:,:,:-1]    # vertical gradient   (B, C, H,   W-1)

        delta_x = x[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_y = y[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_u = torch.abs(delta_x + delta_y)

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_u + epsilon))   # eq.(11) in the paper, mean is used instead of sum.

        """
        region term
        """

        C_in = torch.ones_like(logits)
        C_out = torch.zeros_like(labels)

        region_in = torch.abs(torch.mean(logits[:,0,:,:] * ((labels[:, 0, :, :] - C_in) ** 2)))         # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.abs(torch.mean((1-logits[:,0,:,:]) * ((labels[:, 0, :, :] - C_out) ** 2)))   # equ.(12) in the paper

        return length + self.lambdaP * (self.mu * region_in + region_out)


class ActiveContourLossAlt(nn.Module):
    """
        `Learning Active Contour Models for Medical Image Segmentation <http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf>`_
        Note that is only works for B/W masks right now... which is kind of the point of this loss as contours in RGB should be cast to B/W
        before computing the loss.

        Params:
            :param len_w: (float, default=1.0) - The multiplier to use when adding boundary loss.
            :param reg_w: (float, default=1.0) - The multiplier to use when adding region loss.
            :param apply_log: (bool, default=True) - Whether to transform the log into log space (due to the
    """

    def __init__(self, len_w=1., reg_w=1., apply_log=True, is_binary: bool = False, **_):
        super(ActiveContourLossAlt, self).__init__()
        self.len_w = len_w
        self.reg_w = reg_w
        self.epsilon = 1e-8  # a parameter to avoid square root = zero issues
        self.apply_log = apply_log
        self.is_binary = is_binary

    def forward(self, logits, labels, **_):
        # must convert raw logits to predicted probabilities for each pixel along channel
        if self.is_binary:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        if labels.shape != logits.shape:
            if logits.shape > labels.shape:
                labels.unsqueeze(dim=1)
            else:
                raise Exception(f'Non-matching shapes for logits ({logits.shape}) and labels ({labels.shape})')

        """
        length term:
            - Subtract adjacent pixels from each other in X and Y directions
            - Determine where they differ from the ground truth (targets)
            - Calculate MSE
        """
        # horizontal and vertical directions
        x = probs[:, :, 1:, :] - probs[:, :, :-1, :]      # differences in horizontal direction
        y = probs[:, :, :, 1:] - probs[:, :, :, :-1]      # differences in vertical direction

        target_x = labels[:, :, 1:, :] - labels[:, :, :-1, :]
        target_y = labels[:, :, :, 1:] - labels[:, :, :, :-1]

        # find difference between values of probs and targets
        delta_x = (target_x - x).abs()          # do we need to subtract absolute values or relative?
        delta_y = (target_y - y).abs()

        # get MSE of the differences per pixel
        # importantly because deltas are mostly < 1, a simple square of the error will actually yield LOWER results
        # so we select 0.5 as the middle ground where small error will be further minimized while large error will
        # be highlighted (pushed to be > 1 and up to 2.5 for maximum error).
        # len_error_sq = ((delta_x + 0.5) ** 2) + ((delta_y + 0.5) ** 2)
        # length = torch.sqrt(len_error_sq.sum() + self.epsilon)

        # the length loss here is simply the MSE of x and y deltas
        length_loss = torch.sqrt(delta_x.sum() ** 2 + delta_y.sum() ** 2 + self.epsilon)


        """
        region term (should this be done in log space to avoid instabilities?)
            - compute the error produced by all pixels that are not equal to 0 outside of the ground truth mask
            - compute error produced by all pixels that are not equal to 1 inside the mask
        """
        # reference code for selecting masked values from a tensor
        # t_m_bool = t_mask.type(torch.ByteTensor)
        # t_result = t_in.masked_select(t_m_bool)

        # C_1 = torch.ones((image_size, image_size), device=target.device)
        # C_2 = torch.zeros((image_size, image_size), device=target.device)

        # the sum of all pixel values that are not equal 0 outside of the ground truth mask
        error_in = probs[:, 0, :, :] * ((labels[:, 0, :, :] - 1) ** 2)  # invert the ground truth mask and multiply by probs

        # the sum of all pixel values that are not equal 1 inside of the ground truth mask
        probs_diff = (probs[:, 0, :, :] - labels[:, 0, :, :]).abs()     # subtract mask from probs giving us the errors
        error_out = (probs_diff * labels[:, 0, :, :])                   # multiply mask by error, giving us the error terms inside the mask.

        if self.apply_log:
            loss = torch.log(length_loss) + torch.log(error_in.sum() + error_out.sum())
        else:
            # loss = self.len_w * length_loss
            loss = self.reg_w * (error_in.sum() + error_out.sum())

        return torch.clamp(loss, min=0.0)        # make sure we don't return negative values


# ===================== #
# Sources:  https://github.com/JunMa11/SegLoss
#           https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet (Apache 2.0)
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial.distance import directed_hausdorff
    if len(pred.shape) != 2:
        raise AssertionError
    if pred.shape != target.shape:
        raise AssertionError

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    if preds.shape != target.shape:
        raise AssertionError
    if not one_hot(preds):
        raise AssertionError
    if not one_hot(target):
        raise AssertionError

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.detach().cpu().numpy()
    n_target = target.detach().cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

# ===================== #
# Boundary Loss
# Source:  https://github.com/JunMa11/SegLoss/blob/71b14900e91ea9405d9705c95b451fc819f24c70/test/loss_functions/boundary_loss.py#L102

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    img_gt: segmentation, shape = (batch_size, x, y, z)
    out_shape: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    from scipy.ndimage import distance_transform_edt
    from skimage import segmentation as skimage_seg

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # channel
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


class BDLoss(nn.Module):
    def __init__(self, is_binary: bool = False, **_):
        """
        compute boundary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        self.is_binary = is_binary
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, logits, labels, **_):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        if self.is_binary:
            logits = torch.sigmoid(logits)
        else:
            logits = F.softmax(logits, dim=1)
        with torch.no_grad():
            if len(logits.shape) != len(labels.shape):
                labels = labels.view((labels.shape[0], 1, *labels.shape[1:]))

            if all([i == j for i, j in zip(logits.shape, labels.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = labels
            else:
                labels = labels.long()
                y_onehot = torch.zeros(logits.shape)
                if logits.device.type == "cuda":
                    y_onehot = y_onehot.cuda(logits.device.index)
                y_onehot.scatter_(1, labels, 1)
            gt_sdf = compute_sdf(y_onehot.cpu().numpy(), logits.shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != logits.device:
            phi = phi.to(logits.device).type(torch.float32)
        # pred = net_output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", logits[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss


# ===================== #
# Source:  https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
class TverskyLoss(nn.Module):
    """Computes the Tversky loss [1].
        Args:
            :param alpha: controls the penalty for false positives.
            :param beta: controls the penalty for false negatives.
            :param eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha, beta, eps=1e-7, **_):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, labels, **_):
        """
        Args:
            :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            :param labels: a tensor of shape [B, H, W] or [B, 1, H, W].
            :return: loss
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[labels.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[labels.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, logits.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()

        return 1 - tversky_loss


# ===================== #
# Source:  https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None, **_):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        - Example -
        criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type not in ['arcface', 'sphereface', 'cosface']:
            raise AssertionError
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels, **_):
        '''
        input shape (N, in_features)
        '''
        if len(x) != len(labels):
            raise AssertionError
        if torch.min(labels) < 0:
            raise AssertionError
        if torch.max(labels) >= self.out_features:
            raise AssertionError

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


# ===================== #
# Source:  https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, **_):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, logits, labels, loss_mask=None, **_):
        shp_x = logits.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        tp, fp, fn = get_tp_fp_fn(logits, labels, axes, loss_mask, self.square)# shape: (batch size, class num)
        weight = (self.beta**2)/(1+self.beta**2)
        asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()

        return -asym


# ===================== #
# Source:  https://github.com/BloodAxe/pytorch-toolbelt
# Used to enhance facial segmentation
def wing_loss(output: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction="mean"):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param output:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - output).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


class WingLoss(nn.modules.loss._Loss):
    """
        Used to enhance facial segmentation
    """
    def __init__(self, width=5, curvature=0.5, reduction="mean", **_):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target, **_):
        return wing_loss(prediction, target, self.width, self.curvature, self.reduction)


# ===================== #
# Source: https://github.com/JUNHAOYAN/FPN/tree/master/RMI
# ..which is adapted from: https://github.com/ZJULearning/RMI (MIT License)
# Segmentation loss (memory intensive)
class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self,
                 num_classes=1,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1,
                 device="cuda", **_):
        super(RMILoss, self).__init__()

        self._CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
        self._CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
        self._POS_ALPHA = 5e-4		    # add this factor to ensure the AA^T is positive definite
        self._IS_SUM = 1			        # sum the loss per channel

        self.num_classes = num_classes
        # radius choices
        if rmi_radius not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            raise AssertionError
        self.rmi_radius = rmi_radius
        if rmi_pool_way not in [0, 1, 2, 3]:
            raise AssertionError
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        if rmi_pool_size != rmi_pool_stride:
            raise AssertionError
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255
        self.device = device

    def forward(self, logits, labels, **_):
        if self.num_classes == 1:
            loss = self.forward_sigmoid(logits, labels)
        else:
            loss = self.forward_softmax_sigmoid(logits, labels)
        return loss

    def forward_softmax_sigmoid(self, inputs, targets):
        """
        Using both softmax and sigmoid operations.
        Args:
            inputs 	:	[N, C, H, W], dtype=float32
            targets 	:	[N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=inputs,
                                      target=targets.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = targets < self.num_classes
        valid_onehot_labels_4D = F.one_hot(targets.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = torch.sigmoid(inputs) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=self._CLIP_MIN, max=self._CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + self._CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D 	:	[N, C, H, W], dtype=float32
            probs_4D 	:	[N, C, H, W], dtype=float32
        """
        if labels_4D.size() != probs_4D.size():
            raise AssertionError

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = self.map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.double).to(self.device).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.double).to(self.device)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * self._POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * self.log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * self._POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if self._IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss

    @staticmethod
    def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
        """get map pairs
        Args:
            labels_4D	:	labels, shape [N, C, H, W]
            probs_4D	:	probabilities, shape [N, C, H, W]
            radius		:	the square radius
        Return:
            tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
        """
        # pad to ensure the following slice operation is valid
        # pad_beg = int(radius // 2)
        # pad_end = radius - pad_beg

        # the original height and width
        label_shape = labels_4D.size()
        h, w = label_shape[2], label_shape[3]
        new_h, new_w = h - (radius - 1), w - (radius - 1)
        # https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
        # padding = (pad_beg, pad_end, pad_beg, pad_end)
        # labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

        # get the neighbors
        la_ns = []
        pr_ns = []
        # for x in range(0, radius, 1):
        for y in range(0, radius, 1):
            for x in range(0, radius, 1):
                la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
                pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            # for calculating RMI
            pair_ns = la_ns + pr_ns
            p_vectors = torch.stack(pair_ns, dim=2)
            return p_vectors
        else:
            # for other purpose
            la_vectors = torch.stack(la_ns, dim=2)
            pr_vectors = torch.stack(pr_ns, dim=2)
            return la_vectors, pr_vectors

    @staticmethod
    def log_det_by_cholesky(matrix):
        """
        Args:
            matrix: matrix must be a positive define matrix.
                    shape [N, C, D, D].
        Ref:
            https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
        """
        # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
        # where C is the cholesky decomposition of A.
        chol = torch.cholesky(matrix)
        # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
        return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)


# ===================== #
# Source:  https://github.com/RElbers/region-mutual-information-pytorch
# Segmentation loss (memory intensive)
class RMILossAlt(nn.Module):
    """
    PyTorch Module which calculates the Region Mutual Information loss (https://arxiv.org/abs/1910.12037).
    """

    def __init__(self,
                 with_logits,
                 radius=3,
                 bce_weight=0.5,
                 downsampling_method='max',
                 stride=3,
                 use_log_trace=True,
                 use_double_precision=True,
                 epsilon=0.0005, **_):
        """
        :param with_logits:
            If True, apply the sigmoid function to the prediction before calculating loss.
        :param radius:
            RMI radius.
        :param bce_weight:
            Weight of the binary cross entropy. Must be between 0 and 1.
        :param downsampling_method:
            Downsampling method used before calculating RMI. Must be one of ['avg', 'max', 'region-extraction'].
            If 'region-extraction', then downscaling is done during the region extraction phase. Meaning that the stride is the spacing between consecutive regions.
        :param stride:
            Stride used for downsampling.
        :param use_log_trace:
            Whether to calculate the log of the trace, instead of the log of the determinant. See equation (15).
        :param use_double_precision:
            Calculate the RMI using doubles in order to fix potential numerical issues.
        :param epsilon:
            Magnitude of the entries added to the diagonal of M in order to fix potential numerical issues.
        """
        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, logits, labels, **_):
        labels = labels.unsqueeze(1)
        # Calculate BCE if needed
        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(logits, target=labels)
            else:
                bce = F.binary_cross_entropy(logits, target=labels)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        # Apply sigmoid to get probabilities. See final paragraph of section 4.
        if self.with_logits:
            logits = torch.sigmoid(logits)

        # Calculate RMI loss
        rmi = self.rmi_loss(input_=logits, target=labels)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

    def rmi_loss(self, input_, target):
        """
        Calculates the RMI loss between the prediction and target.
        :return:
            RMI loss
        """

        if input_.shape != target.shape:
            raise AssertionError
        vector_size = self.radius * self.radius

        # Get region vectors
        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input_)

        # Convert to doubles for better precision
        if self.use_double_precision:
            y = y.double()
            p = p.double()

        # Small diagonal matrix to fix numerical issues
        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        # Subtract mean
        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        # Covariances
        y_cov = y @ transpose(y)
        p_cov = p @ transpose(p)
        y_p_cov = y @ transpose(p)

        # Approximated posterior covariance matrix of Y given P
        m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)

        # Lower bound of RMI
        if self.use_log_trace:
            rmi = 0.5 * log_trace(m + eps)
        else:
            rmi = 0.5 * log_det(m + eps)

        # Normalize
        rmi = rmi / float(vector_size)

        # Sum over classes, mean over samples.
        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        """
        Downsamples and extracts square regions from x.
        Returns the flattened vectors of length radius*radius.
        """

        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == 'region-extraction' else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius ** 2, -1))
        return x_regions

    def downsample(self, x):
        # Skip if stride is 1
        if self.stride == 1:
            return x

        # Skip if we pool during region extraction.
        if self.downsampling_method == 'region-extraction':
            return x

        padding = self.stride // 2
        if self.downsampling_method == 'max':
            return F.max_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        if self.downsampling_method == 'avg':
            return F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride, padding=padding)
        raise ValueError(self.downsampling_method)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)


# ====================== #
# Source: https://github.com/NRCan/geo-deep-learning/blob/develop/losses/boundary_loss.py
class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    # in previous implementations theta0=3, theta=5
    def __init__(self, theta0=19, theta=19, ignore_index=None, weight=None, is_binary: bool = False, **_):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
        self.weight = weight
        self.is_binary = is_binary

    def forward(self, logits, labels, **_):
        """
        Input:
            - logits: the output from model (before softmax)
                    shape (N, C, H, W)
            - labels: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, _, _ = logits.shape

        # sigmoid / softmax so that predicted map can be distributed in [0, 1]
        if self.is_binary:
            logits = torch.sigmoid(logits)
        else:
            logits = torch.softmax(logits, dim=1)

        # one-hot vector of ground truth
        # print(gt.shape)
        # zo = F.one_hot(gt, c)
        # print(zo.shape)
        if self.is_binary:
            one_hot_gt = labels
        else:
            one_hot_gt = F.one_hot(labels.long()).permute(0, 3, 1, 2).squeeze(dim=-1).contiguous().float()

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - logits, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - logits

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        eps = 1e-7
        P = (torch.sum(pred_b * gt_b_ext, dim=2) + eps) / (torch.sum(pred_b, dim=2) + eps)
        R = (torch.sum(pred_b_ext * gt_b, dim=2) + eps) / (torch.sum(gt_b, dim=2) + eps)

        # Boundary F1 Score

        BF1 = (2 * P * R + eps) / (P + R + eps)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


# ====================== #
"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
import cv2


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **_):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    @staticmethod
    def distance_field(img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, debug=False, **_) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        labels = labels.unsqueeze(1)

        if logits.dim() not in (4, 5):
            raise AssertionError("Only 2D and 3D supported")
        if (logits.dim() != labels.dim()):
            raise AssertionError("Prediction and target need to be of same dimension")

        # this is necessary for binary loss
        logits = torch.sigmoid(logits)

        pred_dt = torch.from_numpy(self.distance_field(logits.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(labels.detach().cpu().numpy())).float()

        pred_error = (logits - labels) ** 2
        distance = pred_dt.to(logits.device) ** self.alpha + target_dt.to(logits.device) ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.detach().cpu().numpy(),
                (
                    dt_field.detach().cpu().numpy()[0, 0],
                    pred_error.detach().cpu().numpy()[0, 0],
                    distance.detach().cpu().numpy()[0, 0],
                    pred_dt.detach().cpu().numpy()[0, 0],
                    target_dt.detach().cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(self, pred: np.ndarray, target: np.ndarray, debug) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        target = target.unsqueeze(1)
        if pred.dim() not in (4, 5):
            raise AssertionError("Only 2D and 3D supported")
        if (pred.dim() != target.dim()):
            raise AssertionError("Prediction and target need to be of same dimension")

        pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug)
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(self.perform_erosion(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug)).float()

            loss = eroted.mean()

            return loss


# ====================== #
"""
Recall Loss
copy pasted from - all credit goes to original authors:
https://github.com/shuaizzZ/Recall-Loss-PyTorch/blob/master/recall_loss.py
"""
class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None, **_):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, logits, labels, **_):
        N, C = logits.size()[:2]
        _, predict = torch.max(logits, 1)# # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1) # (N, 1, *)
        labels = labels.view(N, 1, -1) # (N, 1, *)
        last_size = labels.size(-1)

        ## convert predict & target (N, 1, *) into one hot vector (N, C, *)
        predict_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1) # (N, C, *)
        target_onehot = torch.zeros((N, C, last_size)).cuda() # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, labels, 1) # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != logits.type():
                self.weight = self.weight.type_as(logits)
                recall = recall * self.weight * C  # (N, C)
        recall_loss = 1 - torch.mean(recall)  # 1

        return recall_loss


# ====================== #
class SoftInvDiceLoss(torch.nn.Module):
    """
    Well-performing loss for binary segmentation
    """
    def __init__(self, smooth=1., is_binary=True, **_):
        super(SoftInvDiceLoss, self).__init__()
        self.smooth = smooth
        self.is_binary = is_binary

    def forward(self, logits, labels, **_):
        # sigmoid / softmax so that predicted map can be distributed in [0, 1]
        if self.is_binary:
            logits = torch.sigmoid(logits)
        else:
            logits = torch.softmax(logits, dim=1)
        iflat = 1 - logits.view(-1)
        tflat = 1 - labels.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))


# ======================= #
# --- COMBINED LOSSES --- #
class OhemBCEDicePenalizeBorderLoss(OhemCrossEntropy2d):
    """
    Combined OHEM (Online Hard Example Mining) process with BCE-Dice penalized loss
    """
    def __init__(self, thresh=0.6, min_kept=0, ignore_index=-100, kernel_size=21, **_):
        super().__init__()
        self.ignore_label = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = BCEDicePenalizeBorderLoss(kernel_size=kernel_size)


class RMIBCEDicePenalizeBorderLoss(RMILossAlt):
    """
    Combined RMI and BCEDicePenalized Loss
    """
    def __init__(self, kernel_size=21, rmi_weight=1.0, bce_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.bce = BCEDicePenalizeBorderLoss(kernel_size=kernel_size)
        self.bce_weight = bce_weight
        self.rmi_weight = rmi_weight

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)

    def forward(self, logits, labels, **_):
        if labels.shape != logits.shape:
            if logits.shape > labels.shape:
                labels.unsqueeze(dim=1)
            else:
                raise Exception(f'Non-matching shapes for logits ({logits.shape}) and labels ({labels.shape})')

        # Calculate RMI loss
        rmi = self.rmi_loss(input_=torch.sigmoid(logits), target=labels)
        bce = self.bce(logits, labels)
        # rmi = rmi.mean() * (1.0 - self.bce_weight)
        return self.rmi_weight * rmi + self.bce_weight * bce