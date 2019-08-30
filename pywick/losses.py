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
"""

##  Various loss calculation functions  ##
# Sources:  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py (?)
#           https://github.com/doodledood/carvana-image-masking-challenge/blob/master/losses.py (MIT)
#           https://github.com/atlab/attorch/blob/master/attorch/losses.py (MIT)
#           https://github.com/EKami/carvana-challenge (MIT)
#           https://github.com/DingKe/pytorch_workplace (MIT)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch import Tensor
from typing import Iterable, Set

VOID_LABEL = 255
N_CLASSES = 1


class StableBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
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
def lovaszloss(logits, labels, prox=False, max_steps=20, debug={}):
    """
    `The Lovasz-Softmax loss <https://arxiv.org/abs/1705.08790>`_

    :param logits:
    :param labels:
    :param prox:
    :param max_steps:
    :param debug:
    :return:
    """

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


def find_proximal(x0, gam, lam, eps=1e-6, max_steps=20, debug={}):
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


def lovasz_binary(margins, label, prox=False, max_steps=20, debug={}):
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


def lovasz_single(logit, label, prox=False, max_steps=20, debug={}):
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

    assert len(A) == len(B)

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
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
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
    def __init__(self, weight=None, size_average=True, **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **kwargs):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        #print('logits: {}, targets: {}'.format(logits.size(), targets.size()))
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
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
    def __init__(self, l=0.5, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(targets * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - targets) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss


class ThresholdedL1Loss(nn.Module):
    def __init__(self, threshold=0.5, **kwargs):
        super(ThresholdedL1Loss, self).__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)
        probs = (probs > 0.5).float()

        losses = torch.abs(targets - probs)
        loss = torch.mean(losses)

        return loss


class BCEDiceTL1Loss(nn.Module):
    def __init__(self, threshold=0.5):
        super(BCEDiceTL1Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.tl1 = ThresholdedL1Loss(threshold=threshold)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets) + self.tl1(logits, targets)


class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''
    def __init__(self, focal_param, weights=[1.0,1.0,1.0], **kwargs):
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(l=focal_param)
        self.weights = weights

    def forward(self, logits, targets):
        logits = logits.squeeze()
        return self.weights[0] * self.bce(logits, targets) + self.weights[1] * self.dice(logits, targets) + self.weights[2] * self.focal(logits.unsqueeze(1), targets.unsqueeze(1))


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class WeightedBCELoss2d(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view(-1)
        t = labels.view(-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = torch.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        smooth = 1.
        score = 2. * ((w2*intersection).sum(1)+smooth) / ((w2*m1).sum(1) + (w2*m2).sum(1)+smooth)
        score = 1 - score.sum()/num
        return score


class BCEDicePenalizeBorderLoss(nn.Module):
    def __init__(self, kernel_size=21, **kwargs):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels):
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

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
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

    def forward(self, logit, target):

        # logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
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

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
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

    def forward(self, inputs, targets):  # variables
        P = F.softmax(inputs)

        if len(inputs.size()) == 3:
            torch_out = torch.zeros(inputs.size())
        else:
            b,c,h,w = inputs.size()
            torch_out = torch.zeros([b,c+1,h,w])

        if inputs.is_cuda:
            torch_out = torch_out.cuda()

        class_mask = Variable(torch_out)
        class_mask.scatter_(1, targets.long(), 1.)
        class_mask = class_mask[:,:-1,:,:]

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # print('alpha',self.alpha.size())
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
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
    def __init__(self, gamma=1.333, eps=1e-6, alpha=1.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
# -------- #


# ==== Additional Losses === #
# Source: https://github.com/atlab/attorch/blob/master/attorch/losses.py
# License: MIT
class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:         # Pytorch 0.4.0 replacement (should be ok to use like this)
            return (output - target * torch.log(output + self.bias)).mean()


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :] * torch.log(output + self.bias)).mean()


class L1Loss3d(nn.Module):
    def __init__(self, bias=1e-12, **kwargs):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).abs().mean()


class MSE3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :]).pow(2).mean()


# ==== Custom ==== #
class BCEWithLogitsViewLoss(nn.BCEWithLogitsLoss):
    '''
    Silly wrapper of nn.BCEWithLogitsLoss because BCEWithLogitsLoss only takes a 1-D array
    '''
    def __init__(self, weight=None, size_average=True, **kwargs):
        super().__init__(weight=weight, size_average=size_average)

    def forward(self, input, target):
        '''
        :param input:
        :param target:
        :return:

        Simply passes along input.view(-1), target.view(-1)
        '''
        return super().forward(input.view(-1), target.view(-1))


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
    def __init__(self, weight=None, size_average=True, num_classes=2, **kwargs):
        super(mIoULoss, self).__init__()
        self.classes = num_classes

    def forward(self, inputs, target_oneHot):
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

    def __init__(self, use_running_mean=False, bce_weight=1, dice_weight=1, eps=1e-6, gamma=0.9, combined_loss_only=True, **kwargs):
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

        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.bce_logits_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, targets):
        # inputs and targets are assumed to be BxCxWxH (batch, color, width, height)
        outputs = outputs.squeeze()       # necessary in case we're dealing with binary segmentation (color dim of 1)
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(-0) == targets.size(-0)
        assert outputs.size(-1) == targets.size(-1)
        assert outputs.size(-2) == targets.size(-2)

        bce_loss = self.bce_logits_loss(outputs, targets)

        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean == False:
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
                 combined_loss_only=False,
                 **kwargs
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

        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def to(self, device):
        super().to(device=device)
        self.nll_loss.to(device=device)

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self,
                outputs,
                targets,
                weights):
        # inputs and targets are assumed to be BxCxWxH
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(2) == targets.size(2)
        assert outputs.size(3) == targets.size(3)

        # weights are assumed to be BxWxH
        # assert that B, W and H are the are the same for target and mask
        assert outputs.size(0) == weights.size(0)
        assert outputs.size(2) == weights.size(1)
        assert outputs.size(3) == weights.size(2)

        if self.use_weight_mask:
            bce_loss = F.binary_cross_entropy_with_logits(input=outputs,
                                                          target=targets,
                                                          weight=weights)
        else:
            bce_loss = self.nll_loss(input=outputs,
                                     target=targets)

        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + self.eps
        dice_loss = (-torch.log(2 * intersection / union))

        if self.use_running_mean == False:
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
# Source: https://github.com/PkuRainBow/OCNet/blob/master/utils/loss.py
# Description: http://www.erogol.com/online-hard-example-mining-pytorch/
# Online Hard Example Loss
class OhemCrossEntropy2d(nn.Module):
    """
    Online Hard Example Loss with Cross Entropy (used for classification)

    OHEM description: http://www.erogol.com/online-hard-example-mining-pytorch/
    """
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def to(self, device):
        super().to(device=device)
        self.criterion.to(device=device)

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


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

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
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

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


# ====================== #
# Source: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
# OHEM Segmentation Loss
class OHEMSegmentationLosses(OhemCrossEntropy2d):
    """
    2D Cross Entropy Loss with Auxiliary Loss

    """
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(OHEMSegmentationLosses, self).__init__(ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def to(self, device):
        super().to(device=device)
        self.bceloss.to(device=device)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(OHEMSegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
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


# Source: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/TverskyLoss/binarytverskyloss.py (MIT)
class FocalBinaryTverskyFunc(Function):
    """
        Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_

        `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.

        Params:
            :param alpha: controls the penalty for false positives.
            :param beta: penalty for false negative.
            :param gamma : focal coefficient range[1,3]
            :param reduction: return mode

        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
            add focal index -> loss=(1-T_index)**(1/gamma)
    """

    def __init__(ctx, alpha=0.5, beta=0.7, gamma=1.0, reduction='mean'):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = 1e-6
        ctx.reduction = reduction
        ctx.gamma = gamma
        sum = ctx.beta + ctx.alpha
        if sum != 1:
            ctx.beta = ctx.beta / sum
            ctx.alpha = ctx.alpha / sum

    # @staticmethod
    def forward(ctx, input, target):
        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon)
        loss = torch.pow((1 - index), 1 / ctx.gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if ctx.reduction == 'none':
            loss = loss
        elif ctx.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (ctx.alpha * (1 - target) + target) * P_G

        dL_dT = (1 / ctx.gamma) * torch.pow((P_G / sum), (1 / ctx.gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = ctx.beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None


class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation

    Args
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each class. If given, it has to be a Tensor of size `C`
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, reduction='mean', weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyFunc(self.alpha, self.beta, self.gamma, self.reduction)
            loss_idx = loss_func(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses


class FocalBinaryTverskyLoss(MultiTverskyLoss):
    """
            Binary version of Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_

            `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.

            Params:
                :param alpha: controls the penalty for false positives.
                :param beta: penalty for false negative.
                :param gamma : focal coefficient range[1,3]
                :param reduction: return mode

            Notes:
                alpha = beta = 0.5 => dice coeff
                alpha = beta = 1 => tanimoto coeff
                alpha + beta = 1 => F beta coeff
                add focal index -> loss=(1-T_index)**(1/gamma)
        """

    def __init__(self, alpha=0.5, beta=0.7, gamma=1.0, reduction='mean', **kwargs):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        """
        super().__init__(alpha, beta, gamma, reduction)

    def forward(self, inputs, targets):
        return super().forward(inputs, targets.unsqueeze(1))

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
    def __init__(self, reduction='mean', **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
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

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


# ===================== #
# Inspired by: https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py (MIT)
# Unfortunately the implementation above seems wrong, so reimplementing per the gist of the paper
class ActiveContourLoss(nn.Module):
    """
        `Learning Active Contour Models for Medical Image Segmentation <http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf>`_
        Note that is only works for B/W masks right now... which is kind of the point of this loss as contours in RGB should be cast to B/W
        before computing the loss.

        Params:
            :param len_w: (float, default=1.0) - The multiplier to use when adding boundary loss.
            :param reg_w: (float, default=1.0) - The multiplier to use when adding region loss.
            :param apply_log: (bool, default=True) - Whether to transform the log into log space (due to the
    """

    def __init__(self, len_w=1., reg_w=1., apply_log=True, **kwargs):
        super(ActiveContourLoss, self).__init__()
        self.len_w = len_w
        self.reg_w = reg_w
        self.epsilon = 1e-8  # a parameter to avoid square root = zero issues
        self.apply_log = apply_log

    def forward(self, logits, target):
        image_size = logits.size(3)
        target = target.unsqueeze(1)

        # must convert raw logits to predicted probabilities for each pixel along channel
        probs = F.softmax(logits, dim=0)

        """
        length term:
            - Subtract adjacent pixels from each other in X and Y directions
            - Determine where they differ from the ground truth (targets)
            - Calculate MSE
        """
        # horizontal and vertical directions
        x = probs[:, :, 1:, :] - probs[:, :, :-1, :]      # differences in horizontal direction
        y = probs[:, :, :, 1:] - probs[:, :, :, :-1]      # differences in vertical direction

        target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_y = target[:, :, :, 1:] - target[:, :, :, :-1]

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
        error_in = probs[:, 0, :, :] * ((target[:, 0, :, :] - 1) ** 2)  # invert the ground truth mask and multiply by probs

        # the sum of all pixel values that are not equal 1 inside of the ground truth mask
        probs_diff = (probs[:, 0, :, :] - target[:, 0, :, :]).abs()     # subtract mask from probs giving us the errors
        error_out = (probs_diff * target[:, 0, :, :])                   # multiply mask by error, giving us the error terms inside the mask.

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
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

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


class BDLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, logits, target, bound):
        """
        Takes 2D or 3D logits.

        logits: (batch_size, class, x,y,(z))
        target: ground truth, shape: (batch_size, 1, x,y,(z))
        bound: precomputed distance map, shape (batch_size, class, x,y,(z))

        Torch Eigensum description: https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
        """
        compute_directive = "bcxy,bcxy->bcxy"
        if len(logits) == 5:
            compute_directive = "bcxyz,bcxyz->bcxyz"

        net_output = softmax_helper(logits)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:,1:, ...].type(torch.float32)

        multipled = torch.einsum(compute_directive, pc, dc)
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

    def __init__(self, alpha, beta, eps=1e-7, **kwargs):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
            :param targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            :return: loss
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targets.squeeze(1).long()]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
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
