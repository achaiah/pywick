"""
Losses are critical to training a neural network well. The training can only make progress if you
provide a meaningful measure of loss for each training step. What the loss looks like usually depends
on your application. Pytorch has a number of `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions/>`_ that
you can use out of the box. However, some more advanced and cutting edge loss functions exist that are not (yet) part of
Pytorch. We include those below for your experimenting.\n
**Caution:** if you decide to use one of these, you will definitely want to peruse the source code first, as it has
many additional useful notes and references which will help you.
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

VOID_LABEL = 255
N_CLASSES = 1


def crossentropyloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, label)
    target = label.view(-1)[mask]
    C = logits.size(1)
    logits = logits.permute(0, 2, 3, 1)  # B, H, W, C
    logits = logits.contiguous().view(-1, C)
    mask2d = mask.unsqueeze(1).expand(mask.size(0), C).contiguous().view(-1)
    logits = logits[mask2d].view(-1, C)
    loss = F.cross_entropy(logits, target)
    return loss


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
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

# WARNING THIS IS VERY SLOW FOR SOME REASON!!
def dice_coefficient(logit, label, isCuda = True):
    '''
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
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        #print('logits: {}, targets: {}'.format(logits.size(), targets.size()))
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        smooth = 1.

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class FocalLoss(nn.Module):
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
    def __init__(self, threshold=0.5):
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
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()
        self.tl1 = ThresholdedL1Loss(threshold=threshold)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets) + self.tl1(logits, targets)

class BCEDiceFocalLoss(nn.Module):
    '''
    :param l: l-parameter for FocalLoss
    :param weight_of_focal: How to weigh the focal loss (between 0 - 1)
    '''
    def __init__(self, l=0.5, weight_of_focal=1.):
        super(BCEDiceFocalLoss, self).__init__()
        # self.bce = BCELoss2d()
        # self.dice = SoftDiceLoss()
        self.dice = BCELoss2d()
        self.focal = FocalLoss(l=l)
        self.weight_of_focal = weight_of_focal

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.weight_of_focal * self.focal(logits, targets)

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)

class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
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
    def __init__(self, kernel_size=21):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def forward(self, logits, labels):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.ones(a.size())

        w0 = weights.sum()
        weights = weights + ind.cpu() * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        loss = self.bce(logits.cpu(), labels.cpu(), weights) + self.dice(logits.cpu(), labels.cpu(), weights)

        return loss

# ==== Another Focal Loss ==== #
# Source: https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
# License: MIT
def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    return mask.scatter_(1, index, ones)


class FocalLoss2(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()

# -------- #
# Source: https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
class BinaryFocalLoss3(nn.Module):
    '''
        gamma = 0 is equivalent to BinaryCrossEntropy Loss
    '''
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()
# -------- #

# ==== Additional Losses === #
# Source: https://github.com/atlab/attorch/blob/master/attorch/losses.py
# License: MIT
class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-12):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:         # Pytorch 0.4.0 replacement (should be ok to use like this)
            return (output - target * torch.log(output + self.bias)).mean()


class PoissonLoss3d(nn.Module):
    def __init__(self, bias=1e-12):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        # _assert_no_grad(target)
        with torch.no_grad:  # Pytorch 0.4.0 replacement (should be ok to use like this)
            lag = target.size(1) - output.size(1)
            return (output - target[:, lag:, :] * torch.log(output + self.bias)).mean()

class L1Loss3d(nn.Module):
    def __init__(self, bias=1e-12):
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
    def __init__(self, weight=None, size_average=True):
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
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

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