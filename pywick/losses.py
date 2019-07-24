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

VOID_LABEL = 255
N_CLASSES = 1


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
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (self.num_class - 1), 1.0 - self.smooth)
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

        b,c,h,w = inputs.size()
        class_mask = Variable(torch.zeros([b,c+1,h,w]).cuda())
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

# ====================== #
# Source: https://github.com/snakers4/mnasnet-pytorch/blob/master/src/models/semseg_loss.py
# Combination Loss from BCE and Dice
class ComboSemsegLoss(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=1,
                 dice_weight=1,
                 eps=1e-10,
                 gamma=0.9
                 ):
        super().__init__()

        self.nll_loss = nn.BCEWithLogitsLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

    def reset_parameters(self):
        self.running_bce_loss.zero_()
        self.running_dice_loss.zero_()

    def forward(self, outputs, targets):
        # inputs and targets are assumed to be BxCxWxH
        assert len(outputs.shape) == len(targets.shape)
        # assert that B, W and H are the same
        assert outputs.size(-0) == targets.size(-0)
        assert outputs.size(-1) == targets.size(-1)
        assert outputs.size(-2) == targets.size(-2)

        bce_loss = self.nll_loss(outputs, targets)

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

        return loss, bce_loss, dice_loss


class ComboSemsegLossWeighted(nn.Module):
    def __init__(self,
                 use_running_mean=False,
                 bce_weight=1,
                 dice_weight=1,
                 eps=1e-10,
                 gamma=0.9,
                 use_weight_mask=False,
                 ):
        super().__init__()

        self.use_weight_mask = use_weight_mask

        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.gamma = gamma

        self.use_running_mean = use_running_mean
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.use_running_mean == True:
            self.register_buffer('running_bce_loss', torch.zeros(1))
            self.register_buffer('running_dice_loss', torch.zeros(1))
            self.reset_parameters()

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
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000, use_weight=True):
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
class FocalBinaryTverskyLoss(Function):
    """
        Focal Tversky Loss as defined in `this paper <https://arxiv.org/abs/1810.07842>`_

        `Authors' implementation <https://github.com/nabsabraham/focal-tversky-unet>`_ in Keras.

        Args
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

    def __init__(ctx, alpha=0.5, beta=0.7, gamma=1.33333, reduction='mean'):
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

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, weights=None):
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
            loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
            loss_idx = loss_func(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses
