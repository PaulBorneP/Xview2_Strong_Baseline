from typing import Generator, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse
eps = 1e-6


def dice_round(outputs: torch.Tensor, targets: torch.tensor) -> torch.Tensor:
    """Convert predictions to float then calculate the dice loss.
    """
    outputs = outputs.float()
    return soft_dice_loss(outputs, targets)


def iou_round(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    outputs = outputs.float()
    return jaccard(outputs, targets)


def soft_dice_loss(outputs: torch.Tensor, targets: torch.Tensor, per_image: bool = False) -> torch.Tensor:
    """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.

        Args:
            outputs: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
            targets: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
            per_image: if True, compute the dice loss per image instead of per batch

        Returns:
            dice_loss: the dice loss.  
    """

    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def jaccard(outputs: torch.Tensor, targets: torch.Tensor, per_image: bool = False) -> torch.Tensor:
    """Jaccard or Intersection over Union.

    Args:
        outputs: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
        targets: a tensor of shape [batch_size, num_classes, *spatial_dimensions]
        per_image: if True, compute the jaccard loss per image instead of per batch

    Returns:
        jaccard_loss: the jaccard loss.
    """
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + \
        torch.sum(dice_target, dim=1) - intersection + eps
    losses = 1 - (intersection + eps) / union
    return losses.mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    union = gts.float() + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = True, ignore: int = None) -> torch.Tensor:
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor, ignore: int = None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_sigmoid(probas: torch.Tensor, labels: torch, per_image: bool = False, ignore: int = None) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_sigmoid_flat(*flatten_binary_scores(prob.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_sigmoid_flat(
            *flatten_binary_scores(probas, labels, ignore))
    return loss


def lovasz_sigmoid_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    fg = labels.float()
    errors = (Variable(fg) - probas).abs()
    errors_sorted, perm = torch.sort(errors, 0, descending=True)
    perm = perm.data
    fg_sorted = fg[perm]
    loss = torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted)))
    return loss


def mean(l: Generator, ignore_nan: bool = False, empty: int = 0) -> float:
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    """ Lovasz hinge loss.

        Args:
            per_image: compute the loss per image instead of per batch
            ignore: void class id
    """
    def __init__(self, ignore_index: int = 255, per_image: bool = True) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_hinge(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class LovaszLossSigmoid(nn.Module):
    """ Lovasz sigmoid loss.

        Args:   
            per_image: compute the loss per image instead of per batch
            ignore: void class id
    """
    def __init__(self, ignore_index: bool = 255, per_image: bool = True) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_sigmoid(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class FocalLoss2d(nn.Module):
    """ Focal loss class.
    
        Args:  
            gamma: gamma value for calculating the modulating factor
            ignore_index: index to ignore from loss calculation
    """
    def __init__(self, gamma: float = 2, ignore_index: int = 255) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        # eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class DiceLoss(nn.Module):
    """ Soft dice loss class.

        Args:
            weight: an array of shape [num_classes,]
            size_average: boolean, True by default
            per_image: if True, compute the dice loss per image instead of per batch
    """

    def __init__(self, weight: torch.Tensor = None, size_average: bool = True, per_image: bool = False) -> None:
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(outputs, targets, per_image=self.per_image)


class JaccardLoss(nn.Module):
    """ Jaccard or Intersection over Union loss class."""

    def __init__(self, weight: torch.Tensor = None, size_average: bool = True, per_image: bool = False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, outputs, targets):
        return jaccard(outputs, targets, per_image=self.per_image)


class StableBCELoss(nn.Module):

    # not sure this it useful
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = - input.abs()
        # todo check correctness
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class ComboLoss(nn.Module):
    """ Combination of multiple losses.

        Args:
            weights: a dictionary of the form {loss_name: weight}
            per_image: if True, compute the loss per image instead of per batch
            """

    def __init__(self, weights: Dict[str, float], per_image: bool = False) -> None:
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        # see if per_image True works
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard,
                        'lovasz': self.lovasz,
                        'lovasz_sigmoid': self.lovasz_sigmoid}
        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
        self.values = {}

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = self.mapping[k](
                sigmoid_input if k in self.expect_sigmoid else outputs, targets)
            self.values[k] = val
            loss += self.weights[k] * val
        return loss
