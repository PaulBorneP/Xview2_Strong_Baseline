# adapted from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/focal_loss.py and legacy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param reduction: (string) Specifies the reduction to apply to the output:
                    'none' | 'mean' | 'sum' |. By default => 'none': no reduction will be applied,
    """

    def __init__(self, gamma=2, smooth=1e-6, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
        if target.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            target = target.view(target.size(0), target.size(1), -1)

        if self.smooth:
          logit = torch.clamp(logit, self.smooth, 1. - self.smooth)
          target = torch.clamp(target, self.smooth, 1. - self.smooth)
        pt = (1 - logit) * (1 - target) + logit * target
        loss = (-(1. - pt) ** self.gamma * torch.log(pt))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            # mean over all dimension except batch
            return loss.mean(dim=list(range(len(loss.shape)))[1:])
            
        else:
            raise NotImplementedError
    
if __name__ == "__main__":
    from legacy.losses import FocalLoss2d
    focal_loss = FocalLoss(reduction='mean')
    logit = torch.rand(12,5,128,128)
    target = torch.rand(12,5,128,128)
    loss = focal_loss(logit,target)
    test_loss = FocalLoss2d()
    loss2 = test_loss(logit,target)
    assert loss == loss2

    focal_loss = FocalLoss(reduction='none')
    logit = torch.rand(12,5,128,128)
    target = torch.rand(12,5,128,128)
    loss = focal_loss(logit,target)
    print(loss.shape)