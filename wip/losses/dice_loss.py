

import torch
from torch import nn
import numpy as np

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, square=False, reduction='none'):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf

        :param smooth: smoothing factor in the denominator
        :param square: if True then fp, tp and fn will be squared before summation
        :param reduction: 'none', 'mean', 'sum'

        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logit, target):
        # N,C,m -> N*c*m (m=d1*d2*...)
        batch_size = logit.size()[0]

        dice_target = target.contiguous().view(batch_size, -1).float()
        dice_output = logit.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + self.smooth
        loss = (1 - (2 * intersection + self.smooth) / union)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            # we changed this to have a per sample loss
            loss = loss
        else:
            raise ValueError(
                "reduction must be one of 'mean', 'sum' or 'none'")
        return loss

if __name__ == "__main__":
    from legacy.losses import DiceLoss
    loss = SoftDiceLoss(reduction='mean')
    loss2 = DiceLoss(per_image=True)

    x = torch.randn(12, 5, 32, 32)
    y = torch.randn(12, 5, 32, 32)

    print(loss(x, y))
    print(loss2(x, y))
