
from typing import Dict

import torch
import torch.nn as nn

from losses.focal_loss import FocalLoss
from losses.dice_loss import SoftDiceLoss

class ComboLoss(nn.Module):
    """ Combination of multiple losses.

        Args:
            weights: a dictionary of the form {loss_name: weight}
            reduction: 'none', 'mean', 'sum'
            """

    def __init__(self, weights: Dict[str, float], reduction) -> None:
        super().__init__()
        self.weights = weights
        self.dice = DiceLoss(per_image=True)
        self.focal = FocalLoss(reduction=reduction)
        self.mapping = {
            'dice': self.dice,
            'focal': self.focal}
        self.expect_sigmoid = {'dice', 'focal'}
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

if __name__=="__main__":
    from legacy import losses
    combo_loss = ComboLoss({'dice': 1, 'focal': 1}, reduction='mean')
    logits = torch.randn(12, 5, 32, 32)
    targets = torch.randn(12, 5, 32, 32)
    combo_loss2 = losses.ComboLoss({'dice': 1, 'focal': 1})
    print("new",combo_loss(logits, targets))
    print("old",combo_loss2(logits, targets))