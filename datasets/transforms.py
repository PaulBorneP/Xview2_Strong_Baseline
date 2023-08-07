import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class ResizeCrop(torch.nn.Module):
    def __init__(self,
                 p: float,
                 input_size: int,
                 weights: Sequence[float] = [
                     1, 9.04788032, 8.68207691, 12.9632271]) -> None:
        """ResizeCrop class.

        Args:
            p: Probability of applying a random size of crop 
                (different from the input size).
            input_size: Size of the input image after cropping and resizing.
            weights: Weights to be applied to the different classes in the
                cropped masks. We have one weight per damage class (4). 
                The default values are the inverse of the frequency of each
                class in the training/validation set with no contamination
                (see generalization section in the paper)
        """

        super().__init__()
        self.p = p
        self.input_size = input_size
        self.w = weights

    def forward(self,
                tensor: torch.Tensor) -> torch.Tensor:

        crop_size = self.input_size
        if random.random() > self.p:
            crop_size = random.randint(
                int(self.input_size / 1.15), int(self.input_size / 0.85))
            # We need to check that the tensor has the expected shape (11,H,W)
            # 11 = img (6 channels) + mask (5 channels) (labeled data)
            assert tensor.shape[0] == 11
            msk = tensor[6:, ...]
            bst_x0 = random.randint(0, tensor.shape[1] - crop_size)
            bst_y0 = random.randint(0, tensor.shape[2] - crop_size)
            bst_sc = -1
            try_cnt = random.randint(1, 10)
            for _ in range(try_cnt):
                x0 = random.randint(0, tensor.shape[1] - crop_size)
                y0 = random.randint(0, tensor.shape[2] - crop_size)
                # We try to get more of certain classes in the cropped masks.
                _sc = msk[2, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[1] + \
                    msk[3, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[2] + \
                    msk[4, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[3] + \
                    msk[1, y0:y0 + crop_size, x0:x0 + crop_size].sum() * \
                    self.w[0]
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0
            tensor = tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]
        # Not to bilinear interpolation because it makes tensor values != 0 and 1
        tensor = TF.resize(img=tensor,
                        size=[self.input_size, self.input_size],
                        interpolation=transforms.InterpolationMode.NEAREST,
                        antialias=True)

        return tensor
