from typing import Sequence, Dict, Any, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

def normalize_image(img: torch.Tensor) -> torch.Tensor:
    img = img.to(torch.float32)
    img /= 127
    img -= 1
    return img

class LabeledDataset(Dataset):
    def __init__(self,
                 all_files: Sequence[str],
                 labeled_idxs: Sequence[int],
                 labeled_transforms:  torchvision.transforms.Compose,
                 train: bool) -> None:
        """
        Args:
            all_files: list of all files in the dataset
            labeled_idxs: list of indexes of labeled data
            labeled_transforms: transforms to apply to labeled data
            train: whether the dataset is used for training
        """

        super().__init__()
        self.all_files = all_files
        self.labeled_idxs = labeled_idxs
        self.labeled_transforms = labeled_transforms
        self.train = train

    def __len__(self) -> int:
        return len(self.labeled_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor,  Any, str]]:
        _idx = self.labeled_idxs[idx]

        fn = self.all_files[_idx]

        img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

        msk_pre = cv2.imread(fn.replace('/images/', '/masks/'),
                             cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)

        msk0 = msk_pre
        msk1 = np.zeros_like(msk_post)
        msk2 = np.zeros_like(msk_post)
        msk3 = np.zeros_like(msk_post)
        msk4 = np.zeros_like(msk_post)
        msk1[msk_post == 1] = 255
        msk2[msk_post == 2] = 255
        msk3[msk_post == 3] = 255
        msk4[msk_post == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (np.greater(msk, 127)) * 1.
        img = np.concatenate([img_pre, img_post], axis=2)

        # Reshaping tensors from (H, W, C) to (C, H, W)
        # we need unit8 for the transforms
        img = torch.from_numpy(img.transpose((2, 0, 1))).to(torch.uint8)
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).float()

        if self.labeled_transforms is not None and self.train:
            # cat img and masks to perform same transformations on both
            concat = torch.cat([img, msk], dim=0)
            concat = self.labeled_transforms(concat)
            img_shape = img.shape
            # check dimensions
            img = concat[:img_shape[0], ...]
            msk = concat[img_shape[0]:, ...]

        # we normalize the image after the transforms and change it back to float
        img = normalize_image(img)
        return {'img': img, 'msk': msk, 'fn': fn}
