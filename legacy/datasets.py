#  Dataset classes for training and validation for localization (binary segmentation)  and classification (semantic segmentation) tasks

import os
from os import path, listdir
from pathlib import Path
import random
from typing import Sequence, Dict, Tuple, List
import shutil
import tqdm
import cv2

from imgaug import augmenters as iaa
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.morphology import square, dilation
from legacy.utils import *
import torch
from torch.utils.data import Dataset

np.random.seed(1)
random.seed(1)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def normalize_image(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


class LocalizationDataset(Dataset):
    """Dataset for localization task.

        Args:
            idxs : Idxs of training or validation images.
            all_files : Path to all images.
            input_shape :  Shape of input images (should be the same for all the 
                images in the dataset) used when resizing or cropping.
            is_train : If True, the dataset is used for training. Data augmentation is 
                only performed at training time If False, the dataset is used for pi
                validation. Defaults to True.
            low_aug : Defines what type of data augmentation to use. If False, only rotation, flipping,
                scaling (zooming), cropping, switching pre and post are applied. If False, shifting,
                switching rgb or hsv channel, clahe, gauss_noise, blur, saturation, brightness
                and contrast can be applied. Not used in eval Defaults to False.
    """

    def __init__(self,
                 idxs: Sequence[int],
                 all_files: Sequence[Path],
                 input_shape: Sequence[int],
                 is_train: bool = True,
                 low_aug: bool = False) -> None:

        super().__init__()
        self.idxs = idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.all_files = all_files
        self.input_shape = input_shape
        self.is_train = is_train
        self.low_aug = low_aug

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        _idx = self.idxs[idx]
        fn = self.all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'),
                          cv2.IMREAD_UNCHANGED)

        if self.is_train:
            img, msk0 = self.augment_loc_data(img, msk0, fn)

        else:
            pass

        img = normalize_image(img)

        # Cleaning up mask and reshaping to image shape
        msk = msk0[..., np.newaxis]
        msk = (msk > 127) * 1

        # Reshaping tensors from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk_tensor = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        return {'img': img_tensor, 'msk': msk_tensor, 'fn': fn}

    def augment_loc_data(self, image: np.ndarray, mask: np.ndarray, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the data.

            Args:
                image : Image to augment.
                mask : Mask to augment.
                filename : Path to the non augmented image.

            Returns:
                Augmented image and mask.
        """

        # Randomly take post-disaster image instead of pre-disaster to make the model more robust
        if random.random() > 0.985:
            image = cv2.imread(filename.replace(
                '_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        # Flipping vertically
        if random.random() > 0.5:
            image = image[::-1, ...]
            mask = mask[::-1, ...]

        # k * 90 degree rotation
        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                image = np.rot90(image, k=rot)
                mask = np.rot90(mask, k=rot)

        # Shifting
        if (not self.low_aug) and (random.random() > 0.8):
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            image = shift_image(image, shift_pnt)
            mask = shift_image(mask, shift_pnt)

        # Rotation and scaling
        if random.random() > 0.2:
            rot_pnt = (image.shape[0] // 2 + random.randint(-320, 320),
                       image.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                image = rotate_image(image, angle, scale, rot_pnt)
                mask = rotate_image(mask, angle, scale, rot_pnt)

        # Cropping (it is actually not really a augmentation per say, but it is
        # treated as one in the submission (only at training time which is weird
        # and in the middle of the data augmentation pipeline). Nevertheless cropping size may vary from the input size as
        # as data augmentation, so I guess it is fine to treat cropping as  data augmentation.)

        crop_size = self.input_shape[0]
        if random.random() > 0.3:
            crop_size = random.randint(
                int(self.input_shape[0] / 1.2), int(self.input_shape[0] / 0.8))

        bst_x0 = random.randint(0, image.shape[1] - crop_size)
        bst_y0 = random.randint(0, image.shape[0] - crop_size)

        bst_sc = -1
        try_cnt = random.randint(1, 5)
        for i in range(try_cnt):
            x0 = random.randint(0, image.shape[1] - crop_size)
            y0 = random.randint(0, image.shape[0] - crop_size)
            _sc = mask[y0:y0 + crop_size, x0:x0 + crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        image = image[y0:y0 + crop_size, x0:x0 + crop_size, :]
        mask = mask[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            image = cv2.resize(image, self.input_shape,
                               interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)

        # Don't execute these augmentations in low augment setting
        if not self.low_aug:

            if random.random() > 0.97:
                image = shift_channels(
                    image, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            elif random.random() > 0.97:
                image = change_hsv(image, random.randint(-5, 5),
                                   random.randint(-5, 5), random.randint(-5, 5))
            if random.random() > 0.93:
                if random.random() > 0.97:
                    image = clahe(image)
                elif random.random() > 0.97:
                    image = gauss_noise(image)
                elif random.random() > 0.97:
                    image = cv2.blur(image, (3, 3))
            elif random.random() > 0.93:
                if random.random() > 0.97:
                    image = saturation(image, 0.9 + random.random() * 0.2)
                elif random.random() > 0.97:
                    image = brightness(image, 0.9 + random.random() * 0.2)
                elif random.random() > 0.97:
                    image = contrast(image, 0.9 + random.random() * 0.2)

            if random.random() > 0.97:
                el_det = self.elastic.to_deterministic()
                image = el_det.augment_image(image)
        return image, mask


class ClassificationDataset(Dataset):
    """Dataset for training classification model."""

    def __init__(self,
                 idxs: Sequence[int],
                 all_files: Sequence[Path],
                 input_shape: Sequence[int],
                 low_aug: bool,
                 dilate: bool,
                 task: str = "training",
                 loc_folder: str = None
                 ) -> None:
        """
        Args:
            idxs : Idxs of training images.
            all_files : Path to all images.
            input_shape :  Shape of input images (should be the same for all the
                images in the dataset) used when resizing or cropping.
            low_aug : Defines what type of data augmentation to use. If False, only rotation, flipping,
                scaling (zooming), cropping, switching pre and post are applied. If False, shifting,
                switching rgb or hsv channel, clahe, gauss_noise, blur, saturation, brightness
                and contrast can be applied. Not used in eval Defaults to False.
            dilate : Defines if the mask should be dilated. Only used for classification. Defaults to False.
            task : Defines if the dataset is used for training, tunning or validation (set to "train", "tune" or "val"). Defaults to "training".
            loc_folder : Path to the folder where masks used for localization (pre_disaster) are stored. Only used in eval. Defaults to None.
         """

        super().__init__()
        self.dilate = dilate
        self.low_aug = low_aug
        self.input_shape = input_shape
        self.all_files = all_files
        self.idxs = idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.task = task
        self.loc_folder = loc_folder

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _idx = self.idxs[idx]

        fn = self.all_files[_idx]

        img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)

        msk_pre = cv2.imread(fn.replace('/images/', '/masks/'),
                             cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
                              cv2.IMREAD_UNCHANGED)

        if self.task == "val":
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
            msk = (msk > 127)
            if self.loc_folder is None:
                msk_loc = torch.Tensor([0.0])
            else:
                msk_loc = cv2.imread(
                    path.join(self.loc_folder, fn.split('/')
                              [-1].replace('.png', '_part1.png')),
                    cv2.IMREAD_UNCHANGED) > (0.3 * 255)
        else:

            if self.task == "tune":
                img_pre, img_post, msk = self.augmentations_tune(
                    img_pre, img_post, msk_pre, msk_post)

            elif self.task == "train":
                img_pre, img_post, msk = self.augmentations_train(
                    img_pre, img_post, msk_pre, msk_post)
            else:
                raise ValueError(
                    "Task should be either 'train', 'tune' or 'val'")

            # Dilating masks we make sure that different classes are not overlapping
            if self.dilate:
                msk[..., 0] = False
                msk[..., 1] = dilation(msk[..., 1], square(5))
                msk[..., 2] = dilation(msk[..., 2], square(5))
                msk[..., 3] = dilation(msk[..., 3], square(5))
                msk[..., 4] = dilation(msk[..., 4], square(5))
                msk[..., 1][msk[..., 2:].max(axis=2)] = False
                msk[..., 3][msk[..., 2]] = False
                msk[..., 4][msk[..., 2]] = False
                msk[..., 4][msk[..., 3]] = False
                msk[..., 0][msk[..., 1:].max(axis=2)] = True

        msk = msk * 1
        lbl_msk = msk[..., 1:].argmax(
            axis=2) if self.task == "val" else msk.argmax(axis=2)
        img = np.concatenate([img_pre, img_post], axis=2)
        img = normalize_image(img)

        # Reshaping tensors from (H, W, C) to (C, H, W)
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        if self.task == "val":
            return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn, 'msk_loc': msk_loc}

        else:
            return {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}

    # The two augmentation sets only seem to differ by their parameters. It would be more elegant to parameterize them
    # and avoid the redundant code. However, that's also too much work for the small benefit.

    def augmentations_tune(self,
                           img_pre: np.ndarray,
                           img_post: np.ndarray,
                           msk_pre: np.ndarray,
                           msk_post: np.ndarray) -> Tuple[np.ndarray]:
        """Augmentations for the tune stage.

            Args:
                img_pre : Pre-disaster image.
                img_post : Post-disaster image.
                msk_pre : Pre-disaster mask. The pre-disaster mask is a binary mask
                    and is considered the ground truth for what is a building or not.
                msk_post : Post-disaster mask. The post-disaster mask is a multi-class
                    mask that gives us the level of damage. The scale of damage goes from
                    1 (no damage) to 4 (destroyed) 0 is left for background
                    (even though we retrieve this information from the pre_mask)."""

        # TODO switching to a List of masks and map operations would be more elegant
        msk1 = np.zeros_like(msk_post)
        msk2 = np.zeros_like(msk_post)
        msk3 = np.zeros_like(msk_post)
        msk4 = np.zeros_like(msk_post)
        msk1[msk_post == 1] = 255
        msk2[msk_post == 2] = 255
        msk3[msk_post == 3] = 255
        msk4[msk_post == 4] = 255

        # flipping vertically
        if random.random() > 0.7:
            img_pre = img_pre[::-1, ...]
            img_post = img_post[::-1, ...]
            msk_pre = msk_pre[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]
        # k * 90 degree rotation
        if random.random() > 0.3:
            rot = random.randrange(4)
            if rot > 0:
                img_pre = np.rot90(img_pre, k=rot)
                img_post = np.rot90(img_post, k=rot)
                msk_pre = np.rot90(msk_pre, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)

        # shifting
        if (not self.low_aug) and (random.random() > 0.99):
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img_pre = shift_image(img_pre, shift_pnt)
            img_post = shift_image(img_post, shift_pnt)
            msk_pre = shift_image(msk_pre, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)

        # scaling and rotation
        if random.random() > 0.5:
            rot_pnt = (img_pre.shape[0] // 2 + random.randint(-320, 320),
                       img_pre.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img_pre = rotate_image(img_pre, angle, scale, rot_pnt)
                img_post = rotate_image(img_post, angle, scale, rot_pnt)
                msk_pre = rotate_image(msk_pre, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        # cropping
        crop_size = self.input_shape[0]
        if random.random() > 0.5:
            crop_size = random.randint(
                int(self.input_shape[0] / 1.1), int(self.input_shape[0] / 0.9))

        bst_x0 = random.randint(0, img_pre.shape[1] - crop_size)
        bst_y0 = random.randint(0, img_pre.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img_pre.shape[1] - crop_size)
            y0 = random.randint(0, img_pre.shape[0] - crop_size)

            # for new dataset
            _sc = msk2[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 9.784601349119356 + \
                msk3[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 7.696044296563575 + \
                msk4[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 17.5313799876951 + \
                msk1[y0:y0 + crop_size, x0:x0 + crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img_pre = img_pre[y0:y0 + crop_size, x0:x0 + crop_size, :]
        img_post = img_post[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk_pre = msk_pre[y0:y0 + crop_size, x0:x0 + crop_size]
        msk1 = msk1[y0:y0 + crop_size, x0:x0 + crop_size]
        msk2 = msk2[y0:y0 + crop_size, x0:x0 + crop_size]
        msk3 = msk3[y0:y0 + crop_size, x0:x0 + crop_size]
        msk4 = msk4[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            img_pre = cv2.resize(img_pre, self.input_shape,
                                 interpolation=cv2.INTER_LINEAR)
            img_post = cv2.resize(img_post, self.input_shape,
                                  interpolation=cv2.INTER_LINEAR)
            msk_pre = cv2.resize(msk_pre, self.input_shape,
                                 interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)

        # Don't execute these augmentations in low augment setting
        if not self.low_aug:

            # augmentation can not be applied to both images
            if random.random() > 0.99:
                img_pre = shift_channels(
                    img_pre, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            elif random.random() > 0.99:
                img_post = shift_channels(
                    img_post, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            if random.random() > 0.99:
                img_pre = change_hsv(img_pre, random.randint(-5, 5),
                                     random.randint(-5, 5), random.randint(-5, 5))
            elif random.random() > 0.99:
                img_post = change_hsv(img_post, random.randint(-5, 5),
                                      random.randint(-5, 5), random.randint(-5, 5))

            # augmentation can be applied to both images
            img_pre = self.non_channelwise_hard_aug(img_pre)
            img_post = self.non_channelwise_hard_aug(img_post)

        # reshape masks and concatenate to get the same shape as images
        msk_pre = msk_pre[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]
        msk = np.concatenate([msk_pre, msk1, msk2, msk3, msk4], axis=2)

        msk = (msk > 127)

        return img_pre, img_post, msk

    # TODO: parameterize probabilities of augmentation
    def non_channelwise_hard_aug(self, img: np.ndarray) -> np.ndarray:
        """Group of augmentations that can be applied to both images
        and hence warpped in a single function for readability.
        """

        if random.random() > 0.99:
            if random.random() > 0.99:
                img = clahe(img)
            elif random.random() > 0.99:
                img = gauss_noise(img)
            elif random.random() > 0.99:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.99:
            if random.random() > 0.99:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = contrast(img, 0.9 + random.random() * 0.2)
        if random.random() > 0.99:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        return img

    # TODO once class for both augmentations (parameterize probabilities of augmentation)
    def augmentations_train(self, img, img2, msk0, lbl_msk1):
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk3 = np.zeros_like(lbl_msk1)
        msk4 = np.zeros_like(lbl_msk1)
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255
        msk3[lbl_msk1 == 3] = 255
        msk4[lbl_msk1 == 4] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)

        if (not self.low_aug) and (random.random() > 0.9):
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)

        if random.random() > 0.6:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320),
                       img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        crop_size = self.input_shape[0]
        if random.random() > 0.2:
            crop_size = random.randint(
                int(self.input_shape[0] / 1.15), int(self.input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk2[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 9.047880324809197 + \
                msk3[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 8.682076906271057 + \
                msk4[y0:y0 + crop_size, x0:x0 + crop_size].sum() * 12.963227101725556 + \
                msk1[y0:y0 + crop_size, x0:x0 + crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0 + crop_size, x0:x0 + crop_size, :]
        img2 = img2[y0:y0 + crop_size, x0:x0 + crop_size, :]
        msk0 = msk0[y0:y0 + crop_size, x0:x0 + crop_size]
        msk1 = msk1[y0:y0 + crop_size, x0:x0 + crop_size]
        msk2 = msk2[y0:y0 + crop_size, x0:x0 + crop_size]
        msk3 = msk3[y0:y0 + crop_size, x0:x0 + crop_size]
        msk4 = msk4[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape,
                             interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, self.input_shape,
                              interpolation=cv2.INTER_LINEAR)

        if not self.low_aug:

            if random.random() > 0.985:
                img = shift_channels(
                    img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
            elif random.random() > 0.985:
                img2 = shift_channels(
                    img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

            if random.random() > 0.985:
                img = change_hsv(img, random.randint(-5, 5),
                                 random.randint(-5, 5), random.randint(-5, 5))
            elif random.random() > 0.985:
                img2 = change_hsv(img2, random.randint(-5, 5),
                                  random.randint(-5, 5), random.randint(-5, 5))

            if random.random() > 0.98:
                if random.random() > 0.985:
                    img = clahe(img)
                elif random.random() > 0.985:
                    img = gauss_noise(img)
                elif random.random() > 0.985:
                    img = cv2.blur(img, (3, 3))
            elif random.random() > 0.98:
                if random.random() > 0.985:
                    img = saturation(img, 0.9 + random.random() * 0.2)
                elif random.random() > 0.985:
                    img = brightness(img, 0.9 + random.random() * 0.2)
                elif random.random() > 0.985:
                    img = contrast(img, 0.9 + random.random() * 0.2)

            if random.random() > 0.98:
                if random.random() > 0.985:
                    img2 = clahe(img2)
                elif random.random() > 0.985:
                    img2 = gauss_noise(img2)
                elif random.random() > 0.985:
                    img2 = cv2.blur(img2, (3, 3))
            elif random.random() > 0.98:
                if random.random() > 0.985:
                    img2 = saturation(img2, 0.9 + random.random() * 0.2)
                elif random.random() > 0.985:
                    img2 = brightness(img2, 0.9 + random.random() * 0.2)
                elif random.random() > 0.985:
                    img2 = contrast(img2, 0.9 + random.random() * 0.2)

            if random.random() > 0.983:
                el_det = self.elastic.to_deterministic()
                img = el_det.augment_image(img)

            if random.random() > 0.983:
                el_det = self.elastic.to_deterministic()
                img2 = el_det.augment_image(img2)

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        return img, img2, msk


def get_stratified_train_val_split() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get train/val split stratified by disaster name."""
    train_dirs = ['/local_storage/datasets/sgerard/xview2/no_overlap/train']

    all_files = []
    for d in train_dirs:
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'images', f))

    # Fixed stratified sample to split data into train/val
    disaster_names = list(
        map(lambda path: Path(path).name.split("_")[0], all_files))
    train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=23,
                                            stratify=disaster_names)
    return train_idxs, val_idxs, all_files


def get_train_val_datasets(seed: int,
                           input_shape: int,
                           low_aug: bool,
                           is_localization: bool,
                           loc_folder: str = None,
                           dilate: bool = None,
                           is_training: bool = None,
                           remove_events: str = "") -> Tuple[Dataset]:
    """Get train/val datasets.

        Args:
            input_shape : size of the input image. It is assumed that the input image is square.
            low_aug : indicates whether we're using low augmentation (for training) or high augmentation (for tuning).
            is_localization : indicates whether we're training a localization model or a classification model.
            loc_folder : must be output of pred_loc_val. 
            is_training : indicates whether we're in a 'train' or 'tune' setting. The aug parameters for each are slightly different.

        Returns:
            both train and val datasets in a tuple.
    """

    train_idxs, val_idxs, all_files = get_stratified_train_val_split()

    # Filter out events that should be removed. This has to be done after train/val splitting, so that the
    # train and validation set do not experience large changes from run to run, impacting comparability.

    remove_events = remove_events.split("#")
    # Empty string is substring of all strings, so all images would be removed
    if "" in remove_events:
        remove_events.remove("")

    if len(remove_events) > 0:
        for i, f in enumerate(all_files):
            for rm_event in remove_events:
                if f.find(rm_event) != -1:
                    if i in val_idxs:
                        val_idxs = val_idxs[val_idxs != i]
                    else:
                        train_idxs = train_idxs[train_idxs != i]
                    continue

    # Oversample images that contain buildings.
    # This should lead to roughly 50-50 distribution between images with and without buildings.
    if not is_localization:
        file_classes = []
        for fn in all_files:
            fl = np.zeros((4,), dtype=bool)
            msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
                              cv2.IMREAD_UNCHANGED)
            for c in range(1, 5):
                fl[c - 1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)

        new_train_idxs = []
        for i in train_idxs:
            new_train_idxs.append(i)
            if file_classes[i, 1:].max():
                new_train_idxs.append(i)
        train_idxs = np.asarray(new_train_idxs)

    if is_localization:
        data_train = LocalizationDataset(
            idxs=train_idxs, all_files=all_files, input_shape=input_shape, is_train=True, low_aug=low_aug)
        val_train = LocalizationDataset(
            idxs=val_idxs, all_files=all_files, input_shape=input_shape, is_train=False)
    else:
        if dilate is None:
            raise ValueError(
                "dilate must be specified for classification task")
        if is_training is None:
            raise ValueError(
                "training_task must be specified for classification task")
        training_task = "train" if is_training else "tune"
        data_train = ClassificationDataset(
            idxs=train_idxs, all_files=all_files, input_shape=input_shape, low_aug=low_aug, dilate=dilate, task=training_task)
        val_train = ClassificationDataset(
            idxs=val_idxs, all_files=all_files, input_shape=input_shape, low_aug=False, dilate=False, task="val", loc_folder=loc_folder)

    return data_train, val_train


def create_train_test_folder() -> None:
    """Create train/test folder with no data contamination  for xView2 dataset."""
    ###To be modified###
    train_dirs = ['/local_storage/datasets/sgerard/xview2/train',
                  '/local_storage/datasets/sgerard/xview2/tier3',
                  '/local_storage/datasets/sgerard/xview2/test']

    all_files = []
    for d in train_dirs:
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'images', f))

    train_disasters = ["lower-puna-volcano",
                       "palu-tsunami",
                       "mexico-earthquake",
                       "socal-fire",
                       "woolsey-fire",
                       "portugal-wildfire",
                       "pinery-bushfire",
                       "nepal-flooding",
                       "midwest-flooding",
                       "moore-tornado",
                       "joplin-tornado",
                       "hurricane-florence",
                       "hurricane-harvey",
                       "hurricane-michael"]

    test_disasters = ['tuscaloosa-tornado',
                      "guatemala-volcano",
                      "sunda-tsunami",
                      "santa-rosa-wildfire",
                      "hurricane-matthew"]

    train_files = list([f for f in all_files if f.split(
        '/')[-1].split('_')[0] in train_disasters])
    test_files = list([f for f in all_files if f.split('/')
                      [-1].split('_')[0] in test_disasters])
    # create train folder
    ###To be modified###
    train_folder = '/local_storage/datasets/sgerard/xview2/no_overlap/train'
    os.makedirs(train_folder, exist_ok="True")
    os.makedirs(path.join(train_folder, 'images'), exist_ok="True")
    os.makedirs(path.join(train_folder, 'masks'), exist_ok="True")

    for f in tqdm.tqdm(train_files):
        shutil.copy(f, path.join(train_folder, 'images'))
        shutil.copy(f.replace('_pre_disaster', '_post_disaster'),
                    path.join(train_folder, 'images'))
        # copy masks
        shutil.copy(f.replace('/images/', '/masks/'),
                    path.join(train_folder, 'masks'))
        shutil.copy(f.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
                    path.join(train_folder, 'masks'))

    # create test folder
    ###To be modified###
    test_folder = '/local_storage/datasets/sgerard/xview2/no_overlap/test'
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(path.join(test_folder, 'images'), exist_ok="True")
    os.makedirs(path.join(test_folder, 'masks'), exist_ok="True")
    os.makedirs(path.join(test_folder, 'labels'), exist_ok="True")
    os.makedirs(path.join(test_folder, 'targets'), exist_ok="True")

    for f in tqdm.tqdm(test_files):
        shutil.copy(f, path.join(test_folder, 'images'))
        shutil.copy(f.replace('_pre_disaster', '_post_disaster'),
                    path.join(test_folder, 'images'))

        # copy labels
        shutil.copy(f.replace('/images/', '/labels/').replace('.png', '.json'),
                    path.join(test_folder, 'labels'))
        shutil.copy(f.replace('/images/', '/labels/').replace('.png', '.json').replace('_pre_disaster', '_post_disaster'),
                    path.join(test_folder, 'labels'))

        # copy target files, as there is no /targets folder in tier3 we create it from /masks
        if "tier3" in f:
            # shutil.copy(f.replace('/images/', '/masks/'),
            #             path.join(test_folder, 'targets', f.split("/")[-1].replace('.png', '_target.png')))
            msk_pre = cv2.imread(f.replace('/images/', '/masks/'),
                                 cv2.IMREAD_UNCHANGED)
            # convert from 0-255 to 0-1
            msk_pre = msk_pre // 255
            cv2.imwrite(path.join(test_folder, 'targets', f.split(
                "/")[-1].replace('.png', '_target.png')), msk_pre)
            shutil.copy(f.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
                        path.join(test_folder, 'targets', f.split("/")[-1].replace('_pre_disaster', '_post_disaster').replace('.png', '_target.png')))

        else:
            shutil.copy(f.replace('/images/', '/targets/').replace('.png', '_target.png'),
                        path.join(test_folder, 'targets'))
            shutil.copy(f.replace('/images/', '/targets/').replace('_pre_disaster', '_post_disaster').replace('.png', '_target.png'),
                        path.join(test_folder, 'targets'))

    print("Done")


if __name__ == "__main__":
    train_data, val_data = get_train_val_datasets(
        1, (512, 512), False, False, None, False, is_training=True)

