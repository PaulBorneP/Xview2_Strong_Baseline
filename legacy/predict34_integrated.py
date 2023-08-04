from utils import *
from zoo.models import Res34_Unet_Double
import timeit
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import torch
from refactor_utils import load_snapshot
import random
import os
import sys
from os import path, makedirs, listdir
import skimage
import cv2

import numpy as np

np.random.seed(1)

random.seed(1)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def normalize_image(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = sys.argv[1]
    dir_prefix = sys.argv[2]
    OTSU = (sys.argv[4]=="True")

    test_dir = '/local_storage/datasets/sgerard/xview2/no_overlap/test/images'
    pred_folder_loc = f'/local_storage/users/paulbp/xview2/predictions/{dir_prefix}/pred34_loc_{seed}'
    pred_folder_cls = f'/local_storage/users/paulbp/xview2/predictions/{dir_prefix}/res34cls2_{seed}_tuned'
    models_folder = f'/local_storage/users/paulbp/xview2/weights/{dir_prefix}/'

    use_all_flips = True

    makedirs(pred_folder_loc, exist_ok=True)
    makedirs(pred_folder_cls, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    # cudnn.benchmark = True

    models = []

    for seed in [seed]:
        snap_to_load = 'res34_cls2_{}_0_best'.format(seed)
        model = Res34_Unet_Double().cuda()
        model = nn.DataParallel(model).cuda()
        load_snapshot(model, snap_to_load, models_folder)
        model.eval()
        models.append(model)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fn.replace(
                    '_pre_', '_post_'), cv2.IMREAD_COLOR)

                img = np.concatenate([img, img2], axis=2)
                img = normalize_image(img)

                inp = []
                inp.append(img)
                if use_all_flips:
                    inp.append(img[::-1, ...])
                    inp.append(img[:, ::-1, ...])
                    inp.append(img[::-1, ::-1, ...])
                inp = np.asarray(inp, dtype='float')
                inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                inp = Variable(inp).cuda()

                pred = []
                for model in models:

                    msk = model(inp)
                    msk = torch.sigmoid(msk)
                    msk = msk.cpu().numpy()

                    pred.append(msk[0, ...])
                    if use_all_flips:
                        pred.append(msk[1, :, ::-1, :])
                        pred.append(msk[2, :, :, ::-1])
                        pred.append(msk[3, :, ::-1, ::-1])

                pred_full = np.asarray(pred)

                pred_full = pred_full.mean(axis=0).transpose(1, 2, 0)
                msk = pred_full * 255
                msk = msk.astype('uint8')
                if OTSU:
                    loc_msk = np.zeros(msk.shape[:-1])
                    for i in range(msk.shape[2]):
                        thresh = skimage.filters.threshold_otsu(msk[:, :, i])
                        loc_msk = np.logical_or(loc_msk, msk[:, :, i] > thresh)
                    loc_msk = loc_msk*255
                    loc_msk = loc_msk.astype('uint8')

                else: 
                    loc_msk = msk[..., 0]

                cv2.imwrite(path.join(pred_folder_loc, f.replace('.png', '_part1.png')), loc_msk,
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])

                cv2.imwrite(path.join(pred_folder_cls, f.replace('.png', '_part1.png')), msk[..., :3],
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(pred_folder_cls, f.replace('.png', '_part2.png')), msk[..., 2:],
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
