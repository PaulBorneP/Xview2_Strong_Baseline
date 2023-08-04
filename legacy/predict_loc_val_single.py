from refactor_utils import load_snapshot
from sklearn.model_selection import train_test_split
from utils import *
from zoo.models import SeResNext50_Unet_Loc, Res34_Unet_Loc
import timeit
from tqdm import tqdm
from torch.autograd import Variable
from torch.backends import cudnn
import torch
import random
import os
import sys
from os import path, makedirs, listdir

import numpy as np
from torch import nn

from datasets import get_stratified_train_val_split

np.random.seed(1)

random.seed(1)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])
    sub_folder = sys.argv[2]
    arch = sys.argv[3]

    test_dir = '/local_storage/datasets/sgerard/xview2/test/images'
    pred_folder = f'/local_storage/users/paulbp/xview2/predictions/{sub_folder}/pred_loc_val/'
    train_dirs = ['/local_storage/datasets/sgerard/xview2/train',
                  '/local_storage/datasets/sgerard/xview2/tier3']
    models_folder = f'/local_storage/users/paulbp/xview2/weights/{sub_folder}'

    makedirs(pred_folder, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    cudnn.benchmark = True

    models = []
    model_idxs = []

    tot_len = 0
    for seed in [seed]:
        train_idxs, val_idxs, all_files = get_stratified_train_val_split()
        tot_len += len(val_idxs)
        model_idxs.append(val_idxs)
        # model_idxs.append(val_idxs)
        # model_idxs.append(val_idxs)
        # model_idxs.append(val_idxs)

        if arch == "resnet50":

            snap_to_load = 'res50_loc_{}_0_best'.format(seed)

            model = SeResNext50_Unet_Loc().cuda()
            load_snapshot(model, snap_to_load, models_folder)

            model.eval()
            models.append(model)

        elif arch == "resnet34":

            snap_to_load = 'res34_loc_{}_1_best'.format(seed)

                
            model = Res34_Unet_Loc().cuda()

            model = nn.DataParallel(model).cuda()
            load_snapshot(model, snap_to_load, models_folder)

            model.eval()
            models.append(model)
        else:
            raise ValueError(f"Wrong architecture: {arch}")

    unique_idxs = np.unique(np.asarray(model_idxs))

    print(tot_len, len(unique_idxs))

    with torch.no_grad():
        for idx in tqdm(unique_idxs):
            fn = all_files[idx]

            f = fn.split('/')[-1]

            img = cv2.imread(fn, cv2.IMREAD_COLOR)
            img = normalize_image(img)

            inp = []
            inp.append(img)
            inp.append(img[::-1, ::-1, ...])
            inp = np.asarray(inp, dtype='float')
            inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
            inp = Variable(inp).cuda()

            pred = []
            _i = -1
            for model in models:
                _i += 1
                if idx not in model_idxs[_i]:
                    continue
                msk = model(inp)
                msk = torch.sigmoid(msk)
                msk = msk.cpu().numpy()

                pred.append(msk[0, ...])
                pred.append(msk[1, :, ::-1, ::-1])

            pred_full = np.asarray(pred).mean(axis=0)

            msk = pred_full * 255
            msk = msk.astype('uint8').transpose(1, 2, 0)
            cv2.imwrite(path.join(pred_folder, f.replace('.png', '_part1.png')), msk[..., 0],
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
