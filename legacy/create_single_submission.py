import sys
from os import path, makedirs, listdir

sys.setrecursionlimit(10000)
from multiprocessing import Pool
import numpy as np

np.random.seed(1)
import random

random.seed(1)

from tqdm import tqdm
import timeit
import cv2

# import seaborn as sns

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

seed = int(sys.argv[1])
dir_prefix = sys.argv[2]
architecture = sys.argv[3]
use_tricks = sys.argv[4] == "True"


output_dir = f"/local_storage/users/paulbp/xview2/predictions/{dir_prefix}/"

# test_dir = '/local_storage/datasets/sgerard/xview2/test/images'

if architecture == "resnet34":
    sub_folder = output_dir + f'submission_resnet34_{dir_prefix}_{seed}'
    pred_folders = [output_dir + f'res34cls2_{seed}_tuned']
    loc_folders = [output_dir + f'pred34_loc_{seed}']
# else:
#     sub_folder = output_dir + f'submission_resnet50_{dir_prefix}_{seed}'
#     pred_folders = [output_dir + f'res50cls_cce_{seed}_tuned']
#     loc_folders = [output_dir + 'pred50_loc_tuned']

pred_coefs = [1.0]
loc_coefs = [1.0]

_thr = [0.38, 0.13, 0.14]


def process_image(f):
    preds = []
    _i = -1
    for d in pred_folders:
        _i += 1
        msk1 = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(d, f.replace('_part1', '_part2')), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        preds.append(msk * pred_coefs[_i])
    preds = np.asarray(preds).astype('float').sum(axis=0) / np.sum(pred_coefs) / 255

    loc_preds = []
    _i = -1
    for d in loc_folders:
        _i += 1
        msk = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        loc_preds.append(msk * loc_coefs[_i])
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1  # get 4-class ids per pixel
    if use_tricks:
        msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | (
                (loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
    else:
        msk_loc = (loc_preds > _thr[0]).astype('uint8')
    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if use_tricks and (_msk.sum() > 0):
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    
    cv2.imwrite(
        path.join(sub_folder, '{0}'.format(f.replace('_pre_', '_localization_').replace('_part1.png', '_prediction.png'))),
        msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(
        path.join(sub_folder, '{0}'.format(f.replace('_pre_', '_damage_').replace('_part1.png', '_prediction.png'))),
        msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder, exist_ok=True)
    print(sub_folder)

    all_files = []
    for f in sorted(listdir(pred_folders[0])):
        if '_part1.png' in f:
            all_files.append(f)

    with Pool(processes=6) as p:
        max_ = len(all_files)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(process_image, all_files):
                pbar.update()
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

    for f in sorted(listdir(pred_folders[0])):
        if '_part1.png' in f:
            all_files.append(f)

    