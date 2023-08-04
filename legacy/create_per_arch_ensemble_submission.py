from os import path, makedirs, listdir
import sys

sys.setrecursionlimit(10000)
from multiprocessing import Pool
import numpy as np

np.random.seed(1)
import random

random.seed(1)

from tqdm import tqdm
import timeit
import cv2

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

output_dir = "/local_storage/users/paulbp/xview2/predictions/winner/"

pred_folders_ = [output_dir + 'dpn92cls_cce_0_tuned', output_dir + 'dpn92cls_cce_1_tuned',
                 output_dir + 'dpn92cls_cce_2_tuned']  + [
                    output_dir + 'res50cls_cce_0_tuned', output_dir + 'res50cls_cce_1_tuned',
                    output_dir + 'res50cls_cce_2_tuned'] + [output_dir + 'se154cls_0_tuned',
                                                            output_dir + 'se154cls_1_tuned',
                                                            output_dir + 'se154cls_2_tuned']
pred_coefs = [1.0] * 3
loc_folders_ = [output_dir + 'pred92_loc_tuned', output_dir + 'pred50_loc_tuned',
                output_dir + 'pred154_loc']
loc_coefs = [1.0] * 1

sub_folder_ = output_dir + 'submission_arch_ensemble'
_thr = [0.38, 0.13, 0.14]


def process_image(id_and_filename):
    arch_id, img_filepath = id_and_filename
    arch_name = pred_folders_[(arch_id * 3)].split("/")[-1].split("cls")[0]
    sub_folder = path.join(sub_folder_, arch_name)
    pred_folders = pred_folders_[(arch_id * 3):((arch_id + 1) * 3)]
    loc_folders = [loc_folders_[arch_id]]

    preds = []
    _i = -1
    for d in pred_folders:
        _i += 1

        msk1 = cv2.imread(path.join(d, img_filepath), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(d, img_filepath.replace('_part1', '_part2')), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        preds.append(msk * pred_coefs[_i])
    preds = np.asarray(preds).astype('float').sum(axis=0) / np.sum(pred_coefs) / 255

    loc_preds = []
    _i = -1
    for d in loc_folders:
        _i += 1
        msk = cv2.imread(path.join(d, img_filepath), cv2.IMREAD_UNCHANGED)
        loc_preds.append(msk * loc_coefs[_i])
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1  # get 4-class ids per pixel
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | (
            (loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')

    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(
        path.join(sub_folder, '{0}'.format(img_filepath.replace('_pre_', '_localization_').replace('_part1.png', '_prediction.png'))),
        msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(
        path.join(sub_folder, '{0}'.format(img_filepath.replace('_pre_', '_damage_').replace('_part1.png', '_prediction.png'))),
        msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder_, exist_ok=True)

    all_files = []
    for f in sorted(listdir(pred_folders_[0])):
        if '_part1.png' in f:
            all_files.append(f)

    with Pool() as pool:
        # Create submission for each architecture separately
        for i in range(len(loc_folders_)):
            arch_name = pred_folders_[(i * 3)].split("/")[-1].split("cls")[0]
            sub_folder = path.join(sub_folder_, arch_name)
            makedirs(sub_folder, exist_ok=True)
            ids_and_filenames = list(zip([i] * len(all_files), all_files))
            with tqdm(total=len(ids_and_filenames), desc=f"Processing {arch_name}") as pbar:
                for _ in pool.imap_unordered(process_image, ids_and_filenames):
                    pbar.update(1)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
 