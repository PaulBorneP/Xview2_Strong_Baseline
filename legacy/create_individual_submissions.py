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

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# output_dir = "predictions/"
output_dir = "/local_storage/users/paulbp/xview2/predictions/winner/"

use_154_model = False
sub_folder = output_dir + f'submission_indiv_single_rot_use154_{use_154_model}'

pred_folders = [output_dir + 'dpn92cls_cce_0_tuned', output_dir + 'dpn92cls_cce_1_tuned',
                output_dir + 'dpn92cls_cce_2_tuned'] + [output_dir + 'res34cls2_0_tuned',
                                                        output_dir + 'res34cls2_1_tuned',
                                                        output_dir + 'res34cls2_2_tuned'] + [
                   output_dir + 'res50cls_cce_0_tuned', output_dir + 'res50cls_cce_1_tuned',
                   output_dir + 'res50cls_cce_2_tuned']
pred_coefs = [1.0] * 9
loc_folders = [output_dir + 'pred92_loc_tuned', output_dir + 'pred34_loc', output_dir + 'pred50_loc_tuned']
loc_coefs = [1.0] * 3

_thr = [0.38, 0.13, 0.14]

if use_154_model:
    pred_folders += [output_dir + 'se154cls_0_tuned', output_dir + 'se154cls_1_tuned', output_dir + 'se154cls_2_tuned']
    loc_folders += [output_dir + 'pred154_loc']
    pred_coefs = [1.0] * 12
    loc_coefs = [1.0] * 4


def process_image(id_and_filename):
    model_id, img_filepath = id_and_filename

    model_name = pred_folders[model_id].split("/")[-1]
    sub_folder_model = path.join(sub_folder, model_name)
    pred_folder = pred_folders[model_id]
    loc_folder = loc_folders[int(model_id / 3)]

    msk1 = cv2.imread(path.join(pred_folder, img_filepath), cv2.IMREAD_UNCHANGED)
    msk2 = cv2.imread(path.join(pred_folder, img_filepath.replace('_part1', '_part2')), cv2.IMREAD_UNCHANGED)
    msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
    preds = msk.astype('float') / 255

    msk = cv2.imread(path.join(loc_folder, img_filepath), cv2.IMREAD_UNCHANGED)
    loc_preds = msk.astype('float') / 255

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1  # get 4-class ids per pixel
    # note that this line doesn't do anything for the integrated model because 
    # the loc prediction surrogate using cls predictions is already thresholded 
    # so loc_preds > _thr[0] is always true for non background pixels
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | (
            (loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')

    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(
        path.join(sub_folder_model,
                  '{0}'.format(img_filepath.replace('_pre_', '_localization_').replace('_part1.png', '_prediction.png'))),
        msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(
        path.join(sub_folder_model,
                  '{0}'.format(img_filepath.replace('_pre_', '_damage_').replace('_part1.png', '_prediction.png'))),
        msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder, exist_ok=True)

    all_files = []
    for f in sorted(listdir(pred_folders[0])):
        if '_part1.png' in f:
            all_files.append(f)

    with Pool(processes=6) as pool:

        for i in range(len(pred_folders)):
            model_name = pred_folders[i].split("/")[-1]
            print(model_name)
            sub_folder_model = path.join(sub_folder, model_name)

            makedirs(sub_folder_model, exist_ok=True)

            ids_and_filenames = list(zip([i] * len(all_files), all_files))
            max_=len(ids_and_filenames)
            with tqdm(total=max_) as pbar:
                for _ in pool.imap_unordered(process_image, ids_and_filenames):
                    pbar.update()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
 