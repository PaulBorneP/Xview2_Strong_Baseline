"""
Investigate different threshold values for existing location predictions, to see which one maximizes the hard dice coeff
"""

from os import listdir
from pathlib import Path

import numpy as np
import pandas as pd
import torch

np.random.seed(1)
import random

random.seed(1)

from tqdm import tqdm
import timeit

from utils import *
from losses import soft_dice_loss

if __name__ == '__main__':

    pred_dir = "pred92_loc_tuned_0_train"
    #pred_dir = "low_aug_pred50_loc_tuned"
    #pred_dir = "pred34_loc_0"

    subset = "train"
    #subset = "test"

    test_path = Path(f'/local_storage/datasets/sgerard/xview2/{subset}/targets')
    pred_path = Path(f'/local_storage/users/paulbp/xview2/predictions/{pred_dir}')
    out_file = f"/local_storage/users/paulbp/xview2/predictions/{pred_dir}_thresholds.csv"

    lower = 0.4
    upper = 0.8
    steps = 10
    thresholds = [(lower + i * (upper - lower) / steps) for i in range(steps)]
    threshold_to_dices = {t: [] for t in thresholds}
    soft_dice_agg = []

    t0 = timeit.default_timer()

    for f in tqdm(sorted(listdir(test_path))):
        if '_pre_' in f:
            target_file = str(test_path / f)
            pred_file = str(pred_path / f).replace("_target", "_part1.png")

            if not Path(pred_file).exists():
                continue

            target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE) / 255

            soft_dice_agg.append(1 - soft_dice_loss(torch.Tensor(pred), torch.Tensor(target)))

            for threshold in thresholds:
                pred_thresholded = (pred >= threshold).astype(np.uint8)
                dice_coeff = dice(target, pred_thresholded)
                threshold_to_dices[threshold].append(dice_coeff)

    mean_dices = {k: np.mean(v) for (k, v) in threshold_to_dices.items()}
    pd.DataFrame(mean_dices.items(), columns=["threshold", "dice"]).to_csv(out_file, index=False)
    avg_soft_dice = np.mean(soft_dice_agg)
    print(avg_soft_dice)
    print(len(soft_dice_agg))

    with open(out_file.replace("thresholds.csv", "soft_dice.txt"), "w") as f:
        f.write(str(avg_soft_dice))

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
