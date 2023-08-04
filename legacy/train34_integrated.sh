#!/bin/bash

LOW_AUG=$1
WEIGHT_FOCAL_LOSS_CLS=$2
WEIGHT_DICE_LOSS_CLS=$3
DILATE=$4
SEED=$5
DIR_PREFIX=$6
USE_TRICKS=$7
CE_WEIGHT="${8}"
VALIDATE_ON_SCORE="${9}"
CLASS_WEIGHTS="${10}"
GROUP="${11}" # Name of experiment group in wandb, should be unique.
REMOVE_EVENTS="${12}" # Event that should be removed from training and validation set, matches via substring
OTSU="False"
ARCH="resnet34"

echo "Training integrated resnet34 model"
python3 train34_integrated.py --low_aug "${LOW_AUG}" --focal_weight_cls "${WEIGHT_FOCAL_LOSS_CLS}" --dice_weight_cls "${WEIGHT_DICE_LOSS_CLS}" --dilate_labels "${DILATE}" --seed "${SEED}" --dir_prefix "${DIR_PREFIX}" --use_tricks "${USE_TRICKS}" --ce_weight "${CE_WEIGHT}" --validate_on_score "${VALIDATE_ON_SCORE}" --class_weights "${CLASS_WEIGHTS}" --wandb_group "${GROUP}" --remove_events "${REMOVE_EVENTS}" || exit 1

echo "Predicting for integrated resnet34 model"
python3 predict34_integrated.py "${SEED}" "${DIR_PREFIX}" "${USE_TRICKS}" "${OTSU}" || exit 1

echo "Creating submission."
python3 create_single_submission.py "${SEED}" "${DIR_PREFIX}" "${ARCH}" "${USE_TRICKS}" || exit 1

echo "Computing metrics"
SUBMISSION_DIR=/local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"/submission_resnet34_"${DIR_PREFIX}"_"${SEED}"
python3 xview2_metrics.py "${SUBMISSION_DIR}" /local_storage/datasets/sgerard/xview2/no_overlap/test/targets "${SUBMISSION_DIR}"/metrics.json "${GROUP}" full
