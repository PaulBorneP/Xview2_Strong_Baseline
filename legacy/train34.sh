#!/bin/bash

LOW_AUG=$1 # Default: True
WEIGHT_FOCAL_LOSS_LOC=$2 # Default: 10
WEIGHT_DICE_LOSS_LOC=$3 # Default: 1
WEIGHT_FOCAL_LOSS_CLS=$4 # Default: 12
WEIGHT_DICE_LOSS_CLS=$5 # Default: 1
DILATE=$6 # Default: False; Dilates building pixels in cls training set
SEED=$7 # Default: 23
DIR_PREFIX=$8 # Name of experiments directory on harddrive
USE_TRICKS=$9 # Default: False
CE_WEIGHT="${10}" # Default: 0; adds CE loss with specified weight to the other loss terms.
VALIDATE_ON_SCORE="${11}" # Default: True; Determine validation performance based on score instead of loss.
CLASS_WEIGHTS="${12}" # Default: no1
GROUP="${13}" # Name of experiment group in wandb, should be unique.
DEBUG="${14}" # Default: False; Use one epoch per script to verify the whole pipeline works.
REMOVE_EVENTS="${15}" # Event that should be removed from training and validation set, matches via substring
ARCH="resnet34"

echo "Training resnet34 localization model"
python3 train34_loc_parametrized.py --low_aug "${LOW_AUG}" --focal_weight_loc "${WEIGHT_FOCAL_LOSS_LOC}" --dice_weight_loc "${WEIGHT_DICE_LOSS_LOC}" --seed "${SEED}" --dir_prefix "${DIR_PREFIX}" --use_tricks "${USE_TRICKS}" --wandb_group "${GROUP}" --debug "${DEBUG}" --remove_events "${REMOVE_EVENTS}" || exit 1

echo "Predicting localization models on validation subset"
python3 predict_loc_val_single.py "${SEED}" "${DIR_PREFIX}" "${ARCH}" || exit 1

echo "Training resnet34 classification model"
python3 train34_cls_parametrized.py --low_aug "${LOW_AUG}" --focal_weight_cls "${WEIGHT_FOCAL_LOSS_CLS}" --dice_weight_cls "${WEIGHT_DICE_LOSS_CLS}" --dilate_labels "${DILATE}" --seed "${SEED}" --dir_prefix "${DIR_PREFIX}" --use_tricks "${USE_TRICKS}" --ce_weight "${CE_WEIGHT}" --validate_on_score "${VALIDATE_ON_SCORE}" --class_weights "${CLASS_WEIGHTS}"  --wandb_group "${GROUP}"  --debug "${DEBUG}" --remove_events "${REMOVE_EVENTS}" || exit 1

if [ "${USE_TRICKS}" = "True" ]; then
  echo "Tuning resnet34 classification model"
  python3 tune34_cls_parametrized.py --low_aug "${LOW_AUG}" --focal_weight_cls "${WEIGHT_FOCAL_LOSS_CLS}" --dice_weight_cls "${WEIGHT_DICE_LOSS_CLS}" --dilate_labels "${DILATE}" --seed "${SEED}" --dir_prefix "${DIR_PREFIX}" --use_tricks "${USE_TRICKS}" --ce_weight "${CE_WEIGHT}" --validate_on_score "${VALIDATE_ON_SCORE}" --class_weights "${CLASS_WEIGHTS}"  --wandb_group "${GROUP}"  --debug "${DEBUG}" --remove_events "${REMOVE_EVENTS}" || exit 1
fi

echo "Predicting localization on test set"
python3 predict34_loc_single.py "${SEED}" "${DIR_PREFIX}" || exit 1
echo "Predicting damage classes on "
python3 predict34cls.py "${SEED}" "${DIR_PREFIX}" "${USE_TRICKS}" || exit 1

echo "Creating submission."
python3 create_single_submission.py "${SEED}" "${DIR_PREFIX}" "${ARCH}" "${USE_TRICKS}" || exit 1
echo "Computing metrics"
SUBMISSION_DIR=/local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"/submission_resnet34_"${DIR_PREFIX}"_"${SEED}"
python3 xview2_metrics.py "${SUBMISSION_DIR}" /local_storage/datasets/sgerard/xview2/no_overlap/test/targets "${SUBMISSION_DIR}"/metrics.json "${GROUP}" both
