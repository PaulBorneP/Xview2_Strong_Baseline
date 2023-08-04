#!/bin/bash

SEED=$1 # Default: 23
DIR_PREFIX=$2 # Name of experiments directory on harddrive
USE_TRICKS=$3 # Default: False
GROUP=$4 # Name of experiment group in wandb, should be unique.
PREFIX=$5 # Prefix of classification (not tuned) model to use for classification
ARCH="resnet34"

mkdir -p /local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"

cp -r /local_storage/users/paulbp/xview2/predictions/"${PREFIX}"/pred34_loc_"${SEED}" /local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"/pred34_loc_"${SEED}"
cp -r /local_storage/users/paulbp/xview2/predictions/"${PREFIX}"/res34cls2_"${SEED}"_tuned /local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"/res34cls2_"${SEED}"_tuned

echo "Creating submission."
python3 create_single_submission.py "${SEED}" "${DIR_PREFIX}" "${ARCH}" "${USE_TRICKS}" || exit 1

echo "Computing metrics"
SUBMISSION_DIR=/local_storage/users/paulbp/xview2/predictions/"${DIR_PREFIX}"/submission_resnet34_"${DIR_PREFIX}"_"${SEED}"
python3 xview2_metrics.py "${SUBMISSION_DIR}" /local_storage/datasets/sgerard/xview2/no_overlap/test/targets "${SUBMISSION_DIR}"/metrics.json "${GROUP}" full
