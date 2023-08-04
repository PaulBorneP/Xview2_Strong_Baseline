SUBDIR="low_augment"
SEED=0

#python3 train50_loc_low_augment.py ${SEED} || exit 1
python3 tune50_loc_low_augment.py ${SEED} || exit 1

echo "predicting localization models on validation subset for seeds 0-2"
python3 predict_loc_val_args.py ${SEED} ${SUBDIR} || exit 1

echo "training seresnext50 classification model"
python3 train50_cls_cce_low_augment.py ${SEED} || exit 1
python3 tune50_cls_cce_low_augment.py ${SEED} || exit 1

python3 predict50_loc_args.py ${SEED} ${SUBDIR} || exit 1
python3 predict50cls_args.py ${SEED} ${SUBDIR} || exit 1
