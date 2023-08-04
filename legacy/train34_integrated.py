from utils import *
from datasets import get_train_val_datasets
from zoo.models import Res34_Unet_Double
import timeit
from losses import ComboLoss
from adamw import AdamW
import wandb
from apex import amp
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch import nn
import torch
import random
import numpy as np
from os import makedirs
import os
import cv2

from refactor_utils import get_parser, load_snapshot
from train_val_methods import evaluate_val_cls, train_epoch_cls

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


np.random.seed(1)

random.seed(1)


torch.manual_seed(1)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    input_shape = (608, 608)
    batch_size = 16
    val_batch_size = 8

    parser = get_parser()
    args = parser.parse_args()
    # We could imagine splitting off loc predictions from cls predictions and then evaluating cls only on areas
    # predicted to contain a building. However, this should be the same as
    # just computing the loss jointly on cls and loc in this case.
    args.validate_on_score = False

    np.random.seed(args.seed + 321)
    random.seed(args.seed + 321)
    torch.manual_seed(args.seed + 321)
    cudnn.benchmark = True

    models_folder = os.path.join(
        '/local_storage/users/paulbp/xview2/weights/', args.dir_prefix)
    makedirs(models_folder, exist_ok=True)

    # We set loc_folder to None, since it is only used to mask out non-building areas during validation.
    # Since we are still training the loc predictions, this doesn't make much sense here.
    # As long as we don't validate on score, this should not cause any errors.
    loc_folder = None

    snapshot_name = 'res34_cls2_{}_0'.format(args.seed)

    data_train, val_train = get_train_val_datasets(seed=args.seed, input_shape=input_shape, low_aug=args.low_aug,
                                                   is_localization=False, loc_folder=loc_folder, dilate=args.dilate_labels,
                                                   is_training=True,
                                                   remove_events=args.remove_events)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False,
                                   drop_last=True)
    val_data_loader = DataLoader(
        val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    model = Res34_Unet_Double().cuda()
    params = model.parameters()
    optimizer = AdamW(params, lr=0.0002, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180,
                                                     190], gamma=0.5)

    model = nn.DataParallel(model).cuda()

    wandb.init(project="xview2_no1_solution", tags=["resnet34", "train", "cls"], notes=f"seed: {args.seed}",
               dir="/local_storage/users/paulbp/xview2/logs/", config=args, group=args.wandb_group,
               job_type="train_cls")
    wandb.watch(model, log="all")

    seg_loss = ComboLoss({'dice': args.dice_weight_cls,
                         'focal': args.focal_weight_cls}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    best_score = -10e6
    torch.cuda.empty_cache()
    for epoch in range(40):
        train_epoch_cls(epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, args.use_tricks,
                        args.ce_weight, args.class_weights)
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            best_score = evaluate_val_cls(val_data_loader, best_score, model, snapshot_name, epoch, models_folder,
                                          args, seg_loss, ce_loss)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
