from datasets import get_train_val_datasets
from utils import *
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

    np.random.seed(args.seed + 357)
    random.seed(args.seed + 357)
    torch.manual_seed(args.seed + 357)
    cudnn.benchmark = True

    models_folder = os.path.join(
        '/local_storage/users/paulbp/xview2/weights', args.dir_prefix)
    loc_folder = os.path.join(
        '/local_storage/users/paulbp/xview2/predictions/', args.dir_prefix, "pred_loc_val")
    makedirs(models_folder, exist_ok=True)

    snapshot_name = 'res34_cls2_{}_tuned'.format(args.seed)

    data_train, val_train = get_train_val_datasets(seed=args.seed, input_shape=input_shape, low_aug=args.low_aug,
                                                   is_localization=False, loc_folder=loc_folder, dilate=args.dilate_labels,
                                                   is_training=False,
                                                   remove_events=args.remove_events)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False,
                                   drop_last=True)
    val_data_loader = DataLoader(
        val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    model = Res34_Unet_Double().cuda()
    params = model.parameters()
    optimizer = AdamW(params, lr=0.000008, weight_decay=1e-6)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[1, 2, 3, 4, 5, 7, 9, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110,
                                                     130, 150, 170, 180, 190], gamma=0.5)

    model = nn.DataParallel(model).cuda()

    snap_to_load = 'res34_cls2_{}_0_best'.format(args.seed)
    load_snapshot(model, snap_to_load, models_folder)

    wandb.init(project="xview2_no1_solution", tags=["resnet34", "cls", "tune"], notes=f"seed: {args.seed}",
               dir="/local_storage/users/paulbp/xview2/logs/", config=args, group=args.wandb_group,
               job_type="tune_cls")
    wandb.watch(model, log=None)

    seg_loss = ComboLoss({'dice': args.dice_weight_cls,
                         'focal': args.focal_weight_cls}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()

    best_score = -10e6
    torch.cuda.empty_cache()

    if args.debug:
        max_epochs = 1
    else:
        max_epochs = 3

    for epoch in range(max_epochs):
        train_epoch_cls(epoch, seg_loss, ce_loss, model, optimizer, scheduler, train_data_loader, args.use_tricks,
                        args.ce_weight, args.class_weights)
        torch.cuda.empty_cache()
        best_score = evaluate_val_cls(val_data_loader, best_score, model, snapshot_name, epoch, models_folder, args,
                                      seg_loss, ce_loss)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
