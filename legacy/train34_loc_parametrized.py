from utils import *
from datasets import get_train_val_datasets
from zoo.models import Res34_Unet_Loc
import timeit
import wandb
from losses import ComboLoss
from adamw import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch import nn
import torch
import random
import numpy as np
from os import makedirs
import os

from refactor_utils import get_parser
from train_val_methods import evaluate_val_loc, train_epoch_loc

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

    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed + 545)
    random.seed(args.seed + 454)
    torch.manual_seed(args.seed + 545)
    cudnn.benchmark = True

    batch_size = 16
    val_batch_size = 8

    models_folder = os.path.join('/local_storage/users/paulbp/xview2/weights', args.dir_prefix)
    makedirs(models_folder, exist_ok=True)

    snapshot_name = 'res34_loc_{}_1'.format(args.seed)

    if args.use_tricks:
        input_shape = (736, 736)
    else:
        input_shape = (608, 608)

    data_train, val_train = get_train_val_datasets(seed=args.seed, input_shape=input_shape, low_aug=args.low_aug,
                                                   is_localization=True, loc_folder=None, dilate=True, is_training=True,
                                                   remove_events=args.remove_events)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False,
                                   drop_last=True)
    val_data_loader = DataLoader(
        val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)

    model = Res34_Unet_Loc()
    params = model.parameters()
    optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[5, 11, 17, 25, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180,
                                                     190], gamma=0.5)

    model = nn.DataParallel(model).cuda()

    wandb.init(project="xview2_no1_solution", tags=["resnet34", "train", "loc"], notes=f"seed: {args.seed}",
               dir="/local_storage/users/paulbp/xview2/logs/", config=args, group=args.wandb_group,
               job_type="train_loc")
    wandb.watch(model, log=None)

    # Default: 1:10
    seg_loss = ComboLoss({'dice': args.dice_weight_loc,
                         'focal': args.focal_weight_loc}, per_image=False).cuda()

    best_score = -10e6
    _cnt = -1
    torch.cuda.empty_cache()

    if args.debug:
        max_epochs = 1
    else:
        max_epochs = 55
    for epoch in range(max_epochs):
        train_epoch_loc(epoch, seg_loss, model, optimizer,
                        scheduler, train_data_loader)
        if epoch % 2 == 0:
            _cnt += 1
            torch.cuda.empty_cache()
            best_score = evaluate_val_loc(
                val_data_loader, best_score, model, snapshot_name, epoch, models_folder)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
