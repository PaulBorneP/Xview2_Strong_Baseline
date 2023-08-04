from os import path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import wandb
from apex import amp
from tqdm import tqdm

from losses import dice_round
from refactor_utils import get_class_weights
from utils import dice, AverageMeter


def train_epoch_cls(current_epoch: int,
                    seg_loss: nn.Module,
                    ce_loss: nn.Module,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler,
                    train_data_loader: DataLoader,
                    use_tricks: bool = True,
                    ce_weight: float = 0.0,
                    class_weights_: str = "no1") -> None:
    """
    Trains model for one epoch.

    Args:
        seg_loss : segmentation loss
        ce_loss : cross entropy loss
        model : model to train
        ce_weight : weight of cross entropy loss
        class_weights_ : Choose class weights to weight the classes separately in computing the loss. 
            'Equal' assigns equal weights, 
            'no1' uses the weights in the no.1 solution, 
            'distr' uses the normalized inverse of the class distribution in the training dataset
    """

    losses = AverageMeter()
    losses1 = AverageMeter()

    dices = AverageMeter()

    class_weights = get_class_weights(class_weights_)

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        out = model(imgs)

        loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
        loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
        loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
        loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
        loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])

        ce_loss_val = torch.Tensor([0.0]).cuda()
        if ce_weight != 0.0:
            class_targets = torch.argmax(msks, dim=1)
            ce_loss_val = ce_loss(out, class_targets) * ce_weight

        seg_loss_val = class_weights[0] * loss0 + class_weights[1] * loss1 + class_weights[2] * loss2 + \
            class_weights[3] * loss3 + class_weights[4] * loss4
        # Catch error that happens if focal and dice loss have weights 0
        seg_loss_val = torch.Tensor([seg_loss_val]).cuda() if (
            type(seg_loss_val) == float) else seg_loss_val
        loss = seg_loss_val + ce_loss_val

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(seg_loss_val.item(), imgs.size(0))
        losses1.update(ce_loss_val.item(), imgs.size(0))  # loss5

        dices.update(dice_sc, imgs.size(0))

        wandb.log({"train_loss_step": losses.val,
                  "train_dice_step": dices.val})
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss CE {loss1.val:.4f} "
            "({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)
    wandb.log({"train_loss_epoch": losses.avg, "train_dice_epoch": dices.avg})
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; loss2 {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
        current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))


def validate_cls(model: nn.Module,
                 data_loader: DataLoader,
                 validate_on_score: bool = True,
                 args: Dict = None,
                 seg_loss: nn.Module = None,
                 ce_loss: nn.Module = None) -> float:
    """Validate the classification model using the loss or the competition metric.

    Args:
        model: model to validate
        validate_on_score : Whether we validate on the competition score (mix of F1 and dice) or the loss
        args : arguments used to call the training script (see train34.sh)
        seg_loss : segmentation loss
        ce_loss : cross entropy loss

    Returns:
        float: validation score (-loss or competition metric)

    """
    dices0 = []
    loss_vals = []

    tp = np.zeros((4,))
    fp = np.zeros((4,))
    fn = np.zeros((4,))

    _thr = 0.3

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            lbl_msk = sample["lbl_msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            msk_loc = sample["msk_loc"].numpy() * 1
            out = model(imgs)

            msk_loc_pred = msk_loc
            msk_pred = torch.sigmoid(out).cpu().numpy()
            msk_damage_pred = msk_pred[:, 1:, ...]

            if validate_on_score:
                for j in range(msks.shape[0]):
                    dice_val = dice(msks[j, 0], msk_loc_pred[j] > _thr)
                    dices0.append(dice_val)
                    wandb.log({"val_dice_step": dice_val})

                    targ = lbl_msk[j][msks[j, 0] > 0]
                    pred = msk_damage_pred[j].argmax(axis=0)
                    pred = pred * (msk_loc_pred[j] > _thr)
                    pred = pred[msks[j, 0] > 0]
                    for c in range(4):
                        tp[c] += np.logical_and(pred == c, targ == c).sum()
                        fn[c] += np.logical_and(pred != c, targ == c).sum()
                        fp[c] += np.logical_and(pred == c, targ != c).sum()

            # Validate on loss function, don't filter by loc predictions of loc model
            else:
                ce_loss_val = 0.0
                if args.ce_weight != 0.0:
                    ce_loss_val = ce_loss(torch.Tensor(
                        msk_damage_pred).cuda(), torch.Tensor(lbl_msk).cuda().long())

                class_weights = get_class_weights(args.class_weights)

                msks = torch.Tensor(msks).cuda()
                loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
                loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
                loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])
                loss3 = seg_loss(out[:, 3, ...], msks[:, 3, ...])
                loss4 = seg_loss(out[:, 4, ...], msks[:, 4, ...])

                seg_loss_val = class_weights[0] * loss0 + class_weights[1] * loss1 + class_weights[2] * loss2 + \
                    class_weights[3] * loss3 + class_weights[4] * loss4
                loss = seg_loss_val + args.ce_weight * ce_loss_val
                loss_vals.append(loss.item())

    if validate_on_score:
        d0 = np.mean(dices0)

        f1_sc = np.zeros((4,))
        for c in range(4):
            f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

        f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

        sc = 0.3 * d0 + 0.7 * f1
        print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, d0, f1,
                                                                                               f1_sc[0],
                                                                                               f1_sc[1],
                                                                                               f1_sc[2],
                                                                                               f1_sc[3]))
        wandb.log({"val_score_epoch": sc,
                  "val_dice_epoch": d0, "val_f1_epoch": f1})
        return sc
    else:
        loss_mean = np.mean(loss_vals)
        wandb.log({"val_loss_epoch": loss_mean})
        return -loss_mean


def evaluate_val_cls(data_val: DataLoader,
                     best_score: float,
                     model: nn.Module,
                     snapshot_name: str,
                     current_epoch: int,
                     models_folder: str,
                     args: Dict,
                     seg_loss: nn.Module,
                     ce_loss: nn.Module) -> float:
    """Evaluate the classification model on the validation set for one epoch and saves best model so far.

        Args:
            data_val : validation data loader
            best_score : initial value for best score on validation set (usually very small)
            model : model to evaluate
            snapshot_name : name of the snapshot from the last best model
            current_epoch : current epoch
            models_folder : folder to save best model
            args : arguments used to call the training script (see train34.sh)
            seg_loss : segmentation loss
            ce_loss : cross entropy loss
        Returns:
            best_score : best score on validation set so far (-loss or competition metric)
    """

    model = model.eval()

    d = validate_cls(model, data_loader=data_val, validate_on_score=args.validate_on_score, args=args,
                     seg_loss=seg_loss,
                     ce_loss=ce_loss)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


######################

def train_epoch_loc(current_epoch: int,
                    seg_loss: nn.Module,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler,
                    train_data_loader: DataLoader) -> None:
    """
    Trains localization model for one epoch.

    Args:
        seg_loss : segmentation loss
    """

    losses = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)

        out = model(imgs)

        loss = seg_loss(out, msks)

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))
        dices.update(dice_sc, imgs.size(0))
        wandb.log({"train_loss_step": losses.val,
                  "train_dice_step": dices.val})

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    wandb.log({"train_loss_epoch": losses.avg, "train_dice_epoch": dices.avg})
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Dice {dice.avg:.4f}".format(
        current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))


def validate_loc(model: nn.Module, data_loader: DataLoader) -> float:
    dices0 = []

    _thr = 0.5

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)

            out = model(imgs)

            msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()

            for j in range(msks.shape[0]):
                dice_val = dice(msks[j, 0], msk_pred[j] > _thr)
                dices0.append(dice_val)
                wandb.log({"val_dice_step": dice_val})

    d0 = np.mean(dices0)
    wandb.log({"val_dice_epoch": d0})
    print("Val Dice: {}".format(d0))
    return d0


def evaluate_val_loc(data_val: DataLoader,
                     best_score: float,
                     model: nn.Module,
                     snapshot_name: str,
                     current_epoch: int,
                     models_folder: str) -> float:
    """ Evaluates the localization model on validation set for one epoch and saves best model so far.

        Args:
            data_val : validation data loader
            best_score : initial value for best score on validation set (usually very small)
            model : model to evaluate
            snapshot_name : name of the snapshot from the last best model
            current_epoch : current epoch
            models_folder : folder to save best model

        Returns:
            best_score : best score on validation set  so far (dice)
    """

    model = model.eval()
    d = validate_loc(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score
