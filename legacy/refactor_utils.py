import argparse
import gc
from distutils import util
from os import path

import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--low_aug", type=lambda x: bool(util.strtobool(x)), default=False)
    parser.add_argument("--dilate_labels",
                        type=lambda x: bool(util.strtobool(x)), default=True)
    parser.add_argument("--validate_on_score", type=lambda x: bool(util.strtobool(x)), default=True,
                        help="Validate on competition score, as opposed to training loss")
    parser.add_argument("--focal_weight_loc", type=float, default=10.0)
    parser.add_argument("--dice_weight_loc", type=float, default=1.0)
    parser.add_argument("--focal_weight_cls", type=float, default=12.0)
    parser.add_argument("--dice_weight_cls", type=float, default=1.0)
    parser.add_argument("--class_weights", type=str, default="no1", choices=["no1", "equal", "distr", "distr_no_overlap"],
                        help="Choose class weights to weigh the classes separately in computing the loss. "
                             "'Equal' assigns equal weights, 'no1' uses the weights in the no.1 solution, "
                             "'distr' uses the normalized inverse of the class distribution in the training dataset.")
    parser.add_argument("--ce_weight", type=float, default=0.0,
                        help="Weight for CrossEntropy loss during classification training")
    parser.add_argument("--dir_prefix", type=str,
                        help="Prefix to use when creating the different subfolders, e.g. for weights, predictions etc.")
    parser.add_argument(
        "--use_tricks", type=lambda x: bool(util.strtobool(x)), default=True)
    parser.add_argument("--debug", type=lambda x: bool(util.strtobool(x)), default=False,
                        help="Only one epoch per subscript to make sure that everything is working as intended.")
    parser.add_argument("--wandb_group", type=str, default="",
                        help="Group argument to be set in wandb. "
                             "Used to identify all partial runs within one experiment.")
    parser.add_argument("--remove_events", type=str, default="", help="Names of event that should be left out of "
                                                                      "the train dataset for cls and loc. Substring"
                                                                      "matching is used on filenames to detect "
                                                                      "which images to leave out. Multiple events are "
                                                                      "separated with #.")
    return parser


def load_snapshot(model, snap_to_load, models_folder):
    print("=> loading checkpoint '{}'".format(snap_to_load))
    state_dict = torch.load(
        path.join(models_folder, snap_to_load), map_location='cpu')["state_dict"]
    if 'module.' in list(state_dict.keys())[0] and 'module.' not in list(model.state_dict().keys())[0]:
        loaded_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        print("modifying keys so that they match the current model, normal if you're using loc weights for cls")
    else:
        loaded_dict = state_dict

    # loaded_dict = checkpoint['state_dict']
    sd = model.state_dict().copy()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
        else:
            print("skipping key: {}".format(k))
    model.load_state_dict(sd)

    del loaded_dict
    del sd
    gc.collect()
    torch.cuda.empty_cache()


def get_class_weights(cw):
    if cw == "no1":
        class_weights = [0.05, 0.2, 0.8, 0.7, 0.4]
    elif cw == "distr":
        class_weights = [1/0.030342701529957234, 1/0.023289585196743044, 1/0.002574037714986485, 1/0.002682490082519425,
                         1/0.0017965885357082826]
        sum_of_weights = sum(class_weights)
        class_weights = [w/sum_of_weights for w in class_weights]
    elif cw == "distr_no_overlap":
        class_weights = [32.20398121025673, 41.516691841904844,
                         406.2242790072747, 319.5142994620793, 727.8449005124751]
        sum_of_weights = sum(class_weights)
        class_weights = [w/sum_of_weights for w in class_weights]
    elif cw == "equal":
        class_weights = [0.2] * 5
    else:
        raise ValueError(f"Not implemented for class weight choice: {cw}")

    return class_weights
