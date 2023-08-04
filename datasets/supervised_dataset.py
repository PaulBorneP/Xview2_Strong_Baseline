from typing import Sequence, Tuple, Optional
import os

import lightning.pytorch as pl
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision

from datasets.base_dataset import LabeledDataset


class SLDataModule(pl.LightningDataModule):
    """Supervised learning datamodule."""

    def __init__(self,
                 data_dirs: Sequence[str],
                 target_disasters: Sequence[str],
                 labeled_transforms: torchvision.transforms.Compose,
                 labeled_batch_size: int,
                 num_workers: int,
                 combine_loaders_mode: str,
                 legacy: bool = False,
                 exclude_disasters: Optional[Sequence[str]] = [],

                 **kwargs
                 ):
        """Initialize supervised learning datamodule.

        Args:
            data_dirs: list of data directories
            target_disasters: list of target disasters, they are used for test
            labeled_transforms: transforms for labeled data
            labeled_batch_size: batch size for labeled data
            num_workers: number of workers for dataloaders
            combine_loaders_mode: mode for combining labeled and 
                unlabeled dataloaders.
            legacy: if True, use legacy code for idxs
            exclude_disasters: list of disasters to exclude from the training

            **kwargs: additional arguments for pl.LightningDataModule 
                (used by the trainer).
            """

        super().__init__(**kwargs)
        self.data_dirs = data_dirs
        self.target_disasters = target_disasters
        self.exclude_disasters = exclude_disasters
        self.labeled_transforms = labeled_transforms
        self.labeled_batch_size = labeled_batch_size
        self.num_workers = num_workers
        self.combine_loaders_mode = combine_loaders_mode
        self.all_files = self.get_all_files() if legacy else self.get_all_files()
        self.idxs = self.get_legacy_idxs() if legacy else self.get_idxs()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = LabeledDataset(
                self.all_files,
                self.idxs["train_lbl"],
                self.labeled_transforms,
                train=True)
            self.val_dataset = LabeledDataset(
                self.all_files,
                self.idxs["val_lbl"],
                self.labeled_transforms,
                train=False)

        if stage == "test":
            self.test_dataset = LabeledDataset(self.all_files,
                                               self.idxs["test"],
                                               self.labeled_transforms,
                                               train=False)

        if stage == "predict":
            self.pred_dataset = LabeledDataset(self.all_files,
                                               self.idxs["test"],
                                               self.labeled_transforms,
                                               train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.labeled_batch_size,
            shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.labeled_batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.labeled_batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.labeled_batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    def get_all_files(self) -> Sequence[str]:
        all_files = []
        for d in self.data_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if '_pre_disaster.png' in f:
                    all_files.append(os.path.join(d, 'images', f))
        return all_files

    @staticmethod
    def get_stratified_train_val_split(all_files: Sequence[str]) -> Tuple[Sequence[int],
                                                                          Sequence[int]]:

       # Fixed stratified sample to split data into train/val
        disaster_names = list(
            map(lambda path: pathlib.Path(path).name.split("_")[0], all_files))
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)),
                                                test_size=0.1,
                                                random_state=23,
                                                stratify=disaster_names)
        return train_idxs, val_idxs

    def get_idxs(self) -> Tuple[Sequence[int], Sequence[int]]:
        """split between train labeled, train unlabeled and test and data."""

        # for each disaster in target_disasters
        # take half of the data for "not-labeled" and half for "test"
        test_idxs = []
        for disaster in self.target_disasters:
            disaster_files = [f for f in self.all_files if disaster in f]
            disaster_idxs = [self.all_files.index(f) for f in disaster_files]
            test_idxs.extend(disaster_idxs)
        excluded_idxs = []
        for disaster in self.exclude_disasters:
            excluded_files = [f for f in self.all_files if disaster in f]
            excluded_idx = [self.all_files.index(f) for f in excluded_files]
            excluded_idxs.extend(excluded_idx)

        # if the idx is not in the train_not_labeled_idxs
        # or test_idxs it is in the train_labeled_idxs
        train_val_labeled_idxs = [i for i in range(
            len(self.all_files)) if i not in test_idxs and i not in excluded_idxs]

        all_train_files = list([self.all_files[i]
                               for i in train_val_labeled_idxs])
        # using temporary indexes because they are indexed on
        # all_train_files and not all_files
        train_tmp, val_tmp = self.get_stratified_train_val_split(
            all_train_files)
        train_labeled_idxs = list([train_val_labeled_idxs[i]
                                  for i in train_tmp])
        val_labeled_idxs = list([train_val_labeled_idxs[i] for i in val_tmp])

        return {"train_lbl": train_labeled_idxs,
                "val_lbl": val_labeled_idxs,
                "test": test_idxs}

    ###########################################
    ##### Used for legacy code comparison #####
    ###########################################

    @staticmethod
    def legacy_get_stratified_train_val_split():
        from os import listdir, path
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        """Get train/val split stratified by disaster name.
        """

        train_dirs = [
            '/local_storage/datasets/sgerard/xview2/no_overlap/train']
        all_files = []
        for d in train_dirs:
            for f in sorted(listdir(path.join(d, 'images'))):
                if '_pre_disaster.png' in f:
                    all_files.append(path.join(d, 'images', f))

        # Fixed stratified sample to split data into train/val
        disaster_names = list(
            map(lambda path: Path(path).name.split("_")[0], all_files))
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)),
                                                test_size=0.1,
                                                random_state=23,
                                                stratify=disaster_names)
        return train_idxs, val_idxs, all_files

    def get_legacy_idxs(self):
        """||!|| This method is used to get the same train/val split as the legacy code.
        this is really hacky and should only be used to check the new code
        work in a similar way as the legacy code."""

        print(" !!!!!!!!!!!! Using legacy idxs !!!!!!!!!!!!")

        legacy_train_idxs, legacy_val_idxs, legacy_all = self.legacy_get_stratified_train_val_split()
        legacy_train = list([legacy_all[i].split(
            "/")[-1].split("_p")[0] for i in legacy_train_idxs])
        legacy_val = list([legacy_all[i].split("/")[-1].split("_p")[0]
                          for i in legacy_val_idxs])
        # this is ugly, should be done in a better way
        train_idx = list([i for i in range(len(self.all_files)) if self.all_files[i].split(
            "/")[-1].split("_p")[0] in legacy_train])
        val_idx = list([i for i in range(len(self.all_files)) if self.all_files[i].split(
            "/")[-1].split("_p")[0] in legacy_val])
        assert len(train_idx) == len(legacy_train_idxs)
        assert len(val_idx) == len(legacy_val_idxs)
        not_train_lbl_idx = list(
            set(range(len(self.all_files))) - set(train_idx) - set(val_idx))
        # remove files from excluded disasters
        test_idx = list([i for i in not_train_lbl_idx if self.all_files[i].split(
            "/")[-1].split("_")[0] not in self.exclude_disasters])

        return {"train_lbl": train_idx,
                "val_lbl": val_idx,
                "test": test_idx, }


if __name__ == "__main__":
    import hydra
    import pytorch_lightning as pl
    import pathlib
    import numpy as np

    hydra.initialize(config_path="../conf")
    cfg = hydra.compose(config_name="supervised_config")
    pl.seed_everything(cfg.seed)
    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup("fit")
    data_module.setup("test")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(len(data_module.train_dataset))
    print(len(data_module.val_dataset))
    print(len(data_module.test_dataset))
    print(len(data_module.all_files))
