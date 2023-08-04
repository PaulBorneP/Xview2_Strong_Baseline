
from pathlib import Path
from typing import Sequence, Tuple


import json
import numpy as np
import cv2
import torchmetrics
import torch
import tqdm
from shapely.wkt import loads
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xview2_metrics import PathHandler
from create_masks import mask_for_polygon


class ClsMetric:
    """Calculate the F1 score for the classification task on the whole images and not just the buildings."""
    def __init__(self, pred_dir: str, target_dir: str) -> None:
        self.f1 = torchmetrics.F1Score(task="multiclass",num_classes=5, average='macro')
        self.channelwise_f1 = torchmetrics.F1Score(task="multiclass",num_classes=5, average='none')
        self.pred_dir = pred_dir
        self.target_dir = target_dir
        self.path_handlers = self.get_path_handlers()
    

    def calculate_all_images(self) -> None:
        for path_handler in tqdm.tqdm(self.path_handlers):
            _,dp, _, dt = path_handler.load_images()
            pred = torch.Tensor(dp).flatten()
            target = torch.Tensor(dt).to(torch.int).flatten()
            self.f1(pred, target)
            self.channelwise_f1(pred, target)
    

    def compute(self) -> float:
        return self.f1.compute(), self.channelwise_f1.compute()
    
    def reset(self) -> None:
        self.f1.reset()
        self.channelwise_f1.reset()

    def get_path_handlers(self):
        print(f"Searching for PNG files in directory: {self.target_dir}")
        path_handlers = []
        for path in self.target_dir.glob('*.png'):
            disaster_name, img_id, pre_post, _, target = path.name.rstrip(
                '.png').split('_')
            assert pre_post in [
                'pre', 'post'], f"target filenames must have 'pre' or 'post' in filename, got {path}"
            assert target == 'target', f"{target} should equal 'target' when getting path handlers"
            if pre_post == 'pre':
                path_handlers.append(PathHandler(
                    self.pred_dir, self.target_dir, img_id, disaster_name))
        print(f"Done: {len(path_handlers)} paths found")
        return path_handlers


    @classmethod
    def save_metrics(cls,pred_root: Path, target_dir: Path,)->None:
        metrics=[]
        for pred_dir in pred_root.glob("*seed*"):
            # print("Processing: ", pred_dir)
            if "distr_crop_no_ov" not in pred_dir.name:
                continue
            # iterate over all the directories in the predictions folder that have the word "submission" in them
            elif len(list(pred_dir.glob('*submission*')))!=1:
                print(f"Skipping {pred_dir} because it doesn't have exactly one submission folder but {len(list(pred_dir.glob('*submission*')))}")
            
            else:
                submission_dir = list(pred_dir.glob('*submission*'))[0]
                loc_metric = cls(submission_dir, target_dir)
                loc_metric.calculate_all_images()
                overall_F1,F1s = loc_metric.compute()
                metric = {"name" :f"{pred_dir.name}", "overall_F1":overall_F1.item()}
                for i, f1 in enumerate(F1s):
                    metric[f"F1_{i}"] = f1.item()
                print(metric)
                metrics.append(metric)
                loc_metric.reset()
        #convert to a dataframe and save to csv
        df = pd.DataFrame(metrics)
        df.to_csv(pred_root/'F1_no_overlap.csv')



if __name__ == '__main__':

    target_dir = Path('/local_storage/datasets/sgerard/xview2/no_overlap/test/targets/')
    pred_root = Path('/local_storage/users/paulbp/xview2/predictions')

    ClsMetric.save_metrics(pred_root,target_dir)
    df = pd.read_csv(pred_root/'F1_no_overlap.csv')
    print(df)


    # @staticmethod
    # def get_masks(json_path,min_area=100):
    #     damage_dict = {
    #                 "no-damage": 1,
    #                 "minor-damage": 2,
    #                 "major-damage": 3,
    #                 "destroyed": 4,
    #                 "un-classified": 1 # ?
    #             }

    #     json_file = json.load(open(json_path))
    #     msk_damage = np.zeros((1024, 1024), dtype='uint8')

    #     for feat in  json_file['features']['xy']:
    #         poly = loads(feat['wkt'])
    #         if poly.area > min_area:
    #             subtype = feat['properties']['subtype']
    #             _msk = mask_for_polygon(poly)
    #             msk_damage[_msk > 0] = damage_dict[subtype]
    #         else:
    #             continue
    #     return msk_damage 


        # @classmethod
    # def plot_building_size_importance(cls,pred_root: Path, target_dir: Path,min_area_list:Sequence[float]) -> Tuple[Sequence[float],Sequence[float]]:
    #     """" Plot the F1 score (total and channelwise) for different min_area values."""
    #     score_list = []
    #     for min_area in min_area_list:
    #         for pred_dir in pred_root.glob('*seed*'):
    #             print("Processing: ", pred_dir)
    #             # iterate over all the directories in the predictions folder that have the word "submission" in them
    #             if len(list(pred_dir.glob('*submission*')))!=1:
    #                 print(f"Skipping {pred_dir} because it doesn't have exactly one submission folder but {len(list(pred_dir.glob('*submission*')))}")
    #                 continue
    #             elif "no_overlap" not in pred_dir.name:
    #                 continue
    #             else:
    #                 submission_dir = list(pred_dir.glob('*submission*'))[0]
    #                 loc_metric = cls(submission_dir, target_dir)
    #                 loc_metric.calculate_all_images_masked(min_area)
    #                 f1, channelwise_f1 = loc_metric.compute()
    #                 score_list.append({"name" :f"{pred_dir.name}","min_area": min_area, "f1":f1, "channelwise_f1":channelwise_f1})
    #                 loc_metric.reset()
        
        # plt.figure(figsize=(10, 5))
        # #plot the total F1 score for different min_area values asa a lineplot
        # df = pd.DataFrame(score_list)
        # # save the dataframe to csv
        # df.to_csv(pred_root/'F1__min_area.csv')
        # df["arch"] = df["name"].apply(lambda x: x.split("_seed")[0])
        # sns.lineplot(x="min_area", y="f1", hue="arch", data=df)
        # plt.savefig(f'/Midgard/home/paulbp/plots_distribs/F1_overallmin_area.png')
        # for i in range(4):
        #     sns.lineplot(x="min_area", y=f"channelwise_f1_{i}", hue="arch", data=df)
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #     plt.savefig(f'/Midgard/home/paulbp/plots_distribs/F1_{i}_min_area.png')                                                                                                                                                                                           
        # return score_list
        # def calculate_all_images_masked(self,min_area:float) -> None:
    #     for path_handler in tqdm.tqdm(self.path_handlers):
    #         _,dp, _, dt = path_handler.load_images()
    #         json_path = Path(str(path_handler.dt).replace('targets','labels').replace('_target.png','.json'))
    #         mask = self.get_masks(json_path,min_area)
    #         # skip image if mask is empty
    #         if mask.sum() == 0:
    #             continue
    #         pred = torch.Tensor(dp).flatten()
    #         target = torch.Tensor(dt).to(torch.int).flatten()
    #         pred = pred[mask.flatten() > 0]
    #         target = target[mask.flatten() > 0]
    #         self.f1(pred, target)
    #         self.channelwise_f1(pred, target)