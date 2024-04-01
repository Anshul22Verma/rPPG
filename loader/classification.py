import datetime
import glob
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvideotransforms import video_transforms, volume_transforms
import sys

sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from utils.helper import read_hdf5, load_sample, get_transforms, load_stl_map_sample
from loader.vipl import prep_splits_V1, prep_splits_V2
from loader.manhob import prep_splits as prep_splits_manhob
from loader.cohface import prep_splits as prep_splits_cohface


random_state = 2023
random.seed(random_state)


def prep_split(root_loc):
    hr_sheet_1 = os.path.join(root_loc, "fold_1.csv")
    hr_sheet_2 = os.path.join(root_loc, "fold_2.csv")

    if not os.path.exists(hr_sheet_1) or not os.path.exists(hr_sheet_2):
        avi_files_fold_1 = []
        avi_files_fold_2 = []
        dataset_fold_1 = []
        dataset_fold_2 = []

        # VIPL-V1
        root_loc_ = r"D:\anshul\remoteHR\VIPL-HR-V1"
        df_fold_1, df_fold_2 = prep_splits_V1(root_loc_)
        avi_files_fold_1 += df_fold_1["AVI File"].values.tolist()
        dataset_fold_1 += ["VIPL-V1" for _ in range(len(df_fold_1))]
        avi_files_fold_2 += df_fold_2["AVI File"].values.tolist()
        dataset_fold_2 += ["VIPL-V1" for _ in range(len(df_fold_2))]

        # VIPL-V2
        root_loc_ = r"D:\anshul\remoteHR\VIPL-HR-V2"
        df_fold_1, df_fold_2 = prep_splits_V2(root_loc_)
        avi_files_fold_1 += df_fold_1["AVI File"].values.tolist()
        dataset_fold_1 += ["VIPL-V2" for _ in range(len(df_fold_1))]
        avi_files_fold_2 += df_fold_2["AVI File"].values.tolist()
        dataset_fold_2 += ["VIPL-V2" for _ in range(len(df_fold_2))]

        # COHFACE
        root_loc_ = r"D:\anshul\remoteHR\4081054\cohface"
        df_fold_1, df_fold_2 = prep_splits_cohface(root_loc_)
        avi_files_fold_1 += df_fold_1["AVI File"].values.tolist()
        dataset_fold_1 += ["COHFACE" for _ in range(len(df_fold_1))]
        avi_files_fold_2 += df_fold_2["AVI File"].values.tolist()
        dataset_fold_2 += ["COHFACE" for _ in range(len(df_fold_2))]

        # MANHOB
        root_loc_ = r"D:\anshul\remoteHR\mahnob"
        df_fold_1, df_fold_2 = prep_splits_manhob(root_loc_)
        avi_files_fold_1 += df_fold_1["AVI File"].values.tolist()
        dataset_fold_1 += ["MANHOB" for _ in range(len(df_fold_1))]
        avi_files_fold_2 += df_fold_2["AVI File"].values.tolist()
        dataset_fold_2 += ["MANHOB" for _ in range(len(df_fold_2))]

        # cassification initialization
        df_fold_1 = pd.DataFrame()
        df_fold_1["AVI File"] = avi_files_fold_1
        df_fold_1["Dataset"] = dataset_fold_1

        df_fold_2 = pd.DataFrame()
        df_fold_2["AVI File"] = avi_files_fold_2
        df_fold_2["Dataset"] = dataset_fold_2

        df_fold_1.to_csv(hr_sheet_1, index=False)
        df_fold_2.to_csv(hr_sheet_2, index=False)

    df_fold_1 = pd.read_csv(hr_sheet_1)
    df_fold_2 = pd.read_csv(hr_sheet_2)
    return df_fold_1, df_fold_2


class DatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict, stl_map: dict = {}, fps: dict = None):
        super(DatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.stl_map = stl_map
        self.fps = fps

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["AVI File"]
        dataset = row_["Dataset"]

        if self.stl_map and "th" in self.stl_map and "group_clip_size" in self.stl_map and "frames_dim" in self.stl_map:
            if self.fps:
                clip_ = load_stl_map_sample(f_path=clip_path, th=self.stl_map["th"],
                                            group_clip_size=self.stl_map["group_clip_size"],
                                            frames_dim=self.stl_map["frames_dim"], from_fps=self.fps[dataset],
                                            to_fps=self.fps["combined"])
            else:
                clip_ = load_stl_map_sample(f_path=clip_path, th=self.stl_map["th"],
                                            group_clip_size=self.stl_map["group_clip_size"],
                                            frames_dim=self.stl_map["frames_dim"])
        else:
            if self.fps:
                clip_ = load_sample(f_path=clip_path, th=10, from_fps=self.fps[dataset], to_fps=self.fps["combined"])
            else:
                clip_ = load_sample(f_path=clip_path, th=10)

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[dataset])

    def __len__(self):
        return len(self.df)


def get_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict, fps: dict = None):
    train_dataset = DatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = DatasetDBClassification(df=test_df, train=False, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return train_loader, test_loader


def get_stl_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict, stl_map: dict, fps: dict = None):
    train_dataset = DatasetDBClassification(df=train_df, train=True, encodings=encodings, stl_map=stl_map)
    test_dataset = DatasetDBClassification(df=test_df, train=False, encodings=encodings, stl_map=stl_map)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    root_loc = r"D:\anshul\remoteHR\DataBias\Classification"
    df_fold_1, df_fold_2 = prep_split(root_loc=root_loc)
    encodings = {
        "COHFACE": 0,
        "VIPL-V1": 1,
        "VIPL-V2": 2,
        "MANHOB": 3,
    }
    fps = {
        "COHFACE": 20,
        "VIPL-V1": 25,
        "VIPL-V2": 26,
        "MANHOB": 61,
        "combined": 20,
    }

    train_loader, test_loader = get_data_loaders(train_df=df_fold_1, test_df=df_fold_2, encodings=encodings)
