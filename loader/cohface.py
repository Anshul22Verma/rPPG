import collections
import cv2
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

random_state = 2023
random.seed(random_state)


def prep_splits(root_loc: str):  # -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hr_sheet = os.path.join(root_loc, "hr_info.csv")
    if not os.path.exists(hr_sheet):
        patients = [f for f in os.listdir(root_loc) if os.path.isdir(os.path.join(root_loc, f)) and f.isnumeric()]

        subject_ids = []
        avi_file = []
        # get subject ID for all the folders with HR dataset and add it to the df
        for dir in patients:
            dir = str(dir).replace("\\", "")
            sessions_dir = os.path.join(root_loc, str(dir))
            for session in os.listdir(sessions_dir):
                session_dir = os.path.join(sessions_dir, session)

                # get the path of the video
                avi_f = None
                hdf5_f = None
                for f in os.listdir(session_dir):
                    if ".avi" in f:
                        avi_f = os.path.join(session_dir, f)
                    if ".hdf5" in f:
                        hdf5_f = os.path.join(session_dir, f)
                avi_file.append(avi_f)
                subject_ids.append(dir)

                # read the hdf5
                # ##### -- PPG Signal (Can use this to estimate HR) -- ##### #
                # subject_hdf5 = read_hdf5(hdf5_f)
                # import matplotlib.pyplot as plt
                # print(np.array(subject_hdf5['pulse']))
                # print(np.array(subject_hdf5['time']))
                # plt.plot(np.array(subject_hdf5['time']), np.array(subject_hdf5['pulse']))
                # plt.show()
        df = pd.DataFrame()
        df["AVI File"] = avi_file
        df["Subject IDs"] = subject_ids
        df.to_csv(hr_sheet, index=False)

    df = pd.read_csv(hr_sheet)
    # create two splits use the existing train-test split in the dataset

    # RESULTS IN AN UNEVEN SPLIT
    # test_subject_ids, train_subject_ids = set(), set()

    # for d in ["all", "clean", "natural"]:
    #     protocols_dir = os.path.join(root_loc, "protocols", d)
    #     test_f = os.path.join(protocols_dir, "test.txt")
    #     test_f = open(test_f, 'r')
    #     l_ = test_f.readlines()
    #     for line in l_:
    #         test_subject_ids.add(line.split("/")[0])
    #
    #     train_f = os.path.join(protocols_dir, "train.txt")
    #     train_f = open(train_f, 'r')
    #     l_ = train_f.readlines()
    #     for line in l_:
    #         train_subject_ids.add(line.split("/")[0])
    #
    # test_subject_ids, train_subject_ids = list(test_subject_ids), list(train_subject_ids)
    # df_fold1 = df[df["Subject IDs"].isin([int(i) for i in train_subject_ids])]
    # df_fold2 = df[~df["Subject IDs"].isin([int(i) for i in train_subject_ids])]

    subjects = np.unique(df["Subject IDs"].values.tolist())
    fold_1 = random.sample(list(subjects), int(len(subjects) / 2))

    df_fold1 = df[df["Subject IDs"].isin(fold_1)]
    df_fold2 = df[~df["Subject IDs"].isin(fold_1)]
    # print(np.unique(df_fold1["Subject IDs"].values.tolist()))
    # print(np.unique(df_fold2["Subject IDs"].values.tolist()))
    return df_fold1, df_fold2


class COHFACEDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict, stl_map: dict = {}):
        super(COHFACEDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "COHFACE"
        self.stl_map = stl_map

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["AVI File"]

        if self.stl_map and "th" in self.stl_map and "group_clip_size" in self.stl_map and "frames_dim" in self.stl_map:
            clip_ = load_stl_map_sample(f_path=clip_path, th=self.stl_map["th"],
                                        group_clip_size=self.stl_map["group_clip_size"],
                                        frames_dim=self.stl_map["frames_dim"])
        else:
            clip_ = load_sample(f_path=clip_path)

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[self.dataset])

    def __len__(self):
        return len(self.df)


def get_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = COHFACEDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = COHFACEDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    df_fold_1, df_fold_2 = prep_splits(root_loc=r"D:\anshul\remoteHR\4081054\cohface")
    avi_file = df_fold_1["AVI File"].values.tolist()[1]
    vid = load_sample(avi_file, th=1)
    # 300 X (640 X 480 X 3) --> frames X H X W X C  --> # not loading over 300 frames only using first 300 frames
    # 300 X (224 X 224 X 3) --> frames X H X W X C

    # print(vid)
    print(len(vid))
    # 1207 frames  -> 60 seconds => 20 fps
