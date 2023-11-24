import collections
import datetime
import glob
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvideotransforms import video_transforms, volume_transforms
import sys

sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from utils.helper import read_hdf5, load_sample, get_transforms

random_state = 2023
random.seed(random_state)


def prep_splits_V2(root_loc: str):  # -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hr_sheet = os.path.join(root_loc, "hr_info.csv")

    root_train_loc = os.path.join(root_loc, "train")
    if not os.path.exists(hr_sheet):
        patients = [f for f in os.listdir(root_train_loc) if os.path.isdir(os.path.join(root_train_loc, f)) and f.isnumeric()]

        subject_ids = []
        avi_files = []
        meanHR = []
        for p in patients:
            patient_dir = os.path.join(root_train_loc, p)

            gt_csv = os.path.join(root_train_loc, p, "gt.csv")
            gt_df = pd.read_csv(gt_csv)

            # print(gt_df)
            # print(gt_df.columns)
            # print(gt_df["video1 "].values.tolist()[0])
            # find all the AVI files
            for f in os.listdir(patient_dir):
                if ".avi" in f:
                    subject_ids.append(p)
                    avi_files.append(os.path.join(patient_dir, f))
                    match f.split(".")[0]:
                        case "video1":
                            meanHR.append(gt_df["video1 "].values.tolist()[0])
                        case "video2":
                            meanHR.append(gt_df["video2 "].values.tolist()[0])
                        case "video3":
                            meanHR.append(gt_df["video3 "].values.tolist()[0])
                        case "video4":
                            meanHR.append(gt_df["video4"].values.tolist()[0])
                        case "video5":
                            meanHR.append(gt_df[" video5"].values.tolist()[0])
        df = pd.DataFrame()
        df["Subject IDs"] = subject_ids
        df["AVI File"] = avi_files
        df["meanHR"] = meanHR
        df.to_csv(hr_sheet, index=False)

    df = pd.read_csv(hr_sheet)
    subjects = np.unique(df["Subject IDs"].values.tolist())
    fold_1 = random.sample(list(subjects), int(len(subjects)/2))
    df_fold_1 = df[df["Subject IDs"].isin(fold_1)]
    df_fold_2 = df[~df["Subject IDs"].isin(fold_1)]

    print(len(df_fold_1))
    print(len(df_fold_2))
    # print(np.unique(df_fold1["Subject IDs"].values.tolist()))
    # print(np.unique(df_fold2["Subject IDs"].values.tolist()))
    return df_fold_1, df_fold_2


class VIPLV2HRDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict):
        super(VIPLV2HRDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "VIPL-V2"

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["AVI File"]

        clip_ = load_sample(f_path=clip_path)

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[self.dataset])

    def __len__(self):
        return len(self.df)


def get_v2_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = VIPLV2HRDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = VIPLV2HRDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def prep_splits_V1(root_loc: str):
    hr_sheet = os.path.join(root_loc, "hr_info.csv")

    root_train_loc = os.path.join(root_loc, "data")
    if not os.path.exists(hr_sheet):
        subjects = [f for f in os.listdir(root_train_loc) if os.path.isdir(os.path.join(root_train_loc, f)) and ".zip" not in f]
        subject_ids = []
        meanHR = []
        minHR = []
        maxHR = []
        avi_files = []

        for s in subjects:
            for ver in os.listdir(os.path.join(root_train_loc, s)):
                for source_ in os.listdir(os.path.join(root_train_loc, s, ver)):
                    for f in os.listdir(os.path.join(root_train_loc, s, ver, source_)):
                        if ".avi" in f:
                            avi_files.append(os.path.join(root_train_loc, s, ver, source_, f))
                            subject_ids.append(s)

                        if f == "gt_HR.csv":
                            hr_csv = pd.read_csv(os.path.join(root_train_loc, s, ver, source_, f))
                            hr = hr_csv["HR"].values.tolist()

                            meanHR.append(round(sum(hr) / len(hr), 2))
                            minHR.append(min(hr))
                            maxHR.append(max(hr))

        df = pd.DataFrame()
        df["Subject IDs"] = subject_ids
        df["AVI File"] = avi_files
        df["meanHR"] = meanHR
        df["minHR"] = minHR
        df["maxHR"] = maxHR
        df.to_csv(hr_sheet, index=False)

    df = pd.read_csv(hr_sheet)
    subjects = np.unique(df["Subject IDs"].values.tolist())
    fold_1 = random.sample(list(subjects), int(len(subjects) / 2))
    df_fold_1 = df[df["Subject IDs"].isin(fold_1)]
    df_fold_2 = df[~df["Subject IDs"].isin(fold_1)]
    print(len(df_fold_1))
    print(len(df_fold_2))

    print(subjects)
    print(fold_1)
    return df_fold_1, df_fold_2


class VIPLV1HRDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict):
        super(VIPLV1HRDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "VIPL-V1"

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["AVI File"]

        clip_ = load_sample(f_path=clip_path)

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[self.dataset])

    def __len__(self):
        return len(self.df)


def get_v1_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = VIPLV1HRDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = VIPLV1HRDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    df_fold_1, df_fold_2 = prep_splits_V2(root_loc=r"D:\anshul\remoteHR\VIPL-HR-V2")

    df_fold_1, df_fold_2 = prep_splits_V1(root_loc=r"D:\anshul\remoteHR\VIPL-HR-V1")
