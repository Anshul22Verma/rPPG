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
from utils.helper import read_hdf5, load_sample, get_transforms, load_stl_map_sample

random_state = 2023
random.seed(random_state)


def prep_splits_V2(root_loc: str):  # -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hr_sheet = os.path.join(root_loc, "hr_info.csv")

    root_train_loc = os.path.join(root_loc, "train")
    if not os.path.exists(hr_sheet):
        patients = [f for f in os.listdir(root_train_loc) if
                    os.path.isdir(os.path.join(root_train_loc, f)) and f.isnumeric()]

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
    fold_1 = random.sample(list(subjects), int(len(subjects) / 2))
    df_fold_1 = df[df["Subject IDs"].isin(fold_1)]
    df_fold_2 = df[~df["Subject IDs"].isin(fold_1)]

    print(len(df_fold_1))
    print(len(df_fold_2))
    # print(np.unique(df_fold1["Subject IDs"].values.tolist()))
    # print(np.unique(df_fold2["Subject IDs"].values.tolist()))
    return df_fold_1, df_fold_2


class VIPLV2HRDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict, stl_map: dict = {}):
        super(VIPLV2HRDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "VIPL-V2"
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


def get_v2_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = VIPLV2HRDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = VIPLV2HRDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def get_v2_stl_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict, stl_map: dict):
    train_dataset = VIPLV2HRDatasetDBClassification(df=train_df, train=True, encodings=encodings, stl_map=stl_map)
    test_dataset = VIPLV2HRDatasetDBClassification(df=test_df, train=True, encodings=encodings, stl_map=stl_map)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def prep_splits_V1(root_loc: str):
    hr_sheet = os.path.join(root_loc, "hr_info.csv")

    root_train_loc = os.path.join(root_loc, "data")
    if not os.path.exists(hr_sheet):
        subjects = [f for f in os.listdir(root_train_loc) if
                    os.path.isdir(os.path.join(root_train_loc, f)) and ".zip" not in f]
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
    # print(len(df_fold_1))
    # print(len(df_fold_2))
    #
    # print(subjects)
    # print(fold_1)
    return df_fold_1, df_fold_2


class VIPLV1HRDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict, stl_map: dict = {}):
        super(VIPLV1HRDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "VIPL-V1"
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


def get_v1_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = VIPLV1HRDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = VIPLV1HRDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def get_v1_stl_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict, stl_map: dict):
    train_dataset = VIPLV1HRDatasetDBClassification(df=train_df, train=True, encodings=encodings, stl_map=stl_map)
    test_dataset = VIPLV1HRDatasetDBClassification(df=test_df, train=True, encodings=encodings, stl_map=stl_map)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def prep_rhythmnet_splits_V1(labels_path: str) -> (pd.DataFrame, pd.DataFrame):
    # maps_path: str,
    # fold1 = [p for p in patients if int(p.relace("p", "")) % 2 == 0]
    # fold2 = [p for p in patients if int(p.relace("p", "")) % 2 == 1]

    # maps_path = r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps'
    # files = os.listdir(maps_path)
    # fold1_files = [os.path.join(maps_path, f) for f in files if int(f.split("_")[0].replace("p", "")) % 2 == 0]
    # fold2_files = [os.path.join(maps_path, f) for f in files if int(f.split("_")[0].replace("p", "")) % 2 == 1]

    # labels_path = r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
    labels = os.listdir(labels_path)
    df_fold_1 = []
    df_fold_2 = []
    for l in labels:
        p = l.split("_")[0]
        if int(p.replace("p", "")) % 2 == 0:
            df_fold_1.append(pd.read_csv(os.path.join(labels_path, l)))
        else:
            df_fold_2.append(pd.read_csv(os.path.join(labels_path, l)))

    df_fold_1 = pd.concat(df_fold_1)
    df_fold_2 = pd.concat(df_fold_2)
    return df_fold_1, df_fold_2  # fold1_files, df_fold_1, fold2_files, df_fold_2


class DatasetRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, df_paths_labels):
        # self.H = 180
        # self.W = 180
        # self.C = 3
        # # self.video_path = data_path
        # self.st_maps_path = st_maps_path
        # # self.resize = resize
        # self.target_path = target_signal_path
        # self.maps = None

        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )
        self.df = df_paths_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r = self.df.iloc[index]
        # print(r)
        # identify the name of the video file to get the ground truth signal
        f_name = r["STL_map"]  # .values.tolist()[0]
        HR = r["HR"]  # .values.tolist()[0]

        # targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # sampling rate is video fps (check)

        # Load the maps for video at 'index'
        map = np.load(f_name)
        map_shape = map.shape
        # print(map_shape)
        map = map.reshape((map_shape[2], map_shape[0], map_shape[1]))
        # print(map.shape)
        # target_hr = calculate_hr(targets, timestamps=timestamps)
        # target_hr = calculate_hr_clip_wise(map_shape[0], targets, timestamps=timestamps)
        # target_hr = get_hr_data(self.video_file_name)
        # To check the fact that we don't have number of targets greater than the number of maps
        # target_hr = target_hr[:map_shape[0]]
        # self.maps = self.maps[:target_hr.shape[0], :, :, :]

        target_hr = int(HR)
        return torch.tensor(map, dtype=torch.float), torch.tensor(target_hr, dtype=torch.float)
        #     {
        #     "st_map": torch.tensor(map, dtype=torch.float),
        #     "target": torch.tensor(target_hr, dtype=torch.float)
        # }


if __name__ == "__main__":
    # df_fold_1, df_fold_2 = prep_splits_V2(root_loc=r"D:\anshul\remoteHR\VIPL-HR-V2")
    #
    # avi_file = df_fold_1["AVI File"].values.tolist()[0]
    # vid = load_sample(avi_file, th=1)
    #
    # # 300 X (780 X 580 X 3) --> frames X H X W X C
    # # 300 X (224 X 224 X 3) --> frames X H X W X C
    #
    # # print(vid)
    # print(len(vid))
    # # 26 fps

    df_fold_1, df_fold_2 = prep_rhythmnet_splits_V1(
        # maps_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps',
        labels_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
    )

    training_data = DatasetRhythmNet(df_paths_labels=df_fold_1)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features)
    print(train_features.shape)
    print(train_labels.shape)
    print(train_labels)

    test_data = DatasetRhythmNet(df_paths_labels=df_fold_2)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    # avi_file = df_fold_1["AVI File"].values.tolist()[0]
    # vid = load_sample(avi_file, th=1)

    # 300 X (780 X 580 X 3) --> frames X H X W X C
    # 300 X (224 X 224 X 3) --> frames X H X W X C

    # print(vid)
    print(len(df_fold_1))
    print(len(df_fold_2))
    # print(f1_files)
    print(df_fold_1.head())
    print(df_fold_2.head())
    # 25 fps
