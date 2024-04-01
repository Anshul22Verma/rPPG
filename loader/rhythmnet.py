import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, df):
        self.df = df
        self.data_root = r'D:\anshul\remoteHR\VIPL-HR-V1\data'
        self.H = 180
        self.W = 180
        self.C = 3
        # self.video_path = data_path
        # self.st_maps_path = st_maps_path
        # # self.resize = resize
        # self.target_path = target_signal_path
        # self.maps = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # identify the name of the video file to get the ground truth signal
        stl_file_path = row["npy_file"]
        target = row["HR"]

        # Load the maps for video at 'index'
        # print(stl_file_path)
        self.maps = np.load(stl_file_path, allow_pickle=True)
        map_shape = self.maps.shape
        self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

        p, v, s = os.path.basename(stl_file_path.replace(".npy", "")).split("_")
        # print(os.path.join(self.data_root, p, v, s, 'gt_HR.csv'))
        gt_HR = pd.read_csv(os.path.join(self.data_root, p, v, s, 'gt_HR.csv'))["HR"].values.tolist()
        # print(gt_HR.head())
        target_hr = []
        target_hr.append(gt_HR[0])
        for h in gt_HR:
            target_hr.append(h)
            target_hr.append(h)
        target_hr.append(gt_HR[-1])
        target_hr = np.array(target_hr)
        # sample HR to get one value for each
        # target_hr = get_hr_data(self.video_file_name)
        # To check the fact that we don't have number of targets greater than the number of maps

        target_hr = target_hr[:map_shape[0]]
        self.maps = self.maps[:target_hr.shape[0], :, :, :]
        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float),
            # "mean_target": torch.tensor(target, dtype=torch.float)
        }
