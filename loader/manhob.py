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
from utils.helper import read_xml, age, load_sample, get_transforms

random_state = 2023
random.seed(random_state)


def prep_splits(root_loc: str):  # -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    hr_subset = os.path.join(root_loc, "session_hr_info.xlsx")
    assert os.path.exists(hr_subset), f"Root location does not have excel containing HR information {hr_subset}"
    df = pd.read_excel(hr_subset)

    root_session_data = os.path.join(root_loc, "Sessions")
    assert os.path.exists(root_session_data), "Root location does not have Sessions data"
    root_subject_data = os.path.join(root_loc, "Subjects")
    assert os.path.exists(root_subject_data), "Root location does not have Subjects data"

    if not os.path.exists(os.path.join(root_loc, "session_hr_info_w_subject.csv")):
        subject_ids = []
        subject_gender = []
        subject_ethinicity = []
        avi_file = []
        subject_age = []  # at time of MANHOB-release 04th August, 2011, format YYYY-M-D
        release_date = "2011-08-11"
        release_date = datetime.datetime.strptime(release_date, "%Y-%m-%d")
        # get subject ID for all the folders with HR dataset and add it to the df
        for dir in df['folder'].values.tolist():
            dir = str(dir).replace("\\", "")
            session_dir = os.path.join(root_session_data, str(dir))
            session_xml = read_xml(os.path.join(session_dir, "session.xml"))
            subject_id = session_xml["session"]["subject"]["@id"]
            subject_ids.append(subject_id)

            # get the path of the video
            avi_f = None
            for f in os.listdir(session_dir):
                if ".avi" in f:
                    avi_f = os.path.join(session_dir, f)
                    break
            avi_file.append(avi_f)

            subject_xml = read_xml(os.path.join(root_subject_data, f"subject{subject_id}.xml"))
            dob = subject_xml["subject"]["@dob"]
            y, m, d = dob.split("-")
            if len(m) == 1:
                m = "0" + m
            if len(d) == 1:
                d = "0" + d
            dob = datetime.datetime.strptime(f"{y}-{m}-{d}", "%Y-%m-%d")
            subject_age.append(age(dob, release_date))
            subject_ethinicity.append(subject_xml["subject"]["@ethnicity"])
            subject_gender.append(subject_xml["subject"]["@gender"])

        df["Subject IDs"] = subject_ids
        df["Subject Age"] = subject_age
        df["Subject Ethnicity"] = subject_ethinicity
        df["Sex"] = subject_gender
        df["AVI File"] = avi_file

        df.to_csv(os.path.join(root_loc, "session_hr_info_w_subject.csv"), index=False)

    df = pd.read_csv(os.path.join(root_loc, "session_hr_info_w_subject.csv"))
    subjects = np.unique(df["Subject IDs"].values.tolist())
    fold_1 = random.sample(list(subjects), int(len(subjects)/2))

    df_fold1 = df[df["Subject IDs"].isin(fold_1)]
    df_fold2 = df[~df["Subject IDs"].isin(fold_1)]

    # print(np.unique(df_fold1["Subject IDs"].values.tolist()))
    # print(np.unique(df_fold2["Subject IDs"].values.tolist()))
    return df_fold1, df_fold2


class MANHOBHRDatasetDBClassification(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict):
        super(MANHOBHRDatasetDBClassification, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.is_train = train
        self.encodings = encodings
        self.dataset = "MANHOB"

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["AVI File"]

        clip_ = load_sample(f_path=clip_path)

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[self.dataset])

    def __len__(self):
        return len(self.df)


def get_data_loaders(train_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict):
    train_dataset = MANHOBHRDatasetDBClassification(df=train_df, train=True, encodings=encodings)
    test_dataset = MANHOBHRDatasetDBClassification(df=test_df, train=True, encodings=encodings)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    df_fold_1, df_fold_2 = prep_splits(root_loc=r"D:\anshul\remoteHR\mahnob")

    avi_file = df_fold_1["AVI File"].values.tolist()[0]
    vid = load_sample(avi_file)

    # 300 X (780 X 580 X 3) --> frames X H X W X C
    # 300 X (224 X 224 X 3) --> frames X H X W X C

    print(vid)
    print(len(vid))
