import pandas as pd
from sklearn.model_selection import train_test_split 
import torch
from torch.utils.data import Dataset, DataLoader

from loader.utils import read_video, preprocess_data, read_gt, extra_info
import logger
import logging

logrPPG = logging.getLogger(__name__)


class rPPGDataloader:
    def __init__(self, name_: str, df_loc: str, split_by: str = None, train_split: str = None, 
                 condition: str = None, pp_config: dict = None):
        """
            A class to create rPPG dataset for training on different rPPG datasets
            arguments:
                name: name of the dataset,
                df_loc: location of the root dataframe (Needs "vid" and "gt" columns in dataframe)
                split_by: column to use to split the dataset (should be one of the column in dataframe)
                train_split: value of the split-by column to be used for creating train-split
                condition: condition for training split (allowed conditions, "==", "<", "<=", ">", ">=", and "!=")
        """
        self.ds_name = name_
        self.df_loc = df_loc
        self.split_by = split_by
        self.train_split = train_split
        self.condition = condition
        self.pp_config = pp_config

        self.df = pd.read_csv(self.df_loc)
        allowed_conditions = ["==", ">", "<", ">=", "<=", "!="]
        assert "vid" in self.df.columns, f"Column 'vid' not in {self.df_loc}"
        assert "gt" in self.df.columns, f"Column 'gt' not in {self.df_loc}"
        if split_by and train_split and condition:
            assert "split_by" in self.df.columns, f"Column 'split_by' not in {self.df_loc}"
            assert self.condition in allowed_conditions, f"condition {self.condition} not in {allowed_conditions}" 
            
    def name(self):
        logrPPG.info(f"Name of the dataset is {self.name}")
    
    def split_df(self):
        match self.condition:
            case "==":
                self.train_df = self.df[self.df[self.split_by] == self.train_split]
                self.test_df = self.df[self.df[self.split_by] != self.train_split]
            case "!=":
                self.train_df = self.df[self.df[self.split_by] != self.train_split]
                self.test_df = self.df[self.df[self.split_by] == self.train_split]
            case "<":
                self.df[[self.split_by]] = self.df[[self.split_by]].apply(pd.to_numeric)
                self.train_df = self.df[self.df[self.split_by] < float(self.train_split)]
                self.test_df = self.df[self.df[self.split_by] >= float(self.train_split)]
            case ">":
                self.df[[self.split_by]] = self.df[[self.split_by]].apply(pd.to_numeric)
                self.train_df = self.df[self.df[self.split_by] > float(self.train_split)]
                self.test_df = self.df[self.df[self.split_by] <= float(self.train_split)]
            case "<=":
                self.df[[self.split_by]] = self.df[[self.split_by]].apply(pd.to_numeric)
                self.train_df = self.df[self.df[self.split_by] <= float(self.train_split)]
                self.test_df = self.df[self.df[self.split_by] > float(self.train_split)]
            case ">=":
                self.df[[self.split_by]] = self.df[[self.split_by]].apply(pd.to_numeric)
                self.train_df = self.df[self.df[self.split_by] >= float(self.train_split)]
                self.test_df = self.df[self.df[self.split_by] < float(self.train_split)]
            case _: 
                logrPPG.error(f"{self.condition} not an allowed condition")
                raise NotImplementedError(f"{self.condition} not implemented")
    
    def get_train_test_set(self, split_val_ratio: float = 0.0):
        """
            split the training set into train and validtion if split_val_ratio > 0
        """
        self.split_df()
        if split_val_ratio > 0:
            train_df, val_df = train_test_split(self.train_df, test_size=split_val_ratio)
            test_df = self.test_df
        else:
            train_df, val_df = self.train_df, None
            test_df = self.test_df
        train_dataset = DFDataset(split="train", dataset=self.ds_name, 
                                  df=train_df, pp_config=self.pp_config)
        if val_df:
            val_dataset = DFDataset(split="validation", dataset=self.ds_name, 
                                    df=val_df, pp_config=self.pp_config)
        else:
            val_dataset = None
        test_dataset = DFDataset(split="test", dataset=self.ds_name,
                                 df=test_df, pp_config=self.pp_config)
        return train_dataset, val_dataset, test_dataset


class DFDataset(Dataset):
    """
        Dataset to load rPPG dataset from a dataframe
        The dataframe needs two columns vid, gt
            vid -> location of the video on the device (.avi, .mkv)
            gt -> location of gt PPG or ECG signal ().csv, .txt, .hdf5)
    """
    def __init__(self, split: str, dataset: str, df: pd.DataFrame, pp_config: dict):
        self.split = split
        self.df = df
        self.dataset = dataset
        self.pp_config = pp_config
    
    def ds_split(self):
        logger.info(f"SPLIT {self.dataset} of the dataset {self.dataset}")
    
    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path, gt_path = row_["vid"], row_["gt"]
        
        additional_info = extra_info(clip_path, gt_path, self.dataset)
        vid, gt = preprocess_data(clip_path, gt_path,self.dataset, self.pp_config)

        return {
            "video": vid,
            "PPG": gt,
            "additional_info": additional_info
        }

    def __len__(self):
        return len(self.df)
