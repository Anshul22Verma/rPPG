import numpy as np
import os
import pandas as pd
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("/home/ubuntu/rPPG/rPPG_pipe")

from model.ts_can import TSCAN
from loader.base_loader import TSCanDataset
from utils.evaluation import calculate_metrics
from exp.COHFACE.exp_01 import *

# 6 has 0-7 sessions
males_fold1 = ["1_0", "1_1", "1_2", "1_3", "3_0", "3_1", "3_2", "3_3", 
               "5_0", "5_1", "5_2", "5_3", "7_0", "7_1", "7_2", "7_3", 
               "9_0", "9_1", "9_2", "9_3", "11_0", "11_1", "11_2", "11_3", 
               "13_0", "13_1", "13_2", "13_3", "15_0", "15_1", "15_2", "15_3", 
               "17_0", "17_1", "17_2", "17_3", "25_0", "25_1", "25_2", "25_3", 
               "31_0", "31_1", "31_2", "31_3", "33_0", "33_1", "33_2", "33_3", 
               "37_0", "37_1", "37_2", "37_3", "39_0", "39_1", "39_2", "39_3"]
males_fold2 = ["2_0", "2_1", "2_2", "2_3", "4_0", "4_1", "4_2", "4_3", 
               "6_0", "6_1", "6_2", "6_3", "6_4", "6_5", "6_6", "6_7", "8_0", "8_1", "8_2", "8_3", 
               "10_0", "10_1", "10_2", "10_3", "12_0", "12_1", "12_2", "12_3", 
               "14_0", "14_1", "14_2", "14_3", "16_0", "16_1", "16_2", "16_3", 
               "20_0", "20_1", "20_2", "20_3", "26_0", "26_1", "26_2", "26_3", 
               "30_0", "30_1", "30_2", "30_3", "32_0", "32_1", "32_2", "32_3", 
               "34_0", "34_1", "34_2", "34_3", "36_0", "36_1", "36_2", "36_3"]

females = ["18_0", "18_1", "18_2", "18_3", "19_0", "19_1", "19_2", "19_3", 
           "21_0", "21_1", "21_2", "21_3", "22_0", "22_1", "22_2", "22_3", 
           "23_0", "23_1", "23_2", "23_3", "24_0", "24_1", "24_2", "24_3", 
           "27_0", "27_1", "27_2", "27_3", "28_0", "28_1", "28_2", "28_3", 
           "29_0", "29_1", "29_2", "29_3", "35_0", "35_1", "35_2", "35_3", 
           "38_0", "38_1", "38_2", "38_3", "40_0", "40_1", "40_2", "40_3"]

if __name__ == "__main__":
    device = torch.device("cuda")
    
    max_epochs = 20
    
    model_dir = "/home/ubuntu/rPPG/models/COHFACE/sex_bias_male_fold1"
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "COHFACE_exp"
    batch_size = 2
    
    base_len, frame_depth, chunk_len = 10, 10, 200
    
    lr = 1e-4
    H, W = 128, 128
    seed = 2025
    fs = 20
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    general_generator = torch.Generator()
    general_generator.manual_seed(seed)
    
    model = TSCAN(in_channels=3, frame_depth=frame_depth, img_size=H).to(device)
    model = load_model(model, model_dir=model_dir, model_file_name=f"{model_file_name}_Epoch8.pth")


    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    train_dataset = TSCanDataset(patient_session_ids=males_fold1,
                                 root_loc="/home/ubuntu/rPPG/data/cohface",
                                 cache_dir="/home/ubuntu/rPPG/data/cache/cohface",
                                 width=W, height=H, chunk_length=chunk_len)
    val_dataset = TSCanDataset(patient_session_ids=males_fold2,
                               root_loc="/home/ubuntu/rPPG/data/cohface",
                               cache_dir="/home/ubuntu/rPPG/data/cache/cohface",
                               width=W, height=H, chunk_length=chunk_len)
    test_dataset = TSCanDataset(patient_session_ids=females,
                                root_loc="/home/ubuntu/rPPG/data/cohface",
                                cache_dir="/home/ubuntu/rPPG/data/cache/cohface",
                                width=W, height=H, chunk_length=chunk_len)
    train_loader = DataLoader(
                dataset=train_dataset,
                num_workers=1,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
    val_loader = DataLoader(
                dataset=val_dataset,
                num_workers=1,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
    test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=1,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    num_train_batches = len(train_loader)
    # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=max_epochs, 
                                                    steps_per_epoch=num_train_batches)
    
    mean_training_losses, mean_valid_losses, lrs, best_epoch = train(model, max_epochs, train_loader, 
                                                         val_loader, device, base_len, optimizer, 
                                                         criterion, scheduler, model_dir, model_file_name)
    
    model = load_model(model, model_dir=model_dir, model_file_name=f"{model_file_name}_Epoch{best_epoch}.pth")
    # TESTING
    val_dir = os.path.join(model_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test(model, val_loader, device, base_len, val_dir, fs)    
    test(model, test_loader, device, base_len, model_dir, fs)