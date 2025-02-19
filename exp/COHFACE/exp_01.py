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


def train(model, n_epochs, train_loader, val_loader, device, 
          base_len, optimizer, criterion, scheduler,
          model_dir, model_file_name):
    mean_training_losses = []
    mean_valid_losses = []
    lrs = []
    best_epoch = 0
    min_valid_loss = None
    for epoch in range(1, n_epochs+1):
        print('')
        print(f"====Training Epoch: {epoch}====")
        running_loss = 0.0
        train_loss = []
        model.train()
        # Model Training
        tbar = tqdm(train_loader, ncols=80, total=len(train_loader))
        for idx, batch in enumerate(tbar):
            tbar.set_description("Train epoch %s" % epoch)

            data, labels = batch[0].to(device), batch[1].to(device)
            N, D, C, H, W = data.shape # batch size, depth (time frames), channels, H, W
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)
            data = data[:(N * D) // base_len * base_len]
            labels = labels[:(N * D) // base_len * base_len]
            
            optimizer.zero_grad()
            pred_ppg = model(data)
            loss = criterion(pred_ppg, labels)
            loss.backward()

            # Append the current learning rate to the list
            lrs.append(scheduler.get_last_lr())

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if idx % 100 == 99:  # print every 100 mini-batches
                print(
                    f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            train_loss.append(loss.item())
            tbar.set_postfix(loss=loss.item())

        # Append the mean training loss for the epoch
        mean_training_losses.append(np.mean(train_loss))

        save_model(model, model_dir, model_file_name, epoch)
        valid_loss = valid(model, val_loader, device, criterion, base_len)
        mean_valid_losses.append(valid_loss)
        print('validation loss: ', valid_loss)
        if min_valid_loss is None:
            min_valid_loss = valid_loss
            best_epoch = epoch
            print("Update best model! Best epoch: {}".format(best_epoch))
        elif (valid_loss < min_valid_loss):
            min_valid_loss = valid_loss
            best_epoch = epoch
            print("Update best model! Best epoch: {}".format(best_epoch))
        torch.cuda.empty_cache()
        
    print("best trained epoch: {}, min_val_loss: {}".format(best_epoch, min_valid_loss))
    return model, mean_training_losses, mean_valid_losses, lrs, best_epoch


def valid(model, val_loader, device, criterion, base_len):
    """ Model evaluation on the validation dataset."""
    print('')
    print("===Validating===")
    valid_loss = []
    model.eval()
    valid_step = 0
    with torch.no_grad():
        vbar = tqdm(val_loader, ncols=80)
        for valid_idx, valid_batch in enumerate(vbar):
            vbar.set_description("Validation")
            data_valid, labels_valid = valid_batch[0].to(device), valid_batch[1].to(device)
            N, D, C, H, W = data_valid.shape
            data_valid = data_valid.view(N * D, C, H, W)
            labels_valid = labels_valid.view(-1, 1)
            data_valid = data_valid[:(N * D) // base_len * base_len]
            labels_valid = labels_valid[:(N * D) // base_len * base_len]
            pred_ppg_valid = model(data_valid)
            loss = criterion(pred_ppg_valid, labels_valid)
            valid_loss.append(loss.item())
            valid_step += 1
            vbar.set_postfix(loss=loss.item())
        valid_loss = np.asarray(valid_loss)
    return np.mean(valid_loss)

def test(model, data_loader, device, base_len, model_dir, fs):
    """ Model evaluation on the testing dataset."""
    print('')
    print("===Testing===")
    predictions = list()
    labels = list()
    patient_session_id = list()

    model = model.to(device)
    model.eval()
    print("Running model evaluation on the testing dataset!")
    
    with torch.no_grad():
        for _, test_batch in enumerate(tqdm(data_loader, ncols=80)):
            ps_id = test_batch[2]
            chunk_id = test_batch[2]
            data_test, labels_test = test_batch[0].to(device), test_batch[1].to(device)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)
            labels_test = labels_test.view(-1, 1)
            data_test = data_test[:(N * D) // base_len * base_len]
            labels_test = labels_test[:(N * D) // base_len * base_len]
            pred_ppg_test = model(data_test)

            labels_test = labels_test.cpu()
            pred_ppg_test = pred_ppg_test.cpu()
            
            labels_test = labels_test.view(N, D)
            pred_ppg_test = pred_ppg_test.view(N, D, -1)
            for l, p, p_s in zip(labels_test, pred_ppg_test, ps_id):
                labels.append(l)
                predictions.append(p)
                patient_session_id.append(p_s)

    df = pd.DataFrame()
    df["patient_session_ID"] = patient_session_id
    df["gt_pulse"] = labels
    df["pred_pulse"] = predictions
    df.to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)
    print('')
    calculate_metrics(predictions, labels, fs=fs, result_dir=model_dir)


def save_model(model, model_dir, model_file_name, index):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_file_name + '_Epoch' + str(index) + '.pth')
    torch.save(model.state_dict(), model_path)
    print('Saved Model Path: ', model_path)

def load_model(model, model_dir, model_file_name):
    model_path = os.path.join(model_dir, model_file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist !!")
    model.load_state_dict(torch.load(model_path))
    print('Loaded Model Path: ', model_path)
    return model


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    device = torch.device("cuda")
    
    max_epochs = 1
    
    model_dir = "/home/ubuntu/rPPG/models/COHFACE"
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "COHFACE_exp"
    batch_size = 3
    
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


    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    train_dataset = TSCanDataset(patient_session_ids=["1_0", "1_1", "1_2", "1_3", "33_0", "33_1", "33_2", "33_3"],
                                 root_loc="/home/ubuntu/rPPG/data/cohface",
                                 cache_dir="/home/ubuntu/rPPG/data/cache/cohface",
                                 width=W, height=H, chunk_length=chunk_len)
    val_dataset = TSCanDataset(patient_session_ids=["1_0", "1_1", "1_2", "1_3"],
                               root_loc="/home/ubuntu/rPPG/data/cohface",
                               cache_dir="/home/ubuntu/rPPG/data/cache/cohface",
                               width=W, height=H, chunk_length=chunk_len)
    test_dataset = TSCanDataset(patient_session_ids=["1_0", "1_1", "1_2", "1_3", "33_0", "33_1", "33_2", "33_3"],
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
    
    # model, mean_training_losses, mean_valid_losses, lrs, be = train(model, max_epochs, train_loader, 
    #                                                             val_loader, device, base_len, optimizer, 
    #                                                             criterion, scheduler, model_dir, model_file_name)
    
    # TESTING
    test(model, test_loader, device, base_len, model_dir, fs)