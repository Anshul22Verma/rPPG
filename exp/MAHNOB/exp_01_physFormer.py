import math
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("C:\\Users\\transponster\\Documents\\anshul\\rPPG")

from model.physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from loader.mahnob_loaders import physFormerDataset
from utils.evaluation import calculate_metrics
from utils.neg_pearson_loss import Neg_Pearson
from utils.helper import get_hr
from utils.physFormerLossComputer import TorchLossComputer


def train(model, n_epochs, train_loader, val_loader, device,
          optimizer, criterion_pearson, fs,
          scheduler, model_dir, model_file_name):
    a_start = 1.0
    b_start = 1.0
    # a: Pearson loss, b: frequency loss
    exp_b = 1.0

    mean_training_losses = []
    mean_valid_losses = []
    lrs = []
    best_epoch = 0
    min_valid_loss = None
    for epoch in range(1, n_epochs + 1):
        print('')
        print(f"====Training Epoch: {epoch}====")
        running_loss = 0.0
        loss_rPPG_avg = []
        loss_peak_avg = []
        loss_kl_avg_test = []
        loss_hr_mae = []

        model.train()
        # Model Training
        tbar = tqdm(train_loader, ncols=80, total=len(train_loader))
        for idx, batch in enumerate(tbar):
            tbar.set_description("Train epoch %s" % epoch)

            # hr --> labels need to get HR from PPG
            hr = torch.tensor([get_hr(i, sr=fs) for i in batch[1]]).float().to(device)
            data, label = batch[0].to(device), batch[1].to(device)


            optimizer.zero_grad()

            gra_sharp = 2.0
            rppg, _, _, _ = model(data, gra_sharp)
            # standard normalization
            rppg = (rppg - torch.mean(rppg, axis=-1).view(-1, 1))/torch.std(rppg, axis=-1).view(-1, 1)
            loss_rppg = criterion_pearson(rppg, label)

            fre_loss, kl_loss, train_mae = 0.0, 0.0, 0.0
            for bb in range(data.shape[0]):
                loss_distribution_kl, fre_loss_temp, traim_mae_temp = \
                    TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rppg[bb], hr[bb], fs, std=1.0
                    )
                fre_loss = fre_loss + fre_loss_temp
                kl_loss = kl_loss + loss_distribution_kl
                train_mae = train_mae + traim_mae_temp
            fre_loss /= data.shape[0]
            kl_loss /= data.shape[0]
            train_mae /= data.shape[0]

            if epoch > 10:
                a, b = 0.05, 5.0
            else:
                a, b = a_start, b_start*math.pow(exp_b, epoch/10.0)

            loss = a*loss_rppg + b*(fre_loss + kl_loss)
            loss.backward()
            # Append the current learning rate to the list
            lrs.append(scheduler.get_last_lr())

            optimizer.step()

            loss_rPPG_avg.append(float(loss_rppg.data))
            loss_peak_avg.append(float(fre_loss.data))
            loss_kl_avg_test.append(float(kl_loss.data))
            loss_hr_mae.append(float(train_mae))
            running_loss += loss.item()

            if idx % 100 == 99:  # print every 100 mini-batches
                print(
                    f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            tbar.set_postfix(loss=loss.item())

        # Append the mean training loss for the epoch
        mean_training_losses.append(np.mean(loss_rPPG_avg))

        save_model(model, model_dir, model_file_name, epoch)
        valid_loss = valid(model, val_loader, device, fs)
        mean_valid_losses.append(valid_loss)
        scheduler.step()
        print('validation loss (RMSE): ', valid_loss)
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


def valid(model, val_loader, device, fs):
    """ Model evaluation on the validation dataset."""
    print('')
    print("===Validating===")
    model.eval()
    with torch.no_grad():
        hrs = []
        vbar = tqdm(val_loader, ncols=80)
        for valid_idx, valid_batch in enumerate(vbar):
            vbar.set_description("Validation")
            data_valid, labels_valid = valid_batch[0].to(device), valid_batch[1].to(device)
            gra_sharp = 2.0
            rppg, _, _, _ = model(data_valid, gra_sharp)
            rppg = (rppg - torch.mean(rppg, axis=-1).view(-1, 1)) / torch.std(rppg, axis=-1).view(-1, 1)
            for _1, _2 in zip(rppg, labels_valid):
                hrs.append((get_hr(_1.cpu().detach().numpy(), sr=fs), get_hr(_2.cpu().detach().numpy(), sr=fs)))

        valid_loss = np.mean([(i-j)**2 for i, j in hrs])**0.5
    return valid_loss


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
            chunk_id = test_batch[3]
            data_test, labels_test = test_batch[0].to(device), test_batch[1].to(device)

            gra_sharp = 2.0
            pred_ppg_test, _, _, _ = model(data_test, gra_sharp)

            labels_test = labels_test.cpu()
            pred_ppg_test = pred_ppg_test.cpu()

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

    model_dir = r"D:\anshul\remoteHR\models\COHFACE\physFormer_exp_01"
    os.makedirs(model_dir, exist_ok=True)
    model_file_name = "COHFACE_exp"
    batch_size = 3

    base_len, frame_depth, chunk_len = 10, 10, 200

    lr = 1e-4
    H, W = 128, 128
    patch_size = 4
    dim, ff_dim, num_heads, num_layers = 96, 144, 4, 12
    dropout_rate, theta = 0.1, 0.7
    seed = 2025
    fs = 20
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    general_generator = torch.Generator()
    general_generator.manual_seed(seed)

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(chunk_len, H, W), patches=(patch_size,)*3,
                                             dim=dim, ff_dim=ff_dim, num_heads=num_heads, num_layers=num_layers,
                                             dropout_rate=dropout_rate, theta=theta).to(device)

    criterion_reg = torch.nn.MSELoss()
    criterion_L1Loss = torch.nn.L1Loss()
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_pearson = Neg_Pearson()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    train_dataset = physFormerDataset(session_ids=["102", "103", "104", "105", "106", "107", "108", "109"],
                                      root_loc=r"D:\anshul\remoteHR\mahnob\Sessions",
                                      cache_dir=r"D:\anshul\remoteHR\cache\MAHNOB",
                                      width=W, height=H, chunk_length=chunk_len)
    val_dataset = physFormerDataset(session_ids=["120", "121", "122", "123"],
                                    root_loc=r"D:\anshul\remoteHR\mahnob\Sessions",
                                    cache_dir=r"D:\anshul\remoteHR\cache\MAHNOB",
                                    width=W, height=H, chunk_length=chunk_len)
    test_dataset = physFormerDataset(session_ids=["102", "103", "104", "105"],
                                     root_loc=r"D:\anshul\remoteHR\mahnob\Sessions",
                                     cache_dir=r"D:\anshul\remoteHR\cache\MAHNOB",
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

    model, mean_training_losses, mean_valid_losses, lrs, be = train(model, max_epochs, train_loader,
                                                                    val_loader, device, optimizer,
                                                                    criterion_pearson, fs,
                                                                    scheduler, model_dir, model_file_name)

    # TESTING
    test(model, test_loader, device, base_len, model_dir, fs)
