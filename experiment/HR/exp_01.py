import os
import glob
import torch
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
import PIL
import sys
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

sys.path.append(r'C:\Users\transponster\Documents\anshul\rPPG')

import models.rhythmNet.config as config
from models.rhythmNet.rhythmnet import RhythmNet
from models.rhythmNet.loss import RhythmNetLoss
from loader.rhythmnet import DataLoaderRhythmNet
import experiment.HR.engine_vipl as engine_vipl


def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


def rmse(l1, l2):

    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):

    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    pearson_per_signal = []
    HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE}


def prep_rhythmnet_splits_V1(labels_path: str) -> (pd.DataFrame, pd.DataFrame):
    # maps_path: str,
    # fold1 = [p for p in patients if int(p.relace("p", "")) % 2 == 0]
    # fold2 = [p for p in patients if int(p.relace("p", "")) % 2 == 1]

    # maps_path = r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps'
    # files = os.listdir(maps_path)
    # fold1_files = [os.path.join(maps_path, f) for f in files if int(f.split("_")[0].replace("p", "")) % 2 == 0]
    # fold2_files = [os.path.join(maps_path, f) for f in files if int(f.split("_")[0].replace("p", "")) % 2 == 1]

    # labels_path = r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
    labels = pd.read_csv(labels_path)
    mask_fold_1 = []
    mask_fold_2 = []
    for idx, r in labels.iterrows():
        p = os.path.basename(r["npy_file"]).split("_")[0]
        if int(p.replace("p", "")) % 2 == 0:
            mask_fold_1.append(True)
            mask_fold_2.append(False)
        else:
            mask_fold_2.append(True)
            mask_fold_1.append(False)

    df_fold_1 = labels[mask_fold_1]
    df_fold_2 = labels[mask_fold_2]
    return df_fold_1, df_fold_2  # fold1_files, df_fold_1, fold2_files, df_fold_2


def load_model_if_checkpointed(model, optimizer, checkpoint_path, load_on_cpu=False):
    loss = 0.0
    checkpoint_flag = False

    # check if checkpoint exists
    if os.path.exists(os.path.join(checkpoint_path, "running_model.pt")):
        checkpoint_flag = True
        if load_on_cpu:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    return model, optimizer, loss, checkpoint_flag


def save_model_checkpoint(model, optimizer, loss, checkpoint_path):
    save_filename = "running_model.pt"
    # checkpoint_path = os.path.join(checkpoint_path, save_filename)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save({
        # 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_path, save_filename))
    print('Saved!')


def gt_vs_est(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    # mean = np.mean([data1, data2], axis=0)
    # diff = data1 - data2                   # Difference between data1 and data2
    # md = np.mean(diff)                   # Mean of the difference
    # sd = np.std(diff, axis=0)            # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(data1, data2)
    plt.title('true labels vs estimated')
    plt.ylabel('estimated HR')
    plt.xlabel('true HR')
    # plt.axhline(md,           color='gray', linestyle='--')
    # plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    # plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    else:
        # plt.show()
        fig.savefig(plot_path + f'/true_vs_est.png', dpi=fig.dpi)


def bland_altman_plot(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    else:
        # plt.show()
        fig.savefig(plot_path + f'/bland-altman_new.png', dpi=fig.dpi)


def create_plot_for_tensorboard(plot_name, data1, data2):
    if plot_name == "bland_altman":
        fig_buf = bland_altman_plot(data1, data2, tb=True)

    if plot_name == "gt_vs_est":
        fig_buf = gt_vs_est(data1, data2, tb=True)

    image = PIL.Image.open(fig_buf)
    image = ToTensor()(image)

    return image


def run_training():

    # check path to checkpoint directory
    if config.CHECKPOINT_PATH:
        if not os.path.exists(config.CHECKPOINT_PATH):
            os.makedirs(config.CHECKPOINT_PATH)
            print("Output directory is created")

    # --------------------------------------
    # Initialize Model
    # --------------------------------------

    model = RhythmNet()

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")

    if config.CHECKPOINT_PATH:
        checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT_PATH)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Output directory is created")

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    # loss_fn = nn.L1Loss()
    loss_fn = RhythmNetLoss()

    testset = trainset = None

    # Initialize SummaryWriter object
    writer = SummaryWriter()

    # Read from a pre-made csv file that contains data divided into folds for cross validation
    folds_df = prep_rhythmnet_splits_V1(config.SAVE_CSV_PATH)
    FOLD = 1
    # Loop for enumerating through folds.
    print(f"Training for {config.EPOCHS} Epochs (each video)")
    # for k in folds_df['iteration'].unique():
    # for k in [1]:
    #     # Filter DF
    #     video_files_test = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'V')]
    #     video_files_train = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'T')]
    #
    #     # Get paths from filtered DF VIPL
    #     video_files_test = [os.path.join(config.ST_MAPS_PATH, video_path.split('/')[-1]) for video_path in
    #                         video_files_test["video"].values]
    #     video_files_train = [os.path.join(config.ST_MAPS_PATH, video_path.split('/')[-1]) for video_path in
    #                          video_files_train["video"].values]

    # video_files_test = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
    #                     video_files_test["video"].values]
    # video_files_train = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
    #                      video_files_train["video"].values]

    # video_files_train = video_files_train[:32]
    # video_files_test = video_files_test[:32]

    # print(f"Reading Current File: {video_files_train[0]}")

    # --------------------------------------
    # Build Dataloaders
    # --------------------------------------

    train_set = DataLoaderRhythmNet(df=folds_df[FOLD])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    print('\nTrain DataLoader constructed successfully!')

    # Code to use multiple GPUs (if available)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if config.DEVICE == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False
    model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path,
                                                                                      load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found! Training from beginning")

    # -----------------------------
    # Start training
    # -----------------------------

    train_loss_per_epoch = []
    for epoch in range(config.EPOCHS):
        target_hr_list, predicted_hr_list, train_loss = engine_vipl.train_fn(model, train_loader, optimizer, loss_fn)

        # Save model with final train loss (script to save the best weights?)
        if checkpointed_loss != 0.0:
            if train_loss < checkpointed_loss:
                save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
                checkpointed_loss = train_loss
            else:
                pass
        else:
            if len(train_loss_per_epoch) > 0:
                if train_loss < min(train_loss_per_epoch):
                    save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
            else:
                save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)

        metrics = compute_criteria(target_hr_list, predicted_hr_list)

        for metric in metrics.keys():
            writer.add_scalar(f"Train/{metric}", metrics[metric], epoch)

        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "HR_MAE : {:.3f} |".format(metrics["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)
              # "Pearsonr : {:.3f} |".format(metrics["Pearson"]), )

        train_loss_per_epoch.append(train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch+1)

        # Plots on tensorboard
        ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
        gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
        writer.add_image('BA_plot', ba_plot_image, epoch)
        writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)

    mean_loss = np.mean(train_loss_per_epoch)
    # Save the mean_loss value for each video instance to the writer
    print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")
    writer.flush()

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if config.DEVICE == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False
    model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer,
                                                                                      checkpoint_path,
                                                                                      load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found! Training from beginning")

    # -----------------------------
    # Start Validation
    # -----------------------------
    test_set = DataLoaderRhythmNet(df=folds_df[[f for f in range(len(folds_df)) if f != FOLD][0]])
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )
    print('\nEvaluation DataLoader constructed successfully!')

    print(f"Finished Training, Validating {len(folds_df[[f for f in range(len(folds_df)) if f != FOLD][0]])} video files for {config.EPOCHS_TEST} Epochs")

    eval_loss_per_epoch = []
    for epoch in range(config.EPOCHS_TEST):
        # validation
        target_hr_list, predicted_hr_list, test_loss = engine_vipl.eval_fn(model, test_loader, loss_fn)

        # truth_hr_list.append(target)
        # estimated_hr_list.append(predicted)
        metrics = compute_criteria(target_hr_list, predicted_hr_list)
        for metric in metrics.keys():
            writer.add_scalar(f"Test/{metric}", metrics[metric], epoch)

        print(f"\nFinished Test [Epoch: {epoch + 1}/{config.EPOCHS_TEST}]",
              "\nTest Loss: {:.3f} |".format(test_loss),
              "HR_MAE : {:.3f} |".format(metrics["MAE"]),
              "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)

        writer.add_scalar("Loss/test", test_loss, epoch)

        # Plots on tensorboard
        ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
        gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
        writer.add_image('BA_plot', ba_plot_image, epoch)
        writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)


    # print(f"Avg Validation Loss: {mean_test_loss} for {config.EPOCHS_TEST} epochs")
    writer.flush()
    # plot_train_test_curves(train_loss_data, test_loss_data, plot_path=config.PLOT_PATH, fold_tag=k)
    # Plots on the local storage.
    gt_vs_est(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH)
    bland_altman_plot(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH)
    writer.close()
    print("done")


if __name__ == '__main__':
    run_training()
