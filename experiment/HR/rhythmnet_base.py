import os


import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from models.rhythmNet.resnet_base import ResNet
from loader.vipl import DatasetRhythmNet, prep_rhythmnet_splits_V1


def train_one_epoch(epoch: int, model: torch.nn.Module, loader: DataLoader,
                    optimizer: torch.optim, criterion: torch.nn.modules.loss,
                    dev: torch.cuda.device) -> (list, list, float):
    running_loss = 0.0
    y_true = []
    y_pred = []
    for i, data in tqdm(enumerate(loader), desc=f"Training epoch-{epoch}", total=len(loader)):
        # data is in format [input, labels, clip-path]
        input, labels = data
        input = input.to(dev)
        labels = labels.to(dev)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input)
        outputs_ = outputs.squeeze()
        loss = criterion(outputs_, labels)
        loss.backward()
        optimizer.step()
        # aggergate the loss
        running_loss += loss.item()
        # output_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs_.data.cpu().numpy())  # Save Prediction
        y_true.extend(labels.data.cpu().numpy())  # Save Truth
    return y_pred, y_true, running_loss/len(loader)  # average loss per batch


def test_(model: torch.nn.Module, loader: DataLoader,
          criterion: torch.nn.modules.loss, dev: torch.cuda.device) -> (list, list, float):
    model = model.to(dev)
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    for i, data in tqdm(enumerate(loader), desc=f"Testing", total=len(loader)):
        # data is in format [input, labels, clip-path]
        input, labels = data
        input = input.to(dev)
        labels = labels.to(dev)

        # forward + backward + optimize
        outputs = model(input)
        outputs_ = outputs.squeeze()
        # print(outputs_)
        loss = criterion(outputs_, labels)
        # aggergate the loss
        running_loss += loss.item()
        # output_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs_.data.cpu().numpy())  # Save Prediction
        y_true.extend(labels.data.cpu().numpy())  # Save Truth
    # print(mean_squared_error(y_pred, y_true))
    return y_pred, y_true, running_loss/len(loader)  # average loss per batch


def train(model: torch.nn.Module, root_loc: str, n_epochs: int = 100, board_loc: str = "res18_rhythmNet", fold: int = 1,
          model_name: str = "res18_rhythmNet.pth", lr: float = None):

    os.makedirs(os.path.join(os.path.dirname(root_loc), "runs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(os.path.dirname(root_loc), "runs", board_loc))
    os.makedirs(os.path.join(os.path.dirname(root_loc), "runs", board_loc, "trainCM"), exist_ok=True)

    if fold == 1:
        train_df, test_df = prep_rhythmnet_splits_V1(
            # maps_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps',
            labels_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
        )
    else:
        test_df, train_df = prep_rhythmnet_splits_V1(
            # maps_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps',
            labels_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
        )

    # reduce the training set for fast training
    training_data = DatasetRhythmNet(df_paths_labels=train_df)
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_data = DatasetRhythmNet(df_paths_labels=test_df)
    # test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    dummy_input = torch.zeros(1, 3, 15, 25)
    writer.add_graph(model, dummy_input)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model.to(dev)
    criterion = torch.nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = lr if lr is not None else 0.01
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    best_epoch = -1
    best_train_loss = None

    for epoch in range(n_epochs):
        model.train(True)
        train_preds, train_labels, train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, dev)

        # we do not need gradients for validation
        model.train(False)

        if best_train_loss is None:
            best_epoch = epoch
            best_train_loss = train_loss
        elif train_loss < best_train_loss:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(root_loc), "runs", board_loc, model_name))
            best_epoch = epoch
            best_train_loss = train_loss

        print(f"Epoch: {epoch}, Training Loss: {train_loss}")
        writer.add_scalars(f"Training vs Validation Loss ({board_loc})",
                           {"train": train_loss},
                           epoch+1)

        # writer.add_image("Training CM", os.path.join(train_cms_loc, f"epoch_{epoch + 1}.png"), epoch+1)
        # writer.add_image("Validation CM", os.path.join(val_cms_loc, f"epoch_{epoch + 1}.png"), epoch + 1)
    print(f"Best Epoch: {best_epoch}, Best Training Loss: {best_train_loss}")


if __name__ == "__main__":
    root_loc = r"D:\anshul\remoteHR\HR\Regression"
    board_loc = "res18_rhythmNet"
    model_name = "res18_rhythmNet.pth"
    oversample = False
    model = ResNet()  # ResNet3D(depth=18, num_classes=4)
    FOLD = 1  # 2
    #
    # stl_map = {
    #     "th": 300,
    #     "group_clip_size": 15,
    #     "frames_dim": (224, 224)
    # }
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(root_loc), "runs", board_loc, model_name)))
    # train(model=model, root_loc=root_loc, n_epochs=300, board_loc=board_loc, fold=FOLD,
    #       model_name=model_name, lr=1e-4)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(root_loc), "runs", board_loc, model_name)))

    # test
    if FOLD == 1:
        train_df, test_df = prep_rhythmnet_splits_V1(
            # maps_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps',
            labels_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
        )
    else:
        test_df, train_df = prep_rhythmnet_splits_V1(
            # maps_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\maps',
            labels_path=r'D:\anshul\remoteHR\st-maps\VIPL-V1\HR'
        )

    test_data = DatasetRhythmNet(df_paths_labels=test_df)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = torch.nn.MSELoss()

    y_true, y_pred, loss = test_(model, test_loader, criterion, dev)
    y_true_ = [y_t for y_t, y_p in zip(y_true, y_pred) if y_p < 200]
    y_pred_ = [y_p for y_t, y_p in zip(y_true, y_pred) if y_p < 200]

    print(f"Test Loss: {mean_squared_error(y_pred_, y_true_)}")
    df = pd.DataFrame()
    df["True"] = y_true_
    df["Predictions"] = y_pred_
    import matplotlib.pyplot as plt
    plt.scatter(y_true_, y_pred_)
    plt.plot([min(min(y_true_), min(y_pred_)) - 1, max(max(y_true_), max(y_pred_)) + 1],
             [min(min(y_true_), min(y_pred_)) - 1, max(max(y_true_), max(y_pred_)) + 1], 'k--')
    # plt.xlim([50, 150])
    # plt.ylim([50, 150])
    plt.show()
    # df.to_csv(os.path.join(os.path.dirname(root_loc), "test_predictions.csv"), index=False)
