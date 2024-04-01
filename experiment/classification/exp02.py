from collections import defaultdict
import os
import numpy as np
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from loader.classification import prep_split, get_data_loaders, get_stl_data_loaders
from models.classification.t3D_CNN import C3DInspired, ResNet3D
from utils.test import save_cm, test_model


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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # aggergate the loss
        running_loss += loss.item()
        output_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_labels)  # Save Prediction
        y_true.extend(labels.data.cpu().numpy())  # Save Truth
    return y_pred, y_true, running_loss/len(loader)  # average loss per batch


def train(model: torch.nn.Module, root_loc: str, n_epochs: int = 100, board_loc: str = "C3D", fold: int = 1,
          model_name: str = "C3D.pth", lr: float = None, class_encodings: dict = {}, stl_map: dict = {}):

    os.makedirs(os.path.join(os.path.dirname(root_loc), "runs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(os.path.dirname(root_loc), "runs", board_loc))
    train_cms_loc = os.path.join(os.path.dirname(root_loc), "runs", board_loc, "trainCM")
    os.makedirs(os.path.join(os.path.dirname(root_loc), "runs", board_loc, "trainCM"), exist_ok=True)

    if fold == 1:
        train_df, test_df = prep_split(root_loc=root_loc)
    else:
        test_df, train_df = prep_split(root_loc=root_loc)

    # reduce the training set for fast training
    ds = np.unique(train_df["Dataset"].values.tolist())
    train_c = []
    for d in ds:
        d_df = train_df[train_df["Dataset"] == d]
        if len(d_df) > 150:
            train_c.append(d_df[:150])
        else:
            train_c.append(d_df)
    train_df = pd.concat(train_c)
    print(f"Number of training samples: {len(train_df)}")

    train_loader, test_loader = get_stl_data_loaders(train_df=train_df, test_df=test_df,
                                                     encodings=class_encodings, stl_map=stl_map)
    classes = list(class_encodings.values())
    classes.sort()  # encoded - classes are in sorted order

    dummy_input = torch.zeros(1, 3, 10, 224, 224)
    writer.add_graph(model, dummy_input)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model.to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = lr if lr is not None else 0.01
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    best_epoch = -1
    best_train_acc = None

    for epoch in range(n_epochs):
        model.train(True)
        train_preds, train_labels, train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, dev)
        train_acc = save_cm(y_pred=train_preds, y_true=train_labels, classes=classes,
                            img_path=os.path.join(train_cms_loc, f"epoch_{epoch+1}.png"))
        # we do not need gradients for validation
        model.train(False)

        if best_train_acc is None:
            best_epoch = epoch
            best_train_acc = train_acc
        elif train_acc > best_train_acc:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(root_loc), "runs", board_loc, model_name))
            best_epoch = epoch
            best_train_acc = train_acc

        print(f"Epoch: {epoch}, Training Loss: {train_loss}, Training Accuracy: {train_acc}")
        writer.add_scalars(f"Training vs Validation Loss ({board_loc})",
                           {"train": train_loss},
                           epoch+1)
        writer.add_scalars(f"Training vs Validation Accuracy ({board_loc})",
                           {"train": train_acc},
                           epoch+1)

        # writer.add_image("Training CM", os.path.join(train_cms_loc, f"epoch_{epoch + 1}.png"), epoch+1)
        # writer.add_image("Validation CM", os.path.join(val_cms_loc, f"epoch_{epoch + 1}.png"), epoch + 1)

    print(f"Best Epoch: {best_epoch}, Best Training Accuracy: {best_train_acc*100} %")


if __name__ == "__main__":
    root_loc = r"D:\anshul\remoteHR\DataBias\Classification"
    class_encodings = {
        "COHFACE": 0,
        "VIPL-V1": 1,
        "VIPL-V2": 2,
        "MANHOB": 3,
    }
    n_classes = len(class_encodings.keys())
    board_loc = "C3DInspired_STL"
    model_name = "C3D_Inspired_STL.pth"
    oversample = False
    model = C3DInspired(n_classes=n_classes)  # ResNet3D(depth=18, num_classes=4)
    FOLD = 1  # 2
    #
    # stl_map = {
    #     "th": 300,
    #     "group_clip_size": 15,
    #     "frames_dim": (224, 224)
    # }
    # train(model=model, root_loc=root_loc, n_epochs=10, board_loc=board_loc, fold=FOLD,
    #       model_name=model_name, lr=1e-2, class_encodings=class_encodings, stl_map=stl_map)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(root_loc), "runs", board_loc, model_name)))

    if FOLD == 1:
        train_df, test_df = prep_split(root_loc=root_loc)
    else:
        test_df, train_df = prep_split(root_loc=root_loc)

    _, test_loader = get_stl_data_loaders(train_df=train_df, test_df=test_df,
                                          encodings=class_encodings, stl_map=stl_map)
    classes = list(class_encodings.values())
    classes.sort()  # encoded - classes are in sorted order

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_acc = test_model(model=model, loader=test_loader,
                          img_path=os.path.join(os.path.dirname(root_loc), "runs", board_loc, "test.png"),
                          classes=classes, dev=dev)

    print(f"Test accuracy of the best model is {test_acc*100} %")
    # model = C3D(n_classes=n_classes)
    # train(model=model, root_loc=root_loc, n_epochs=50, board_loc="C3D_Weighted_Oversample", model_name="C3D_WOS.pth",
    #       lr=1e-4)
