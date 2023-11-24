import os
import sys
import torch
import torch.nn as nn
from torchsummary import summary

sys.path.append(r"C:\Users\transponster\Documents\anshul\rPPG")
from models.classification.resnet_model import generate_model


def load_pretrained_model(model: nn.Module, pretrain_path: str, n_finetune_classes: int):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                 n_finetune_classes)

    return model


class ResNet3D(nn.Module):
    def __init__(self, depth: int, num_classes: int = 4,
                 weights_loc: str = "C:\\Users\\transponster\\Documents\\anshul\\task\\ft_weights"):
        super(ResNet3D, self).__init__()
        assert depth in [18, 34, 50]

        # model = generate_model(model_depth=depth, sample_size=224, sample_duration=30)
        weights_file = [f for f in os.listdir(weights_loc) if f"{depth}-kinetics.pth" in f]
        # model.load_state_dict(torch.load(os.path.join(weights_loc, weights_file[0])))

        initial_classes = 700  # kinemetics dataset
        model = generate_model(model_depth=depth, n_classes=initial_classes,
                               n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                               no_max_pool=False, shortcut_type='B', widen_factor=1.0,)
        model = load_pretrained_model(model=model, pretrain_path=os.path.join(weights_loc, weights_file[0]),
                                      n_finetune_classes=num_classes)
        self.num_classes = num_classes
        self.model = model

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        logits = self.model(x)

        # logits = self.fc(x)
        probs = torch.log_softmax(logits, dim=1)
        return probs


class C3DInspired(nn.Module):
    """
        Inspired from original convolutional network, not reducing the number of frames by great extent
        picking 1 of every 2 frame in the video.

        The input videos are 30 frames/second and with this down-sampling it becomes 1 frame per second

        Inspired by, ref: https://github.com/DavideA/c3d-pytorch
    """

    def __init__(self, n_classes: int = 4):
        super(C3DInspired, self).__init__()
        '''
            (N, C, D, H, W) is the shape of the input, 
            N - batch size, 
            C - channel size (if RGB then 3),
            D - depth of sample (number of frames),
            H - height of frame in a video
            W - width of frame in a video

        '''
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(num_features=64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(num_features=128)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(2, 1, 1))
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(2, 1, 1))
        self.bn4a = nn.BatchNorm3d(num_features=512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(2, 1, 1))
        self.bn4b = nn.BatchNorm3d(num_features=512)
        self.pool4 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(2, 1, 1))
        self.bn5a = nn.BatchNorm3d(num_features=512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(2, 1, 1))
        self.bn5b = nn.BatchNorm3d(num_features=512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.conv6 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(num_features=512)
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3), padding=(0, 1, 1))

        # self.adaptive_pool = nn.AdaptiveAvgPool3d(output_size=(3, 1, 1))

        self.fc = nn.Linear(4608, n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.pool1(x)

        x = self.bn2(self.relu(self.conv2(x)))
        x = self.pool2(x)

        x = self.bn3a(self.relu(self.conv3a(x)))
        x = self.bn3b(self.relu(self.conv3b(x)))
        x = self.pool3(x)

        x = self.bn4a(self.relu(self.conv4a(x)))
        x = self.bn4b(self.relu(self.conv4b(x)))
        x = self.pool4(x)

        x = self.bn5a(self.relu(self.conv5a(x)))
        x = self.bn5b(self.relu(self.conv5b(x)))
        x = self.pool5(x)

        x = self.bn6(self.relu(self.conv6(x)))
        x = self.pool6(x)

        # print(f"After Encoding: {x.shape}")
        # x = self.adaptive_pool(x)
        x = x.view(-1, 4608)

        logits = self.fc(x)
        probs = torch.log_softmax(logits, dim=1)
        return probs


if __name__ == "__main__":
    # model = C3DInspired(n_classes=4)
    # summary(model, (3, 30, 224, 224), batch_size=-1)

    model = ResNet3D(depth=18, num_classes=4)
    summary(model, (3, 30, 224, 224), batch_size=-1)
