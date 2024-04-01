import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchsummary import summary


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.base = resnet18(wegiths=None)
        resnet = resnet18(weights=None)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        self.resnet_linear = nn.Linear(512, 1000)
        self.fc_regression = nn.Linear(1000, 1)

    def forward(self, st_map):
        x = self.resnet18(st_map)
        x = x.flatten(1)
        # print(x.shape)
        x = self.resnet_linear(x)
        x = self.fc_regression(x)
        return x


if __name__ == "__main__":
    model = ResNet()
    model = model.to(torch.device("cuda"))
    summary(model, (3, 15, 25))
