import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    '''
    Takes in 224x224[x3] images as inputs.
    https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    '''
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(5*5*256, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, X, log_softmax=False):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        if log_softmax:
            X = F.log_softmax(X, dim=1)
        return X
