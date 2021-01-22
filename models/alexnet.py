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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc_layers = nn.Sequential(
            nn.Dropout(),

            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, X):
        X = self.conv_layers(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc_layers(X)
        
        return X
