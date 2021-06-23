import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    '''
    Takes in 32x32[x1] images as inputs.
    http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
    '''
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=16),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.BatchNorm1d(num_features=120),
            nn.ReLU(inplace=True),

            nn.Linear(120, 84),
            nn.BatchNorm1d(num_features=84),
            nn.ReLU(inplace=True),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x
