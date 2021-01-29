import torch
import torch.nn as nn

class SuperNet(nn.Module):
    '''
    This network shows how to implement custom weight initialization.
    Takes in 224x224[x3] images as inputs.
    '''
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AvgPool2d((14, 14))

        self.fc_layers = nn.Sequential(
            nn.Linear(64*14*14, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(84, num_classes)
        )

        self.initialize_weights()

    def forward(self, X):
        X = self.conv_layers(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc_layers(X)
        return X

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
