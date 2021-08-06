import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    '''
    Takes in 224x224[x3] images as inputs.
    https://arxiv.org/pdf/1409.1556.pdf
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''
    def __init__(self, in_channels:int, cfg:list, num_classes=1000,
        batch_norm:bool=False):
        super().__init__()
        self.features = self._make_layers(in_channels, cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, in_channels:int, cfg:list, batch_norm=False):
        layers = []

        for l in cfg:
            if l == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                layers.append(nn.Conv2d(in_channels, l, kernel_size=3,
                    padding=1))

                if batch_norm:
                    layers.append(nn.BatchNorm2d(num_features=l))

                layers.append(nn.ReLU(inplace=True))
                in_channels = l

        return nn.Sequential(*layers)

cfgs = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def VGG11(in_channels:int=3, num_classes:int=1000, batch_norm:bool=False):
    model = VGG(in_channels, cfgs['A'], num_classes, batch_norm)
    return model

def VGG13(in_channels:int=3, num_classes:int=1000, batch_norm:bool=False):
    model = VGG(in_channels, cfgs['B'], num_classes, batch_norm)
    return model

def VGG16(in_channels:int=3, num_classes:int=1000, batch_norm:bool=False):
    model = VGG(in_channels, cfgs['D'], num_classes, batch_norm)
    return model

def VGG19(in_channels:int=3, num_classes:int=1000, batch_norm:bool=False):
    model = VGG(in_channels, cfgs['E'], num_classes, batch_norm)
    return model
