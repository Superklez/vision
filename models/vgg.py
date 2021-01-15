import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    '''
    Takes in 224x224[x3] images as inputs.
    https://arxiv.org/pdf/1409.1556.pdf
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    '''
    def __init__(self, cfg, num_classes=1000, batch_norm=False):
        super().__init__()
        self.conv_layers = self.make_layers(cfg, batch_norm)
        self.fc_layers = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes)
        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        input_channels = 3

        for l in cfg:
            if l == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                l = int(l)
                layers.append(nn.Conv2d(input_channels, l, kernel_size=3, padding=1))

                if batch_norm:
                    layers.append(nn.BatchNorm2d(num_features=l))

                layers.append(nn.ReLU(inplace=True))
                input_channels = l

        return nn.Sequential(*layers)

    def forward(self, X, log_softmax=False):
        X = self.conv_layers(X)
        X = self.fc_layers(X)
        if log_softmax:
            X = F.log_softmax(X, dim=1)
        return X

cfgs = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def VGG11(bn=False):
    model = VGG(cfgs['A'], batch_norm=bn)
    return model

def VGG13(bn=False):
    model = VGG(cfgs['B'], batch_norm=bn)
    return model

def VGG16(bn=False):
    model = VGG(cfgs['D'], batch_norm=bn)
    return model

def VGG19(bn=False):
    model = VGG(cfgs['E'], batch_norm=bn)
    return model
