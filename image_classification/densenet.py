import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DenseLayer(nn.Module):
    def __init__(self, in_channels:int, growth_rate:int, bn_channels:int):
        super(DenseLayer, self).__init__()
        if bn_channels > 0:
            self.dense_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, bn_channels, kernel_size=(1, 1),
                    stride=(1, 1), padding=(0, 0), bias=False),

                nn.BatchNorm2d(bn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(bn_channels, growth_rate, kernel_size=(3, 3),
                    stride=(1, 1), padding=(1, 1), bias=False)
            )
        elif bn_channels == 0:
            self.dense_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=(3, 3),
                    stride=(1, 1), padding=(1, 1), bias=False)
            )

    def forward(self, x):
        return torch.cat((x, self.dense_layer(x)), dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bn_channels):
        super(DenseBlock, self).__init__()
        dense_block = []
        for l in range(num_layers):
            in_features = in_channels + l * growth_rate
            dense_block.append(DenseLayer(in_features, growth_rate,
                bn_channels))
        self.dense_block = nn.Sequential(*dense_block)
      
    def forward(self, x):
        for dense_layer in self.dense_block:
            x = dense_layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels:int, compression_factor:float=0.5):
        super(TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, int(in_channels * compression_factor),
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        return self.transition_layer(x)

class DenseNet(nn.Module):
    '''
    Takes in 112x112[x3] images as inputs.
    https://arxiv.org/pdf/1608.06993v5.pdf
    '''
    def __init__(self, in_channels:int, num_classes:int, cfg:list,
            init_features:int, growth_rate:int, bn_channels:int,
            compression_factor:float):
        super(DenseNet, self).__init__()
        features = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, init_features, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        ]
        in_channels = init_features
        for i, num_layers in enumerate(cfg):
            features.append(DenseBlock(in_channels, num_layers, growth_rate,
                bn_channels))
            in_channels = in_channels + num_layers * growth_rate
            if i != len(cfg) - 1:
                features.append(TransitionLayer(in_channels, 
                    compression_factor))
                in_channels = int(in_channels * compression_factor)

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cfgs = {
    'A' : [6, 12, 24, 16],
    'B' : [6, 12, 32, 32],
    'C' : [6, 12, 48, 32],
    'D' : [6, 12, 64, 48]
}

def DenseNet_121(in_channels:int, num_classes:int, cfg=cfgs['A'],
    init_features:int=64, growth_rate:int=32, bn_channels:int=128,
    compression_factor:float=0.5):
    model = DenseNet(in_channels, num_classes, cfg, init_features, growth_rate,
        bn_channels, compression_factor)
    return model

def DenseNet_169(in_channels:int, num_classes:int, cfg=cfgs['B'],
    init_features:int=64, growth_rate:int=32, bn_channels:int=128,
    compression_factor:float=0.5):
    model = DenseNet(in_channels, num_classes, cfg, init_features, growth_rate,
        bn_channels, compression_factor)
    return model

def DenseNet_201(in_channels:int, num_classes:int, cfg=cfgs['C'],
    init_features:int=64, growth_rate:int=32, bn_channels:int=128,
    compression_factor:float=0.5):
    model = DenseNet(in_channels, num_classes, cfg, init_features, growth_rate,
        bn_channels, compression_factor)
    return model
 
def DenseNet_264(in_channels:int, num_classes:int, cfg=cfgs['D'],
    init_features:int=64, growth_rate:int=32, bn_channels:int=128,
    compression_factor:float=0.5):
    model = DenseNet(in_channels, num_classes, cfg, init_features, growth_rate,
        bn_channels, compression_factor)
    return model