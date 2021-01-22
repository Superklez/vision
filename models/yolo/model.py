import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, True)

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.lrelu(X)
        return X

class YOLOv1(nn.Module):

    def __init__(self, cfg=cfg, in_channels=3, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.darknet = self.make_conv_layers(cfg)
        self.fc_layers = self.make_fc_layers(*args, **kwargs)

    def forward(self, X):
        X = self.darknet(X)
        X = torch.flatten(X, 1)
        X = self.fc_layers(X)
        return X

    def make_conv_layers(self, cfg):
        layers = []
        nc_in = self.in_channels

        for l in cfg:
            if type(l) == tuple:
                nc_out, ks, s, p = l
                layers.append(CNNBlock(nc_in, nc_out, kernel_size=ks, stride=s, padding=p))
                nc_in = nc_out

            elif l == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif type(l) == list:
                for i in range(l[-1]):
                    for nc_out, ks, s, p in l[:-1]:
                        layers.append(CNNBlock(nc_in, nc_out, kernel_size=ks, stride=s, padding=p))
                        nc_in = nc_out

        return nn.Sequential(*layers)


    def make_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        block = nn.Sequential(
            nn.Linear(1024*S*S, 496), # originally 4096
            nn.Dropout(),
            nn.LeakyReLU(0.1, True),
            nn.Linear(496, S*S*(C+B*5))
        )
        return block
