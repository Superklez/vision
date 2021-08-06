import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion,
            kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion,
            kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_channels=3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7,
            stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = _make_layers(block, layers[0], 64, stride=1)
        self.layer2 = _make_layers(block, layers[1], 128, stride=2)
        self.layer3 = _make_layers(block, layers[2], 256, stride=2)
        self.layer4 = _make_layers(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _make_layers(self, block, num_blocks, out_channels, stride=1):
    downsample = None
    layers = []

    if stride != 1 or self.in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels*block.expansion,
                kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels*block.expansion)
        )

    layers.append(block(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels * block.expansion

    for i in range(1, num_blocks):
        layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)

def ResNet18(num_classes=1000, img_channels=3):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, img_channels)
    return model

def ResNet34(num_classes=1000, img_channels=3):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, img_channels)
    return model

def ResNet50(num_classes=1000, img_channels=3):
    model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes, img_channels)
    return model

def ResNet101(num_classes=1000, img_channels=3):
    model = ResNet(BottleNeck, [3, 4, 23, 3], num_classes, img_channels)
    return model

def ResNet152(num_classes=1000, img_channels=3):
    model = ResNet(BottleNeck, [3, 8, 36, 3], num_classes, img_channels)
    return model
