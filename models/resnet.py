import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, X):
        identity = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)

        if self.downsample is not None:
            identity = self.downsample(X)

        X += identity
        X = self.relu(X)
        return X

class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        identity = X

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)

        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu(X)

        X = self.conv3(X)
        X = self.bn3(X)

        if self.downsample is not None:
            identity = self.downsample(X)

        X += identity
        X = self.relu(X)

        return X

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_channels=3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self.make_layers(block, layers[0], 64, stride=1)
        self.layer2 = self.make_layers(block, layers[1], 128, stride=2)
        self.layer3 = self.make_layers(block, layers[2], 256, stride=2)
        self.layer4 = self.make_layers(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)

        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)

        return X

    def make_layers(self, block, num_blocks, out_channels, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride),
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
