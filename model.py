import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, 2)
        self.conv3_128 = self.__make_layer(128, 2)
        self.conv3_256 = self.__make_layer(256, 4)
        self.conv3_512a = self.__make_layer(512, 4)
        self.conv3_512b = self.__make_layer(512, 4)
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        return out

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)
