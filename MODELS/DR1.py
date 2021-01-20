import torch.nn as nn
from .se_module import *

class Bottleneck_DR1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck_DR1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # change here
        self.se_b = SEBLayer(planes)
        self.se_a = ALayer_DR1_v1_light_v1(planes,stride)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        # conv2 is changed to DR1_light conv
        out2 = self.se_a(out1, self.conv2.weight) # apply HxW 9xCin to x
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2) # B
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_DR1_new(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck_DR1_new, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # change here
        # self.se_b = SEBLayer(planes)
        self.se_a = ALayer_A_v2_2(planes, stride)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        # conv2 is changed to DR1_light conv
        out2 = self.se_a(out1, self.conv2.weight) # apply HxW 9xCin to x
        out = self.bn2(out2)
        # out = self.se_b(out1, out2) # B
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_ADR1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck_ADR1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # change here
        # self.se_b = SEBLayer(planes)
        self.se_a = ALayer_ADR1(planes,stride)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        # conv2 is changed to DR1_light conv
        out2 = self.se_a(out1, self.conv2.weight) # apply HxW 9xCin to x
        out = self.bn2(out2)
        # out = self.se_b(out1, out2) # B
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out