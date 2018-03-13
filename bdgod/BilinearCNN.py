#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


'''vgg make layers'''


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3)]
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BilinearCNN(nn.Module):

    def __init__(self, num_classes=3):
        super(BilinearCNN, self).__init__()

        '''resnet50'''
        # resnet50
        # block = BasicBlock
        # layers = [2, 2, 2, 2]
        # self.inplanes = 64
        # self.features_A = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #               bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(block, 64, layers[0]),
        #     self._make_layer(block, 128, layers[1], stride=2),
        #     self._make_layer(block, 256, layers[2], stride=2),
        #     self._make_layer(block, 512, layers[3], stride=2),
        #     nn.AvgPool2d(7, stride=1),
        # )
        # for m in self.features_A.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # self.classif_alex = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(inplace=True),
        # )

        '''vgg16 bn =True'''
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features_A = make_layers(cfg, batch_norm=True)
        # self.classier_A =  nn.Sequential(
        #     nn.Linear(512 * 3 * 3, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        # )
        self._initialize_weights(self.features_A)
        # self._initialize_weights(self.classier_A)

        '''resnet18'''
        block = BasicBlock
        layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.features_B = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            nn.AvgPool2d(7, stride=1),
        )
        for m in self.features_B.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.classifier = nn.Sequential(
        #     nn.Bilinear(1024, 512, 256),
        #     nn.Linear(256, num_classes),
        # )
        self.class1 = nn.Bilinear(2048, 512, 256)
        self.class2 = nn.Linear(256, num_classes)

    '''resnet make layer'''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    '''vgg init '''

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x_A = x.clone()
        x_B = x.clone()
        # print x_A.size()
        # print x_B.size()
        x_A = self.features_A(x_A)
        # print x_A.size()
        x_A = x_A.view(x_A.size(0), -1)
        # x_A = self.classier_A(x_A)
        # x_A = self.classif_alex(x_A)
        # print x_A.size()
        # print x_B.size()
        x_B = self.features_B(x_B)
        x_B = x_B.view(x_B.size(0), -1)
        # print x_A.size()
        # print x_B.size()
        # x = self.classifier(x_A, x_B)
        x = self.class1(x_A, x_B)
        x = self.class2(x)
        return x


if __name__ == '__main__':
    bcnn = BilinearCNN(3)

    bcnn = bcnn.cuda()

    print bcnn
