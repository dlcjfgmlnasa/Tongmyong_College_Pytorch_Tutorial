# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class DNNMnistClassify(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims = 576
        output_dims = 10
        self.linear = nn.Sequential(
            nn.Linear(input_dims, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100, output_dims)
        )

    def forward(self, inputs):
        _, c, w, h = inputs.shape
        inputs = inputs.reshape(-1, c*w*h)
        output = self.linear(inputs)
        output = torch.softmax(output, dim=-1)
        return output


class CNNMnistClassify(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, inputs):
        cnn_output = self.cnn(inputs)
        b = cnn_output.size(0)
        cnn_output_reshape = cnn_output.reshape(b, -1)
        output = self.fc(cnn_output_reshape)
        output = torch.softmax(output, dim=-1)
        return output


cfg = {
    'VGG_D': [64,     'M', 128,      'M', 256, 256,           'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_type, batch_norm=True):
        super().__init__()
        self.vgg_type = vgg_type
        self.batch_norm = batch_norm
        self.cnn = self.cnn_layers()
        self.fc = nn.Sequential(
            nn.Linear(2304, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 10)
        )

    def forward(self, inputs):
        cnn_output = self.cnn(inputs)
        x = cnn_output.view(cnn_output.size(0), -1)
        output = self.fc(x)
        output = torch.softmax(output, dim=-1)
        return output

    def cnn_layers(self):
        layers = []

        input_channel = 1
        for layer in cfg[self.vgg_type]:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, layer, kernel_size=(3, 3), padding=1)]

            if self.batch_norm:
                layers += [nn.BatchNorm2d(layer)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = layer

        return nn.Sequential(*layers)