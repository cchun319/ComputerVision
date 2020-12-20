"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
architecture_config = [
    (7, 64, 2, 3), # 448 x 448 x 3 -> 224 x 224 x 64
    "M", # 224 x 224 x 64 -> 112 x 112 x 64
    (3, 192, 1, 1), # 112 x 112 x 64 -> 112 x 112 x 192
    "M", # 112 x 112 x 192 -> 56 x 56 x 192
    (1, 128, 1, 0), # 56 x 56 x 192 -> 56 X 56 x 128
    (3, 256, 1, 1), # 56 X 56 x 128 -> 56 X 56 x 256
    (1, 256, 1, 0), # 56 X 56 x 256 -> 56 X 56 x 256 
    (3, 512, 1, 1), # 56 X 56 x 256 -> 56 X 56 x 512
    "M", # 56 X 56 x 512 -> 28 x 28 x 512
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # 28 x 28 x 512 -> 28 X 28 X 512
    (1, 512, 1, 0), # 28 x 28 x 512 -> 28 X 28 X 512
    (3, 1024, 1, 1), # 28 x 28 x 512 -> 28 X 28 X 1024
    "M", # 28 x 28 x 1024 -> 14 X 14 X 1024
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], # 14 X 14 X 1024 -> 14 X 14 X 1024
    (3, 1024, 1, 1),
    (3, 1024, 2, 1), # 14 X 14 X 1024 -> 7 X 7 X 1024
    (3, 1024, 1, 1),
    (3, 1024, 1, 1), 
]

class CNNLayer(nn.Module):
  def __init__(self, in_channels, out_channels, bias = False, **kwargs):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
  def forward(self, x):
    return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
  def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

  def forward(self, x):
      return self.fcs(torch.flatten(self.darknet(x), start_dim=1))

  def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNLayer(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNLayer(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNLayer(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

  def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )