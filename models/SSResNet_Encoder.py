# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride, is_bn, is_dropout, p):
        super(BasicBlock, self).__init__()

        bn1 = None
        drop1 = None
        conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        if is_bn:
            bn1 = nn.BatchNorm2d(out_channels)
        # act1 = nn.ReLU(True)
        act1 = nn.LeakyReLU(0.2, inplace=True)
        if is_dropout:
            drop1 = nn.Dropout2d(p)
        layer1 = filter(lambda x: x is not None, [conv1, bn1, act1, drop1])
        self.layer1 = nn.Sequential(*layer1)

        bn2 = None
        conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        if is_bn:
            bn2 = nn.BatchNorm2d(out_channels)
        layer2 = filter(lambda x: x is not None, [conv2, bn2])
        self.layer2 = nn.Sequential(*layer2)

        # self.act2 = nn.ReLU(True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.drop2 = None
        if is_dropout:
            self.drop2 = nn.Dropout2d(p)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        # y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # y = self.bn2(self.conv2(y))
        # y += self.shortcut(x)
        # y = F.relu(y, inplace=True)  # apply ReLU after addition
        y = self.layer1(x)
        y = self.layer2(y)
        y += self.shortcut(x)
        y = self.act2(y)
        if self.drop2 is not None:
            y = self.drop2(y)
        return y

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input
#         # when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super(ResNetEncoder, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        channels = config['channels']
        blocks = config['blocks']
        is_bn = config['is_bn']
        is_dropout = config['is_dropout']
        p = config['p']

        # base_channels = config['base_channels']
        # block_type = config['block_type']
        # depth = config['depth']
        # assert block_type in ['basic', 'bottleneck']
        # if block_type == 'basic':
        #     block = BasicBlock
        #     n_blocks_per_stage = (depth - 2) // 6
        #     assert n_blocks_per_stage * 6 + 2 == depth
        # else:
        #     block = BottleneckBlock
        #     n_blocks_per_stage = (depth - 2) // 9
        #     assert n_blocks_per_stage * 9 + 2 == depth

        # n_channels = [
        #     base_channels,
        #     base_channels * 2 * block.expansion,
        #     base_channels * 4 * block.expansion
        # ]

        self.conv = nn.Conv2d(
            input_shape[1],
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

        # self.stage1 = self._make_stage(
        #     channels[0], channels[0], blocks[0], BasicBlock, stride=1,
        #     is_bn=is_bn, s_dropout=is_dropout, p=p)
        # self.stage2 = self._make_stage(
        #     channels[0], channels[1], blocks[1], BasicBlock, stride=2,
        #     is_bn=is_bn, is_dropout=is_dropout, p=p)
        # self.stage3 = self._make_stage(
        #     channels[1], channels[2], blocks[2], BasicBlock, stride=2,
        #     is_bn=is_bn, is_dropout=is_dropout, p=p)

        self.blocks = self._make_stage(channels, channels, blocks, BasicBlock,
                                       stride=1, is_bn=is_bn,
                                       is_dropout=is_dropout, p=p)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    is_bn, is_dropout, p):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(block_name,
                                 block(in_channels,
                                       out_channels,
                                       stride=stride,
                                       is_bn=is_bn,
                                       is_dropout=is_dropout,
                                       p=p))
            else:
                stage.add_module(block_name,
                                 block(out_channels,
                                       out_channels,
                                       stride=1,
                                       is_bn=is_bn,
                                       is_dropout=is_dropout,
                                       p=p
                                       ))
        return stage

    def _forward_conv(self, x):
        # x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)
        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

    @property
    def name(self) -> str:
        return 'ResNet'
