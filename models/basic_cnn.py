# -*- coding: UTF-8 -*-

"""A basic CNN

network structure: input -> [ conv -> (bn) -> relu -> pool ] -> [ ... ] -> ... -> fc -> output

"""


import torch
from torch import nn

def my_conv(input_channels, output_channels, is_bn=False, conv_mode='valid'):
    assert conv_mode in ['same']
    conv_layer = nn.Sequential()
    if conv_mode == 'same':
        conv_layer.add_module('conv2d', nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1))
    if is_bn:
        conv_layer.add_module('bn2d', nn.BatchNorm2d(output_channels))
    conv_layer.add_module('act', nn.ReLU(True))
    return conv_layer

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        # get parameters
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        conv_layers = config['conv_layers']
        feature_nums = config['feature_nums']
        is_bn = config['is_bn']
        conv_mode = config['conv_mode']
        
        # construct the convolutional layers and max pooling layers
        assert conv_layers == len(feature_nums)
        conv_i = None
        for i in range(conv_layers):
            if i == 0:
                conv_i = [my_conv(input_shape[1], feature_nums[i], is_bn=is_bn, conv_mode=conv_mode), nn.MaxPool2d(2)]
            else:
                conv_i += [my_conv(feature_nums[i - 1], feature_nums[i], is_bn=is_bn, conv_mode=conv_mode), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*conv_i)
    
        # compute conv feature size for the final fully connected layer
        with torch.no_grad():
            self.feature_size = self.conv(torch.zeros(*input_shape)).view(-1).shape[0]
        
        # construct the final fully connected layer
        self.fc = nn.Linear(self.feature_size, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


