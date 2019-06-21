#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
PSMnet 中使用的特征提取模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
#import math
#import numpy as np


import logging
logger = logging.getLogger(__name__)
fun_active2d = nn.ReLU(inplace=True) # nn.LeakyReLU(negative_slope=0.1, inplace=True)

def conv2d3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 Conv2d[no bias] with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def downsample2d_res(in_planes, out_planes, stride):
    downsample = None
    if stride != 1 or in_planes != out_planes:
       downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
                )
    return downsample

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock2d, self).__init__()
        
        self.conv1 = conv2d3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv2d3x3(planes, planes, 1, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.relu = fun_active2d # nn.ReLU(inplace=True)
        out_planes = planes*BasicBlock2d.expansion
        self.downsample = downsample2d_res(inplanes, out_planes, stride)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x) if(self.downsample) else x

        return out


class ResBlock2d(nn.Module):
    
    def __init__(self, block, blocks, inplanes, planes, stride=1, dilation=1):
        super(ResBlock2d, self).__init__()

        layers = [block(inplanes, planes, stride, dilation)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, 1, dilation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.layers(x)

class FeatureExtraction(nn.Module):
    def __init__(self, planes=32):
        super(FeatureExtraction, self).__init__()
        self.firstconv = nn.Sequential(
                            conv2d3x3(3     , planes, stride=2), nn.BatchNorm2d(planes), fun_active2d, 
                            conv2d3x3(planes, planes, stride=1), nn.BatchNorm2d(planes), fun_active2d, 
                            conv2d3x3(planes, planes, stride=1), nn.BatchNorm2d(planes), fun_active2d, 
                            )

        block = BasicBlock2d
        k = block.expansion
        self.layer1 = ResBlock2d(block, 3 , planes*1  , planes*1, stride=1, dilation=1)
        self.layer2 = ResBlock2d(block, 16, planes*1*k, planes*2, stride=2, dilation=1)
        self.layer3 = ResBlock2d(block, 3 , planes*2*k, planes*4, stride=1, dilation=2)
        self.layer4 = ResBlock2d(block, 3 , planes*4*k, planes*4, stride=1, dilation=2)

        self.kernels_size = [8, 16, 32, 64]
        self.branchs = nn.ModuleList([self.branch_create(planes*4, planes) for i in range(4)])

        self.lastconv = nn.Sequential(
                            conv2d3x3(planes*10, planes*4, stride=1), nn.BatchNorm2d(planes*4), fun_active2d,
                            conv2d3x3(planes*4 , planes*1, stride=1), 
                            )

    def branch_create(self, inplanes=128, outplanes=32):
        return nn.Sequential(
                                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False), 
                                nn.BatchNorm2d(outplanes), fun_active2d, 
                                )

    def forward(self, x):
        
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        branchs = [output_raw, output_skip]
        h, w = output_skip.shape[-2:]
        for i in range(4):
            kernel_size = self.kernels_size[i]
            branch = F.avg_pool2d(output_skip, kernel_size, padding=kernel_size//2)
            branch = self.branchs[i](branch)
            branch = F.upsample(branch, (h, w), mode='bilinear', align_corners=True)
            branchs.append(branch)

        output_feature = torch.cat(branchs, 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    # prepare data
    img = torch.rand(2, 3, 257, 257).cuda()
    feature = FeatureExtraction().cuda().eval()(img)

    logger.info('shape of input: %s' % str(img.shape))
    logger.info('shape of output: %s\n' % str(feature.shape))
