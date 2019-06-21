#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
PSMnet 中使用的匹配损失计算模块
'''

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import math
#import numpy as np


import logging
logger = logging.getLogger(__name__)
fun_active3d = nn.ReLU(inplace=True) # nn.LeakyReLU(negative_slope=0.1, inplace=True)

def conv3d3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 Conv3d[no bias] with padding and dilation"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def deconv3d3x3(in_planes, out_planes, stride=2):
    """3x3 ConvTranspose3d[no bias] with padding and dilation"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=1, bias=False)


class Hourglass(nn.Module):
    def __init__(self, planes):
        super(Hourglass, self).__init__()

        self.conv1 = nn.Sequential(conv3d3x3(planes*1, planes*2, stride=2), nn.BatchNorm3d(planes*2), fun_active3d, )
        self.conv2 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=1), nn.BatchNorm3d(planes*2), )
        
        self.conv3 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=2), nn.BatchNorm3d(planes*2), fun_active3d, )
        self.conv4 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=1), nn.BatchNorm3d(planes*2), fun_active3d, )

        self.conv5 = nn.Sequential(deconv3d3x3(planes*2, planes*2, stride=2), nn.BatchNorm3d(planes*2), ) # +conv2
        self.conv6 = nn.Sequential(deconv3d3x3(planes*2, planes*1, stride=2), nn.BatchNorm3d(planes*1), ) # +x

        self.relu = fun_active3d

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        _, _, d1, h1, w1 = x.shape # 1/4
        _, _, d2, h2, w2 = out.shape # 1/8
        
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None: 
            pre = pre + postsqu
        pre = self.relu(pre)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        post = self.conv5(out)[..., :d2, :h2, :w2] #in:1/16 out:1/8
        if presqu is not None:
            post = post + presqu
        else: 
            post = post + pre
        post = self.relu(post)
        logger.debug('self.conv5: %s\n shape of input and output: %s, %s\n' %(
                        str(self.conv5), str(out.shape), str(post.shape) ) )
        assert out.shape[-1]*2 < w2+2
        

        out  = self.conv6(post)[..., :d1, :h1, :w1]  #in:1/8 out:1/4
        logger.debug('self.conv6: %s\n shape of input and output: %s, %s\n' %(
                        str(self.conv6), str(post.shape), str(out.shape) ) )
        assert post.shape[-1]*2 < w1+2

        return out, pre, post


class CostCompute(nn.Module):
    def __init__(self, inplanes=64, planes=32, C=1):
        super(CostCompute, self).__init__()

        self.fun_active = fun_active3d
        self.dres0 = self.dres0_create(inplanes, planes)
        self.dres1 = self.dres_create(planes)

        self.dres2 = Hourglass(planes)
        self.dres3 = Hourglass(planes)
        self.dres4 = Hourglass(planes)

        self.classif1 = self.classify_create(planes, C)
        self.classif2 = self.classify_create(planes, C)
        self.classif3 = self.classify_create(planes, C)

    def dres0_create(self, inplanes, planes):
        return nn.Sequential(
                    conv3d3x3(inplanes, planes, stride=1), nn.BatchNorm3d(planes), fun_active3d, 
                    conv3d3x3(planes  , planes, stride=1), nn.BatchNorm3d(planes), fun_active3d, 
                    )

    def dres_create(self, planes):
        return nn.Sequential(
                    conv3d3x3(planes, planes, stride=1), nn.BatchNorm3d(planes), fun_active3d, 
                    conv3d3x3(planes, planes, stride=1), nn.BatchNorm3d(planes), 
                    ) 

    def classify_create(self, planes, C=1):
        return nn.Sequential(
                    conv3d3x3(planes, planes, stride=1), nn.BatchNorm3d(planes), fun_active3d, 
                    conv3d3x3(planes, C     , stride=1),  
                    ) 

    def forward(self, cost):

        cost = self.dres0(cost)
        cost = self.dres1(cost) + cost

        out1, pre1, post1 = self.dres2(cost, None, None) 
        out1 = out1 + cost

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2 + cost

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3 + cost

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        bn, c, d, h, w = cost3.shape

        costs = [cost1, cost2, cost3] if self.training else [cost3]
        if(c > 1):
            costs = [tcost.transpose(1, 2).reshape(bn, 1, c*d, h, w).contiguous() for tcost in costs]
        
        return costs


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    # prepare data
    fcost = torch.rand(1, 64, 20, 65, 65).cuda()
    costs = CostCompute().cuda().eval()(fcost)

    logger.info('shape of input: %s' % str(fcost.shape))
    logger.info('shape of output: %s\n' % str([cost.shape for cost in costs]))
