#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import math

import logging
logger = logging.getLogger(__name__)


# weight init
def weight_init(modules):

    for m in modules:

        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


# weight init
def weight_init0(modules):

    for m in modules:
        
        if(hasattr(m, 'bias') and m.bias is not None):
            m.bias.data.zero_()
        
        if isinstance(m, nn.Linear):
            v = 1. / math.sqrt(m.out_features)
            m.weight.data.uniform_(-v, v)
        
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        
        elif isinstance(m, nn.ConvTranspose3d):
            w = make_bilinear_weights(m.in_channels, m.out_channels, m.kernel_size)
            m.weight.data = w.type_as(m.weight.data)
        elif isinstance(m, nn.ConvTranspose2d):
            w = make_bilinear_weights(m.in_channels, m.out_channels, m.kernel_size)
            m.weight.data = w.type_as(m.weight.data)
        elif isinstance(m, nn.ConvTranspose1d):
            w = make_bilinear_weights(m.in_channels, m.out_channels, m.kernel_size)
            m.weight.data = w.type_as(m.weight.data)
        
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)


# make bilinear weights for nn.ConvTranspose
def make_bilinear_weights(in_channels, out_channels, kernel_size):

    n = len(kernel_size)
    factor = [(kz+1)//2 for kz in kernel_size]
    center = [factor[i]-1.0 if kernel_size[i]%2==1 else factor[i]-0.5 for i in range(n)]

    if n==1:
        og = [np.ogrid[:kernel_size[0]]]
    elif n==2:
        og = np.ogrid[:kernel_size[0], :kernel_size[1]]
    elif n==3:
        og = np.ogrid[:kernel_size[0], :kernel_size[1], :kernel_size[2]]

    filts = [(1.0 - abs(og[i] - center[i])/factor[i]) for i in range(n)]
    filt = filts[0]
    for i in range(1, n):
        filt = filt * filts[i]
#    print(og, factor, center, filts, kernel_size, filt)

    filt = torch.from_numpy(filt)
    size = [in_channels, out_channels] + list(kernel_size)
    w = torch.zeros(*size)
    for i in range(in_channels):
        for j in range(out_channels):
            if i == j:
                w[i, j] = filt

    return w

if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    m = nn.Conv2d(1, 1, 3, 3)
    logger.info('Weight of before initializing: \n %s \n' % str(m.weight.data))
    weight_init(m.modules())
    logger.info('Weight of after initializing with weight_init: \n %s \n' % str(m.weight.data))
    weight_init0(m.modules())
    logger.info('Weight of after initializing with weight_init0: \n %s \n' % str(m.weight.data))

