#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
#import torch.nn as nn
#import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


def cost_volume_gen(fL, fR, shift, stride=1):
    bn, c, h, w = fL.shape
    cost = torch.FloatTensor(bn, c*2, shift,  h,  w).zero_().type_as(fL.data)
    for i in range(0, shift):
        idx = i*stride
        cost[:, :c, i, :, idx:] = fL[..., idx:]
        cost[:, c:, i, :, idx:] = fR[..., :w-idx]
    return cost.contiguous()

def cost_volume_gen1(fL, fR, shift, stride=1):
    
    bn, c, h, w = fL.shape
    cost = torch.FloatTensor(bn, c*2, shift, h, w).zero_().type_as(fL.data)
    cost[:, :c] = fL.unsqueeze(2).expand_as(cost[:, :c])
    for i in range(0, shift):
        idx = i*stride
        cost[:, c:, i, :, idx:] = fR[..., :w-idx]
    return cost.contiguous()

if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    # prepare data
    fL = torch.rand(2, 16, 64, 64).cuda()
    fR = torch.rand(2, 16, 64, 64).cuda()
    shift, stride = 8, 2
    logger.info('Shape of fL: \n %s \n' % str(fL.shape) )

    logger.info('Shape of cost_volume_gen(fL, fR, %d, %d): \n %s \n' % 
                (shift, stride, str(cost_volume_gen(fL, fR, shift, stride).shape)) )
    
    logger.info('Shape of cost_volume_gen1(fL, fR, %d, %d): \n %s \n' % 
                (shift, stride, str(cost_volume_gen1(fL, fR, shift, stride).shape)) )
    
    
