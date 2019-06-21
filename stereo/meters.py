#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

# Stereo Meters
class StereoMeters(object):

    def __init__(self):
        
        self._data = defaultdict(lambda:[])

    def append(self, dict_meters):
        
        for k, v in dict_meters.items():
            self._data[k].append(v)

    def extend(self, dict_datas):
        
        for k, v in dict_datas.items():
            self._data[k].extend(v)

    @property
    def data(self):
        return {k:v for k, v in self._data.items()}

# creat meter
def CreateMeter(mtype='avg', **kargs):
    
    if('avg' == mtype):
        return AverageMeter(**kargs)
        
    elif('smooth' == mtype):
        return SmoothMeter(**kargs)
        
    else:
        logger.info('Not Support The Meter[%s] !'%mtype)

# SmoothMeter
class SmoothMeter(object):
    """
    计算并存储参数当前值和延迟平滑值
    Computes and stores the current value and the delay smooth value
    """
    def __init__(self, delay=0.99):
        
        self.reset()
        self.delay = delay

    def reset(self):
        
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        
        self.val = val
        self.count += n
        if(self.count > 20):
            delay = self.delay**n
            self.avg = self.avg*delay + val*(1-delay)
        else:
            self.sum += val * n
            self.avg = self.sum / self.count

# AverageMeter
class AverageMeter(object):
    """
    计算并存储参数当前值和平均值
    Computes and stores the average and current value
    """
    def __init__(self):
        
        self.reset()

    def reset(self):
        
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count



