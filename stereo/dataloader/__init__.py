#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from torch.utils.data import DataLoader
from .dataset import dataset_by_name
from .ImageFloder import ImageFloder

def dataloader_by_name(names='k2015-tr, k2012-tr', roots='./kitti, ./kitti', 
                        bn=1, training=False, crop_size=None, n=4):
    
    names = names.split(',')
    roots = roots.split(',')
    assert 1==len(roots) or len(names) == len(roots)
    
    datasets = [dataset_by_name(names[i], roots[i%len(roots)]) for i in range(len(names))]
    tImageFloder = ImageFloder(datasets, n, training, crop_size)
    if(tImageFloder.crop_size is None):
        bn = 1
    
    dataloader = DataLoader(tImageFloder, batch_size=bn, shuffle=training, num_workers=bn, drop_last=False)
    
    return dataloader
