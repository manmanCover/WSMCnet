#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from .PSMnet import PSMNet
from .WSMCnet import WSMCnet

def model_by_name(name, maxdisp, C, S, rand_shift=False):
    
    if name.lower() == 'wsmcnet':
        model = WSMCnet(maxdisp, C, S, rand_shift)
    elif name.lower() == 'psmnet':
        model = PSMNet(maxdisp)
    else:
        print('no model')
    
    return model



