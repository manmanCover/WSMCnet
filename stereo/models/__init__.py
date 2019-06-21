#!/usr/bin/env python
# -*- coding: UTF-8 -*-


def model_by_name(name, maxdisp):
    
    # WSMCnet*
    if('WSMCnet'.lower() in name.lower()):
        from .WSMCnet import get_model_by_name
        model = get_model_by_name(name, maxdisp)

    # Unsupported model
    else:
        raise Exception('Unsupported model: ' + name)
    
    return model



