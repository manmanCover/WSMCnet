#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
#import math
#import numpy as np
import Submodules
import logging
logger = logging.getLogger(__name__)


class WSMCnetBase(nn.Module):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetBase, self).__init__()
        
        # maximum of disparity
        self.maxdisp = maxdisp
        
        # parse S, C and F3 from name_model
        S, C, F3 = self._postfix_parse(name_model)
        self.S = S
        self.C = C
        self.F3 = F3

        # modules of [feature_extraction and cost_compute]
        self.feature_extraction = None
        self.cost_compute = None


    @property
    def name(self):
        return self._get_name() + self.postfix


    @property
    def postfix(self):
        return '_S%dC%dF%d'%(self.S, self.C, self.F3)


    def _postfix_parse(self, name_model):

        S, C, F3 = 1, 1, 32
        strs = name_model.lower().split('_')
        if(2 == len(strs)):
            subfix = strs[1]
            idxs = [subfix.find(tmp) for tmp in ['s', 'c', 'f']]
            if(idxs[1] > 0):
                S = int(subfix[idxs[0]+1:idxs[1]])
                if(idxs[2] > 0):
                    C = int(subfix[idxs[1]+1:idxs[2]])
                    F3 = int(subfix[idxs[2]+1:])
                else:
                    C = int(subfix[idxs[1]+1:])
            else:
                S = int(subfix[idxs[0]+1:])
            S = max(1, min(16, S))
            C = max(1, min(16, C))
            F3 = max(8, min(64, F3))

        return (S, C, F3)


    def get_matching_costs(self, left, right):

        # feature extraction
        bn = left.shape[0]
        features = torch.cat([left, right], dim=0)
        features = self.feature_extraction(features)
        fL, fR = features[:bn], features[bn:]

        # compute matching cost
        stride = self.S
        shift = (self.maxdisp//(stride<<2)) + 1
        cost = Submodules.cost_volume_gen(fL, fR, shift, stride)
        costs = self.cost_compute(cost)
        
        return costs
    
    
    def cost_disps(self, disps, disp_true):
        
        # split disp
        output1, output2, output3 = disps
    
        # mask of validate labels
        maxdisp = self.maxdisp
        mask =  (disp_true > 0) & (disp_true < maxdisp)
        
        # compute loss
        lossfun = F.smooth_l1_loss
        loss = 0.5*lossfun(output1[mask], disp_true[mask], size_average=True) + \
                0.7*lossfun(output2[mask], disp_true[mask], size_average=True) + \
                lossfun(output3[mask], disp_true[mask], size_average=True) 

        return loss


    def cost_similarity(self, similarities, disp_true):
        
        # split disp
        output1, output2, output3 = similarities
    
        # mask of validate labels
        maxdisp = self.maxdisp
        mask_invalid =  ~((disp_true > 0) & (disp_true < maxdisp))
        disp_true[mask_invalid] = float('inf')
        
        # compute loss
        lossfun = Submodules.subpixel_cross_entropy_loss
        loss = 0.5*lossfun(output1, disp_true, self.disp_step) + \
                0.7*lossfun(output2, disp_true, self.disp_step) + \
                lossfun(output3, disp_true, self.disp_step) 

        return loss

    def cost_similarity_disps(self, output, disp_true):
        
        # split output
        similarities, disps = output
        
        # L1 loss
        loss_L1 = self.cost_disps(disps, disp_true)
        
        # crose-entropy loss
        loss_ce = self.cost_similarity(similarities, disp_true)

        if(loss_L1<10):
            return loss_ce + 0.2*loss_L1
        else:
            return loss_ce


class WSMCnetTri_L1(WSMCnetBase):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetTri_L1, self).__init__(name_model, maxdisp)

        # upsample and softmax
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.disp_step = float(self.S)/self.C


    def forward(self, left, right):

        # compute matching costs
        costs = self.get_matching_costs(left, right)
        
        # disparty regression
        h, w = left.shape[-2:]
        disps = []
        for tcost in costs:
            tcost = self.upsample(tcost).squeeze(1)[..., :h, :w]
            disps.append(Submodules.disp_regression(tcost, self.disp_step))
        
        # return
        logger.debug('training: %s \n' % str(self.training))
        return disps if self.training else disps[-1]


    def compute_cost(self, disps, disp_true):
        return self.cost_disps(disps, disp_true)


class WSMCnetBi_L1(WSMCnetBase):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetBi_L1, self).__init__(name_model, maxdisp)

        # upsample and softmax
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.disp_step = float(self.S<<2)/self.C


    def forward(self, left, right):

        # compute matching costs
        costs = self.get_matching_costs(left, right)
        
        # disparty regression
        h, w = left.shape[-2:]
        disps = []
        for tcost in costs:
            tcost = self.upsample(tcost.squeeze(1))[..., :h, :w]
            disps.append(Submodules.disp_regression(tcost, self.disp_step))
        
        # return
        logger.debug('training: %s \n' % str(self.training))
        return disps if self.training else disps[-1]


    def compute_cost(self, disps, disp_true):
        return self.cost_disps(disps, disp_true)


class WSMCnet(WSMCnetTri_L1):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnet, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostCompute import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetB(WSMCnetBi_L1):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetB, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostCompute import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetL(WSMCnetTri_L1):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetL, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostComputeL import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetLB(WSMCnetBi_L1):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetLB, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostComputeL import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetTri_CE(WSMCnetBase):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetTri_CE, self).__init__(name_model, maxdisp)

        # upsample and softmax
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.disp_step = float(self.S)/self.C


    def forward(self, left, right):

        # compute matching costs
        costs = self.get_matching_costs(left, right)
        
        # disparty regression
        h, w = left.shape[-2:]
        costs = [self.upsample(tcost).squeeze(1)[..., :h, :w] for tcost in costs]
        disps = [Submodules.disp_regression_nearby(tcost, self.disp_step) for tcost in costs]

        if(self.training):
            return costs, disps
        else:
            return disps[-1]


    def compute_cost(self, output, disp_true):
        return self.cost_similarity_disps(output, disp_true)


class WSMCnetBi_CE(WSMCnetBase):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetBi_CE, self).__init__(name_model, maxdisp)

        # upsample and softmax
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.disp_step = float(self.S<<2)/self.C


    def forward(self, left, right):

        # compute matching costs
        costs = self.get_matching_costs(left, right)
        
        # disparty regression
        h, w = left.shape[-2:]
        costs = [self.upsample(tcost.squeeze(1))[..., :h, :w] for tcost in costs]
        disps = [Submodules.disp_regression_nearby(tcost, self.disp_step) for tcost in costs]

        if(self.training):
            return costs, disps
        else:
            return disps[-1]


    def compute_cost(self, output, disp_true):
        return self.cost_similarity_disps(output, disp_true)


class WSMCnetE(WSMCnetTri_CE):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetE, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostCompute import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetEB(WSMCnetBi_CE):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetEB, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostCompute import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetEL(WSMCnetTri_CE):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetEL, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostComputeL import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


class WSMCnetELB(WSMCnetBi_CE):
    
    def __init__(self, name_model, maxdisp):
        super(WSMCnetELB, self).__init__(name_model, maxdisp)

        # modules of [feature_extraction and cost_compute]
        from Submodules.FeatureExtraction import FeatureExtraction
        from Submodules.CostComputeL import CostCompute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F3, C=self.C)

        # init weight
        Submodules.weight_init(self.modules())


def get_model_by_name(name, maxdisp):
    
    name_class = name.split('_')[0]
    try:
        return eval(name_class)(name, maxdisp)
    except:
        raise Exception('Unsupported model: ' + name)

if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
    for name in [
                'WSMCnet', 'WSMCnetB', 'WSMCnetL', 'WSMCnetLB', 
                'WSMCnetE', 'WSMCnetEB', 'WSMCnetEL', 'WSMCnetELB', 
                ]:
        model = get_model_by_name(name + '_S2C3F32', 192)
        logger.info('%s passed!\n ' % model.name + 
                    'Fun_active   : %s \n '% str(model.cost_compute.fun_active) + 
                    'Fun_upsample : %s \n '% str(model.upsample) + 
                    'disp_step : %s \n '% str(model.disp_step))

