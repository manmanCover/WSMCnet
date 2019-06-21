#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
#import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


# It is faster than disp_regression1
def disp_regression(similarity, disp_step):
    """Returns predicted disparity with argsofmax(disp_similarity).

    Predicted disparity is computed as: d_predicted = sum_d( d * P_predicted(d))

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())
    
    P = F.softmax(similarity, dim=1)
    disps = torch.arange(0, P.size(-3)).type_as(P.data)*disp_step
    return torch.sum(P*disps[None, :, None, None], 1)
    

# It is slower than disp_regression
def disp_regression1(similarity, disp_step):
    """Returns predicted disparity with argsofmax(disp_similarity).

    Predicted disparity is computed as: d_predicted = sum_d( d * P_predicted(d))

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())
    
    P = F.softmax(similarity, dim=1)
    disps = torch.arange(0, P.size(1)).type_as(P.data)*disp_step
    return P.permute(0, 2, 3, 1).matmul(disps)
    

# It is faster than disp_regression_nearby1
def disp_regression_nearby(similarity, disp_step, half_support_window=2):
    """Returns predicted disparity with subpixel_map(disp_similarity).

    Predicted disparity is computed as: 
    
    d_predicted = sum_d( d * P_predicted(d)), 
    where | d - d_similarity_maximum | < half_size

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        half_support_window: defines size of disparity window in pixels
                             around disparity with maximum similarity,
                             which is used to convert similarities
                             to probabilities and compute mean.
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())

    # In every location (x, y) find disparity with maximum similarity score.
    similar_maximum, idx_maximum = torch.max(similarity, dim=1, keepdim=True)
    idx_limit = similarity.size(1)-1
    
    # Collect similarity scores for the disparities around the disparity
    # with the maximum similarity score.
    support_idx_disp = []
    for idx_shift in range(-half_support_window, half_support_window+1):

        idx_disp = idx_maximum + idx_shift
        idx_disp[idx_disp < 0] = 0
        idx_disp[idx_disp >= idx_limit] = idx_limit
        support_idx_disp.append(idx_disp)

    support_idx_disp = torch.cat(support_idx_disp, dim=1) 
    support_similar = torch.gather(similarity, 1, support_idx_disp.long())
    support_disp = support_idx_disp.float()*disp_step

    # Convert collected similarity scores to the disparity distribution
    # using softmax and compute disparity as a mean of this distribution.
    prob = F.softmax(support_similar, dim=1)
    disp = torch.sum(prob * support_disp.float(), dim=1)

    return disp
    

# It is slower than disp_regression_nearby
def disp_regression_nearby1(similarity, disp_step, half_support_window=2):
    """Returns predicted disparity with subpixel_map(disp_similarity).

    Predicted disparity is computed as: 
    
    d_predicted = sum_d( d * P_predicted(d)), 
    where | d - d_similarity_maximum | < half_size

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        half_support_window: defines size of disparity window in pixels
                             around disparity with maximum similarity,
                             which is used to convert similarities
                             to probabilities and compute mean.
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())

    # In every location (x, y) find disparity with maximum similarity score.
    similar_maximum, idx_maximum = torch.max(similarity, dim=1, keepdim=True)
    support_disp, support_similar = [], []
    idx_limit = similarity.size(1)-1
    
    # Collect similarity scores for the disparities around the disparity
    # with the maximum similarity score.
    for idx_shift in range(-half_support_window, half_support_window+1):
        
        idx_disp = idx_maximum + idx_shift
        idx_disp[idx_disp < 0] = 0
        idx_disp[idx_disp >= idx_limit] = idx_limit
        nearby_similar = torch.gather(similarity, 1, idx_disp.long())
        support_similar.append(nearby_similar)
        nearby_disp = idx_disp.float()*disp_step
        support_disp.append(nearby_disp)
    
    support_similar = torch.cat(support_similar, dim=1)
    support_disp = torch.cat(support_disp, dim=1)

    # Convert collected similarity scores to the disparity distribution
    # using softmax and compute disparity as a mean of this distribution.
    prob = F.softmax(support_similar, dim=1)
    disp = torch.sum(prob * support_disp.float(), dim=1)

    return disp


def cmp_time():
    import time
    count = 50
    for fun in [disp_regression, disp_regression1, disp_regression_nearby, disp_regression_nearby1]:
        x = torch.rand(1, 192, 540, 960).cuda()
        time_start = time.time()
        for i in range(count):
            fun(x, 1)
        time_end = time.time()
        time_mean = (time_end-time_start)/count
        print('%s: %.3f' % (str(fun), time_mean))


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    # prepare data
    x = torch.rand(2, 20, 1, 1).cuda()
    logger.info('Values of x: \n %s \n' % str(x.view(-1).cpu()) )
    logger.info('Values of softmax(x, dim=1): \n %s \n' % 
                str(F.softmax(x, dim=1).view(-1).cpu()) )
    
    x_max, idx_max = torch.max(x, dim=1)
    logger.info('Values of max(x, dim=1): \n %s \n %s \n' % 
                (str(x_max.view(-1).cpu()), str(idx_max.view(-1).cpu())) )
    
    logger.info('Values of disp_regression(x, 1): \n %s \n' % 
                str(disp_regression(x, 1).view(-1).cpu()) )
    logger.info('Values of disp_regression1(x, 1): \n %s \n' % 
                str(disp_regression1(x, 1).view(-1).cpu()) )
    
    logger.info('Values of disp_regression_nearby(x, 1): \n %s \n' % 
                str(disp_regression_nearby(x, 1).view(-1).cpu()) )
    logger.info('Values of disp_regression_nearby1(x, 1): \n %s \n' % 
                str(disp_regression_nearby1(x, 1).view(-1).cpu()) )

    cmp_time()
    
