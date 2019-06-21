#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

def _laplace_probability(value, center, diversity=1.0):
    return torch.exp(-torch.abs(center - value) / diversity) # / (2 * diversity)


def subpixel_cross_entropy_loss(similarity, disp_true, disp_step, diversity=1.0, weights=None):
    """Returns sub-pixel cross-entropy loss.

    Cross-entropy is computed as

    - sum_d [ log(P_predicted(d)) * P_target(d) ]
      -------------------------------------------------
                        sum_d P_target(d)

    We need to normalize the cross-entropy by sum_d P_target(d),
    since the target distribution is not normalized.

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_true: Tensor with ground truth disparities with
                    indices [example_index, y, x]. The
                    disparity values are floats. The locations with unknown
                    disparities are filled with 'inf's.
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        diversity: diversity of the target Laplace distribution,
                   centered at the sub-pixel ground truth.
        weights  : Tensor with weights of individual locations.
    """

    mask_valid = (disp_true.data != float('inf'))
    if(len(mask_valid[mask_valid])==0):
        return torch.zeros(1)

    log_P_predicted = F.log_softmax(similarity, dim=1)
    sum_P_target = torch.zeros_like(disp_true)
    sum_P_target_x_log_P_predicted = torch.zeros_like(disp_true)

    for idx_disp in range(similarity.size(-3)):
        disparity = idx_disp * disp_step
        P_target = _laplace_probability(disparity, disp_true, diversity)
        sum_P_target += P_target
        sum_P_target_x_log_P_predicted += (log_P_predicted[:, idx_disp] * P_target)
    
    entropy = -sum_P_target_x_log_P_predicted[mask_valid] / sum_P_target[mask_valid]
    
    if weights is not None:
        weights_valid = weights[mask_valid]
        return (weights_valid * entropy).sum() / (weights_valid.sum() + 1e-15)

    return entropy.mean()

if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    # prepare data
    cost = torch.rand(2, 200, 256, 256).cuda()
    disp_true = torch.rand(2, 256, 256).cuda()*100
    y = subpixel_cross_entropy_loss(cost, disp_true, 0.5)

    logger.info('shape of cost: %s \n' % str(cost.shape))
    logger.info('shape of disp_true: %s \n' % str(disp_true.shape))
    logger.info('output of loss_cross_entropy: %.3f \n' % y.item())
