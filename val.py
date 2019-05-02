#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
#import numpy as np
import torch
import torch.nn as nn

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_setting():

    import argparse
    parser = argparse.ArgumentParser(description='Pytorch Implementation of WSMCnet')

    # Argument of dataset
    parser.add_argument('--dataname', default='k2015',
                        help='datapath')
    parser.add_argument('--datapath', default='/media/qjc/D/data/kitti/',
                        help='datapath')
    parser.add_argument('--bn', type=int, default=1,
                        help='batch size')
    parser.add_argument('--crop_width', type=int, default=512,
                        help='batch size')
    parser.add_argument('--crop_height', type=int, default=256,
                        help='batch size')

    # Argument of model
    parser.add_argument('--arch', default='WSMCnet',
                        help='select arch of model')
    parser.add_argument('--maxdisp', type=int ,default=192,
                        help='maxium disparity')
    parser.add_argument('--C', type=int ,default=1,
                        help='number of cost classify')
    parser.add_argument('--S', type=int ,default=1,
                        help='stride of shift right feature map')
    parser.add_argument('--rand_shift', action='store_true', default=False,
                        help='use rand shift in training')
    parser.add_argument('--loadmodel', default=None, 
                        help='load model')

    # Argument of optimizer
    parser.add_argument('--freq_optim', type=int, default=8,
                        help='frequent of optimize weight')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learnig rate')
    parser.add_argument('--lr_epoch0', type=int, default=10,
                        help='learnig rate')
    parser.add_argument('--lr_stride', type=int, default=10,
                        help='learnig rate')
    parser.add_argument('--lr_delay', type=float, default=0.1,
                        help='learnig rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='learnig rate')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='learnig rate')

    # Arguments of training
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--nloop', type=int, default=1,
                        help='count of loop dataset in a epoch')
    parser.add_argument('--freq_print', type=int, default=20,
                        help='frequent of print infomation')
    parser.add_argument('--dir_save', default='./trained/',
                        help='dirpath for save result')

    # Other Arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Parse Arguments
    args = parser.parse_args()

    # Add Arguments
    args.betas = [args.beta1, args.beta2]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.crop_size = [args.crop_width, args.crop_height]

    # Log Arguments setted
    items = args.__dict__.items()
    items.sort()
    msg = 'The parameters are set as follow: '
    for k, v in items:
        msg += '\n [%s]: %s' % (k, str(v))
    msg += '\n'
    logger.info(msg)

    return args

def test(imgL, imgR, disp_true):
    model.eval()

    # carry data to GPU
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    # predict disp
    with torch.no_grad():
        output3 = model(imgL,imgR)
    _, h, w = disp_true.shape
    disp_pred = output3.data.cpu()[:,-h:,-w:]

    # compute epe and 3px(D1)
    err_epe, err_3px = evalutate(disp_pred, disp_true, args.maxdisp)

    return err_epe, err_3px

def evalutate(disp_pred, disp_true, maxdisp):
    # select valid pixels
    mask = (disp_true > 0) & (disp_true < maxdisp)
    if len(mask[mask])==0:
       return 0, 0

    # computing end-point-error
    diff = torch.abs(disp_pred[mask] - disp_true[mask])
    err_epe = torch.mean(diff).item()

    # computing 3-px error #
    correct = ((diff < 3) | (diff < disp_true[mask]*0.05))
    err_3px = 100*(1 - float(len(correct[correct]))/len(mask[mask]))
    
#    import matplotlib.pyplot as plt
#    disp_err = torch.zeros_like(disp_true)
#    disp_err[mask] = diff.clamp(0, 3)
#    plt.subplot(311);plt.imshow(disp_pred[0].numpy())
#    plt.subplot(312);plt.imshow(disp_true[0].numpy())
#    plt.subplot(313);plt.imshow(disp_err[0].numpy())
#    plt.show()
    
    return err_epe, err_3px

if __name__ == '__main__':
    # get setting
    args = get_setting()

    # set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    # create dataloader
    from dataloader import dataloader_by_name
    ValImgLoader = dataloader_by_name(args.dataname, args.datapath, 1, training=False, crop_size=None, n=3)

    # create model
    from models import model_by_name
    model = model_by_name(args.arch, args.maxdisp, args.C, args.S)
    msg = 'Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    logger.info(msg)


    # carry model to cuda
    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    # load model weight
    if os.path.isfile(args.loadmodel):
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        msg = 'Load pretrained weight successively! \n loadmodel: %s\n'%str(args.loadmodel)
        logger.info(msg)
    elif('None'!=str(args.loadmodel).strip()):
        msg = 'No pretrained weight! \n loadmodel: %s\n\n'%str(args.loadmodel)
        logger.warning(msg)

    # start test
    ## Test dataset ##
    start_full_time = time.time()
    total_test_err_epe = 0
    total_test_err_3px = 0
    count = 0
    torch.cuda.empty_cache()
    
    for batch_idx, (_, imgL, imgR, disp_L) in enumerate(ValImgLoader):
        bn = imgL.shape[0]
        count += bn
        start_time = time.time()

        # Val of a batch sample
        test_err_epe, test_err_3px = test(imgL,imgR, disp_L)
        total_test_err_epe += test_err_epe*bn
        total_test_err_3px += test_err_3px*bn
        
        # log msg
        if(batch_idx % args.freq_print == 0):
            msg = 'Val %2d/%d | err_epe = %6.3f(%6.3f) | err_3px = %6.3f(%6.3f) | time = %.3f' %(batch_idx, len(ValImgLoader), 
                    test_err_epe, total_test_err_epe/count, test_err_3px, total_test_err_3px/count, time.time()-start_time)
            logger.info(msg)

   
    # log msg
    msg = '\n Weight: %s\n Mean err_epe = %.3f | Mean err_3px = %.3f\n' % (
            args.loadmodel, total_test_err_epe/count, total_test_err_3px/count)
    logger.info(msg)
    full_time = (time.time() - start_full_time)
    msg = 'Full Val time = %.2f s (%.2f s)' %(full_time, full_time/count)
    logger.info(msg)

