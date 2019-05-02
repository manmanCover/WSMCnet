#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
#import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import skimage
#import matplotlib.pyplot as plt

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

def test(imgL, imgR):
    model.eval()

    # carry data to GPU
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    # predict disp
    start_time = time.time()
    with torch.no_grad():
        output3 = model(imgL,imgR)
    disp_pred = output3.data.cpu().numpy()

    return disp_pred, time.time()-start_time

def get_pad(num, ceil):
    mod = num % ceil
    pad = ceil - mod if mod>0 else 0
    return pad
    
if __name__ == '__main__':

    # get setting
    args = get_setting()

    # set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    # create dataloader
    from dataloader import dataloader_by_name
    ValImgLoader = dataloader_by_name(args.dataname, args.datapath, 1, training=False, crop_size=None, n=2)

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

    # create dir_save
    dir_save = args.dir_save
    if(not os.path.exists(dir_save)):
        os.makedirs(dir_save)

    ## Test dataset ##
    times = []
    start_full_time = time.time()
    for batch_idx, (filename, imgL, imgR) in enumerate(ValImgLoader):
        torch.cuda.empty_cache()

        # pad image
        top_pad = get_pad(imgL.shape[-2], 16)
        left_pad = get_pad(imgL.shape[-1], 16)
        imgL = F.pad(imgL, [left_pad, 0, top_pad, 0])
        imgR = F.pad(imgR, [left_pad, 0, top_pad, 0])

        # predict disp
        pred_disp, runtime = test(imgL, imgR)
        times.append(runtime) # record runtime
        
        # log msg
        msg = 'Submission %2d/%d | %s | time = %.3f' %(batch_idx, len(ValImgLoader), filename[0], runtime)
        logger.info(msg)
        
        # save result
        path_save = os.path.join(dir_save, filename[0])
        img = pred_disp[0][top_pad:, left_pad:]
        skimage.io.imsave(path_save, (img*256).astype('uint16'))
#        plt.imshow(img)
#        plt.pause(1)
    full_time = time.time() - start_full_time
    msg = 'Mean runtime = %.3f | mean time = %.3f' % (sum(times)/len(times), full_time/len(ValImgLoader))
    logger.info(msg)







