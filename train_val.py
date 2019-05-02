#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
import shutil
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def train(imgL, imgR, disp_L, flag_optim):
    model.train()

    # carry data to GPU
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # predict disp
    output1, output2, output3 = model(imgL, imgR)
    output1 = torch.squeeze(output1,1)
    output2 = torch.squeeze(output2,1)
    output3 = torch.squeeze(output3,1)

    # compute loss
    mask =  (disp_true > 0) & (disp_true < args.maxdisp)
    mask.detach_()
    loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + \
            0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + \
            F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 

    # backward
    if(loss > 0):
        loss.backward()

    # update weight
    if(flag_optim):
        optimizer.step()
        optimizer.zero_grad()

    return max(0, loss.data.item())

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

    return err_epe, err_3px

def lr_adjust(optimizer, epoch, args):
    
    lr = args.lr
    depoch = epoch - args.lr_epoch0
    if(depoch >= 0):
   
        lr *= args.lr_delay**((epoch-args.lr_epoch0)//args.lr_stride)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #lr_curr = [prm_grp['lr'] for prm_grp in optimizer.param_groups]
        #lr_decay = max(lr_curr)/lr
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] *= lr_decay

    return lr

if __name__ == '__main__':
    # get setting
    args = get_setting()

    # set gpu id used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # create dataloader
    from dataloader import dataloader_by_name
    datanames_tr = args.dataname.replace('_', '-tr, ') + '-tr'
    datanames_val= args.dataname.replace('_', '-val, ') + '-val'
    TrainImgLoader = dataloader_by_name(datanames_tr, args.datapath, args.bn, training=True, crop_size=args.crop_size, n=3)
    ValImgLoader = dataloader_by_name(datanames_val, args.datapath, min(2, args.bn), training=False, crop_size=[0, 0], n=3)

    # create model
    from models import model_by_name
    model = model_by_name(args.arch, args.maxdisp, args.C, args.S, args.rand_shift)

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
        
        msg = 'No pretrained weight! \nloadmodel: %s\n\n'%str(args.loadmodel)
        logger.warning(msg)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)

    msg = 'Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    logger.info(msg)

    #---------progess of last interrupt------------------------------------------------------------------
    dir_weight = args.dir_save # dirpath for saving weight
    if(not os.path.isdir(dir_weight)):
        os.makedirs(dir_weight)
    path_checkpoint = os.path.join(dir_weight, 'checkpoint.pkl')
    path_training_info = os.path.join(dir_weight, 'training_info.pkl')
    
    # load checkpoint if it exist
    epoch0_train, epoch0_val = 0, 0
    train_loss_mean = []
    test_err_epe_mean = []
    test_err_3px_mean = []
    
    if(os.path.isfile(path_training_info)):
        
        info_dict = torch.load(path_training_info)
        train_loss_mean = info_dict['train_loss']
        test_err_epe_mean = info_dict['test_err_epe']
        test_err_3px_mean = info_dict['test_err_3px']
        epoch0_val = len(train_loss_mean)
    
    if(os.path.isfile(path_checkpoint)):
        
        data = torch.load(path_checkpoint)
        model.load_state_dict(data['state_dict'])
        epoch0_train = data['epoch']
        if(epoch0_train > epoch0_val):
            train_loss_mean.append(data['train_loss'])

    # log msg
    msg = 'Trained %d epoch | Valed %d epoch \n' % (epoch0_train, epoch0_val)
    logger.info(msg)

    # start training
    start_full_time = time.time()
    for epoch in range(epoch0_val + 1, args.epochs+1):
        lr = lr_adjust(optimizer, epoch, args)
        msg = 'This is %d-th epoch, lr=%f' %(epoch, lr)
        logger.info(msg)

        #---------training-----------------------------------------------------------------------
        if(epoch0_train < epoch):
            torch.cuda.empty_cache()
            total_train_loss = 0
            count = 0
            Iter_all = args.nloop*len(TrainImgLoader)
            for i in range(args.nloop):
                for batch_idx, (_, imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
                    bn = imgL_crop.shape[0]
                    count += bn
                    start_time = time.time()
    
                    # train of a batch sample
                    flag_step = (batch_idx % args.freq_optim == 0) or (len(TrainImgLoader)-1 == batch_idx)
                    loss = train(imgL_crop, imgR_crop, disp_crop_L, flag_step)
                    total_train_loss += loss*bn
                    
                    # log msg
                    if(batch_idx % args.freq_print == 0):
                        Iter = batch_idx + i*len(TrainImgLoader)
                        msg = 'Train [%d | %2d/%d] | training loss = %6.3f(%6.3f) | time = %.3f' %(
                                epoch, Iter, Iter_all, loss, total_train_loss/count, time.time() - start_time)
                        logger.info(msg)

            # record train meters
            train_loss_mean.append(total_train_loss/count)

            # log msg
            msg = 'Train %d | Mean training loss = %.3f' % (epoch, train_loss_mean[-1])
            logger.info(msg)
    
            # save checkpoint
            path_checkpoint_epoch = os.path.join(dir_weight, 'checkpoint_%d.pkl' % epoch)
            state_dict = {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'train_loss': train_loss_mean[-1],
                          }
            torch.save(state_dict, path_checkpoint_epoch)
            shutil.copyfile(path_checkpoint_epoch, path_checkpoint)
            
            # log msg
            msg = 'Full training time = %.2f HR\n' %((time.time() - start_full_time)/3600)
            logger.info(msg)

        #------------- TEST ------------------------------------------------------------
        if(True):
            torch.cuda.empty_cache()
            total_test_err_epe = 0
            total_test_err_3px = 0
            count = 0
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
                    msg = 'Val [%d | %3d/%d] | err_epe = %6.3f(%6.3f) | err_3px = %6.3f(%6.3f) | time = %.3f' %(epoch, batch_idx, len(ValImgLoader), 
                            test_err_epe, total_test_err_epe/count, test_err_3px, total_test_err_3px/count, time.time()-start_time)
                    logger.info(msg)

            # record Val meters
            test_err_epe_mean.append(total_test_err_epe/count)
            test_err_3px_mean.append(total_test_err_3px/count)

            # log msg
            msg = 'Val %d | Mean err_epe = %.3f | Mean err_3px = %.3f' % (
                    epoch, test_err_epe_mean[-1], test_err_3px_mean[-1])
            logger.info(msg)

            # save training information
            path_tmp = path_training_info + '.tmp~'
            dict_info = {'train_loss': train_loss_mean, 
                        'test_err_epe': test_err_epe_mean,
                        'test_err_3px': test_err_3px_mean,
                        }
            torch.save(dict_info, path_tmp)
            shutil.move(path_tmp, path_training_info)

            # log msg
            msg = 'Full training time = %.2f HR\n' %((time.time() - start_full_time)/3600)
            logger.info(msg)


