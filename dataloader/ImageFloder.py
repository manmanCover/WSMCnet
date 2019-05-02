#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
import preprocess 
import traceback

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')


class ImageFloder(data.Dataset):
    def __init__(self, datasets, n, training, crop_size):
 
        # self.n = 2/3/4
        # 2-->left, right 
        # 3-->left, right, disp_left 
        # 4-->left, right, disp_left, disp_right
        n = min([ds.num_in_group for ds in datasets] + [n])
        self.n = max(2, min(4, n)) 
        
        self.count_datasets = len(datasets)
        self.count = max([len(ds) for ds in datasets])
        self.datasets = datasets
        self.training = training

        # reset crop_size
        if (crop_size is not None) and ([0, 0]==list(crop_size)): # crop_size=[width, height]
            common_sizes = [ds.common_size for ds in datasets]
            if(None in common_sizes):
                crop_size = None
            elif(len(common_sizes)==1):
                crop_size = common_sizes[0]
            else:
                ws = [wh[0] for wh in common_sizes]
                hs = [wh[1] for wh in common_sizes]
                crop_size = [min(ws), min(hs)]
        self.crop_size = crop_size # [width, height]

    def pad_imgs(self, crop_size, left, right, disp_left=None, disp_right=None):
        w, h = left.size
        tw, th = crop_size
        x, y = min(0, w-tw), min(0, h-th)
        if(tw > w or th > h):
            left = left.crop([x, y, w, h])
            right = right.crop([x, y, w, h])
            if(tw > w and th > h):
                if(disp_left is not None):
                    disp_left = np.pad(disp_left, ((th-h, 0), (tw-w, 0)), 'constant')
                if(disp_right is not None):
                    disp_right = np.pad(disp_right, ((th-h, 0), (tw-w, 0)), 'constant')
            elif(tw > w):
                if(disp_left is not None):
                    disp_left = np.pad(disp_left, ((0, 0), (tw-w, 0)), 'constant')
                if(disp_right is not None):
                    disp_right = np.pad(disp_right, ((0, 0), (tw-w, 0)), 'constant')
            elif(th > h):
                if(disp_left is not None):
                    disp_left = np.pad(disp_left, ((th-h, 0), (0, 0)), 'constant')
                if(disp_right is not None):
                    disp_right = np.pad(disp_right, ((th-h, 0), (0, 0)), 'constant')
        return left, right, disp_left, disp_right
        
    def __getitem__(self, index):
        try:
            # get paths and loader and dploader
            idx = index % self.count_datasets
            paths_grp = self.datasets[idx][index//self.count_datasets]
            loader = self.datasets[idx].img_loader
            dploader = self.datasets[idx].disp_loader
            
            # loader image and disp
            tn = len(paths_grp)
            assert tn >= self.n
            left = loader(paths_grp[0])
            right = loader(paths_grp[1])
            disp_left = None
            disp_right = None

            # Random Horizontal Flip
            if(self.training and tn%2 == 0 and random.random() > 0.5):
                left_t = right.transpose(Image.FLIP_LEFT_RIGHT)
                right = left.transpose(Image.FLIP_LEFT_RIGHT)
                left = left_t
                if(tn == 4): 
                    disp_left = np.fliplr(dploader(paths_grp[3]))
                    disp_left = np.ascontiguousarray(disp_left)
                    if(self.n == 4):
                        disp_right = np.fliplr(dploader(paths_grp[2]))
                        disp_right = np.ascontiguousarray(disp_right)
            
            else:
                if(self.n >= 3):
                    disp_left = dploader(paths_grp[2])
                if(self.n >= 4):
                    disp_right = dploader(paths_grp[3])
            

            
            # crop_size
            if self.crop_size is not None:
                # pad image 
                left, right, disp_left, disp_right = self.pad_imgs(self.crop_size, left, right, disp_left, disp_right)
                # crop
                w, h = left.size
                tw, th = self.crop_size
                if self.training: 
                    x1 = random.randint(0, w - tw) if w>tw else 0
                    y1 = random.randint(0, h - th) if h>th else 0
                else:
                    x1 = w - tw
                    y1 = h - th
                left = left.crop((x1, y1, x1 + tw, y1 + th))
                right = right.crop((x1, y1, x1 + tw, y1 + th))
                if(self.n >= 3):
                    disp_left = disp_left[y1:y1+th, x1:x1+tw]
                if(self.n >= 4):
                    disp_right = disp_right[y1:y1+th, x1:x1+tw]
            
            # data augment
            processed = preprocess.get_transform(augment=self.training)  
            left   = processed(left)
            right  = processed(right)

            filename = os.path.basename(paths_grp[0])
            # return
            if(self.n == 2):
                return filename, left, right
            elif(self.n == 3):
                return filename, left, right, disp_left
            elif(self.n == 4):
                return filename, left, right, disp_left, disp_right

        except Exception as err:
            logging.error(traceback.format_exc())
            msg = '[ Loadering data ] An exception happened: %s \n\t left: %s' % (str(err), paths_grp[0])
            logging.error(msg)
            index = random.randint(0, len(self)-1)
            return self.__getitem__(index)

    def __len__(self):
        return self.count*self.count_datasets

