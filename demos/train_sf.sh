#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
set -x


## arch of model
arch=${1-WSMCnet_S2C4F32}
bn=${2-4}
freq_optim=${3-4}

maxdisp=192
crop_width=512
crop_height=256
freq_print=20
lr_stride=10
lr_delay=0.1
echo


## sceneflow
dir_root="/media/qjc/D/data/sceneflow/"
datas_train="sf-tr"
datas_val="sf-val"
dir_datas_train=${dir_root}
dir_datas_val=${dir_root}
mode="Train"
loadmodel="None"
echo

# log_filepath and dir_save
flag="${mode}_${arch}_${mode:0:1}(${datas_train})"
dir_save="./results/${flag}"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"
echo

# train model
epochs=25
nloop=1
lr=0.001
lr_epoch0=16
python main.py --mode ${mode} --arch $arch --maxdisp $maxdisp --bn $bn \
               --loadmodel $loadmodel \
               --datas_train $datas_train --dir_datas_train $dir_datas_train \
               --datas_val $datas_val --dir_datas_val $dir_datas_val \
               --crop_width $crop_width --crop_height $crop_height \
               --epochs $epochs --nloop $nloop --freq_print $freq_print \
               --freq_optim $freq_optim \
               --lr $lr --lr_epoch0 $lr_epoch0 \
               --lr_stride $lr_stride --lr_delay $lr_delay \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"
echo


echo -e "************ end of ${0} ${*} ************\n\n\n"

