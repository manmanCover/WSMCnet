#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
set -x


## arch of model
arch=${1-WSMCnet_S2C3F32}
bn=${2-1}
maxdisp=${3-192}
datas_val=${4-k15-val,k12-val}

freq_print=1
echo


## k2015
dir_root="/media/qjc/D/data/kitti"
dir_datas_val="${dir_root}"
mode="Submission"
datas_train="T(sf-tr)_F(k15-tr,k12-tr)"
nckp=20
loadmodel="./results/Train_${arch}_${datas_train}/weight_${nckp}.pkl"

# log_filepath and dir_save
flag="${mode}_${datas_val}/${arch}_${datas_train}"
dir_save="./results/${flag}"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"
echo

# val model
python main.py --mode ${mode} --arch $arch --maxdisp $maxdisp --bn $bn \
               --loadmodel $loadmodel \
               --datas_val $datas_val --dir_datas_val $dir_datas_val \
               --freq_print $freq_print \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"
echo


echo -e "************ end of ${0}\n\n\n"



