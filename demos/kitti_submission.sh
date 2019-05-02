#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
set -x

## augments
C=${1-4}
S=${2-2}
ID=${3-k15}
echo

arch="WSMCnet"
maxdisp=192
bn=1
freq_print=1
echo


## log_dirpath
log_dir="logs"
mkdir -p "${log_dir}"
echo


## k2015(ID=k15)/k2012(ID=k12)
dataname="${ID}-te"
datapath="/media/qjc/D/data/kitti"
mode="Submission"
dataname_train="T(sf)_F(k15_k12)_F(${ID})"
nckp=10
loadmodel="./results/Train_${arch}_C${C}S${S}_${dataname_train}/checkpoint_${nckp}.pkl"
#dataname_train="pretrained"
#loadmodel="./pretrained/pretrained_model_KITTI2015.tar"
echo

# log_filepath and dir_save
flag="${mode}_${dataname}_${arch}_C${C}S${S}_${dataname_train}"
LOG="$log_dir/log_${flag}_`date +%Y-%m-%d_%H-%M-%S`.txt"
dir_save="./results/${flag}"
echo

python submission.py --arch $arch --maxdisp $maxdisp --C $C --S $S \
                   --loadmodel $loadmodel \
                   --dataname $dataname --datapath $datapath \
                   --freq_print $freq_print \
                   --dir_save $dir_save \
                   2>&1 | tee -a "${LOG}"
echo


echo -e "************ end of ${0}\n\n\n"


