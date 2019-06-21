#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
set -x

bn=4
maxdisp=192

datas_val="k15-val"
./demos/val_kitti.sh "WSMCnet_S1C1F32" ${bn} ${maxdisp} ${datas_val}
./demos/val_kitti.sh "WSMCnetEB_S2C3F32" ${bn} ${maxdisp} ${datas_val}
./demos/val_kitti.sh "WSMCnetEB_S3C3F32" ${bn} ${maxdisp} ${datas_val}

datas_val="k12-val"
./demos/val_kitti.sh "WSMCnet_S1C1F32" ${bn} ${maxdisp} ${datas_val}
./demos/val_kitti.sh "WSMCnetEB_S2C3F32" ${bn} ${maxdisp} ${datas_val}
./demos/val_kitti.sh "WSMCnetEB_S3C3F32" ${bn} ${maxdisp} ${datas_val}


echo -e "************ end of ${0} ${*} ************\n\n\n"


