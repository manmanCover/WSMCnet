#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
set -x


Cn='8 4 2 1 ' # '8 4 2 1 '
Sn='1 2 4 8 16 ' # '1 2 4 8 16 ' 
for S in $Sn ; do
    for C in $Cn ; do
        ./demos/wsmcnet_finetune_sceneflow.sh $C $S
    done
done


echo -e "************ end of ${0}\n\n\n"

