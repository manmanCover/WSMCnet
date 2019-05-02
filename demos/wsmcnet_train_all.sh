#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0}\n"
set -x


./demos/wsmcnet_train.sh 4 2
#./demos/wsmcnet_train.sh 8 4
./demos/wsmcnet_train.sh 1 1

echo -e "************ end of ${0}\n\n\n"

