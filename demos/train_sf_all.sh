#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo -e "************ start of ${0} ${*} ************\n"
set -x

### S={1, 2, 3, 4, 5, 6, 7, 8}
#./demos/train_sf.sh "WSMCnet_S1C1F32" 4 4
#./demos/train_sf.sh "WSMCnet_S2C1F32" 4 4
#./demos/train_sf.sh "WSMCnet_S3C1F32" 8 2
#./demos/train_sf.sh "WSMCnet_S4C1F32" 8 2
#./demos/train_sf.sh "WSMCnet_S5C1F32" 8 2
#./demos/train_sf.sh "WSMCnet_S6C1F32" 8 2
#./demos/train_sf.sh "WSMCnet_S7C1F32" 8 2
#./demos/train_sf.sh "WSMCnet_S8C1F32" 8 2

### C={2, 3, 4, 5, 6}
#./demos/train_sf.sh "WSMCnet_S2C2F32" 4 4
#./demos/train_sf.sh "WSMCnet_S2C3F32" 4 4
#./demos/train_sf.sh "WSMCnet_S2C4F32" 4 4
#./demos/train_sf.sh "WSMCnet_S2C5F32" 4 4
#./demos/train_sf.sh "WSMCnet_S2C6F32" 4 4
#
### C={2, 3, 4, 5, 6}
#./demos/train_sf.sh "WSMCnet_S3C2F32" 4 4
#./demos/train_sf.sh "WSMCnet_S3C3F32" 4 4
#./demos/train_sf.sh "WSMCnet_S3C4F32" 4 4
#./demos/train_sf.sh "WSMCnet_S3C5F32" 4 4
#./demos/train_sf.sh "WSMCnet_S3C6F32" 4 4


## others
#./demos/train_sf.sh "WSMCnetEB_S3C3F32" 4 4
#./demos/train_sf.sh "WSMCnetE_S3C3F32" 4 4
#./demos/train_sf.sh "WSMCnetE_S3C3F32" 4 4


echo -e "************ end of ${0} ${*} ************\n\n\n"


