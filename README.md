# Pytorch Implementation of the Wide Stride Multi-Classification Stereo Matching Network

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

For the stereo matching method based on deep learning, the network architecture is critical for the accuracy of the algorithm, while the efficiency of the algorithm is also an important factor to be considered in practical applications. Based on three-dimensional convolutional neural network(3DCNN), a stereo matching method with wide shift stride is proposed. When forming a cost volume with concatenated features, the wide stride is used to shift the right-view feature map so that the memory and computational resources required for the matching cost calculation phase are reduced by multiples. In the stage of calculating the matching cost by using 3DCNN, the matching cost within a stride is sampled by means of multi-classification output, and the accuracy of the algorithm is improved while ensuring the efficiency. 

## Usage

### Dependencies

- [Python-2.7](https://www.python.org/downloads/)
- [PyTorch-0.4.0+](http://pytorch.org)
- [torchvision-0.2.1](http://pytorch.org) (higher version may cause issues)
- [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Usage of KITTI and SceneFlow dataset in [dataloader/README.md](dataloader/README.md)

### Train
As an example, use the following command to train a WSMCnet on SceneFlow

```
dataname=sceneflow
datapath=/root_sceneflow/
arch=WSMCnet
C=4
S=2

LOG="log_${arch}_C${C}_S${S}_`date +%Y-%m-%d_%H-%M-%S`.txt"
bn=2
python train_val.py --arch $arch --maxdisp 192 --C $C --S $S \
               --dataname $dataname --datapath $datapath --bn $bn \
               --lr 0.001 \
               --epochs 10 --nloop 1 --freq_update 8 --freq_print 20 \
               --loadmodel (optional) \
               --dir_save (path for saving model) \
               2>&1 | tee -a "$LOG"
```

As another example, use the following command to finetune a WSMCnet on KITTI 2015

```
dataname=k2015
datapath=/root_kitti/
arch=WSMCnet
C=4
S=2

LOG="log_${arch}_C${C}_S${S}_`date +%Y-%m-%d_%H-%M-%S`.txt"
bn=2
python train_val.py --arch $arch --maxdisp 192 --C $C --S $S \
               --dataname $dataname --datapath $datapath --bn $bn \
               --lr 0.001 \
               --epochs 10 --nloop 30 --freq_update 8 --freq_print 20 \
               --loadmodel (pretrained PSMNet) \
               --dir_save (path for saving model) \
               2>&1 | tee -a "$LOG"
```
You can also see those examples in [demos/*.sh] for details.

### Evaluation
Use the following command to evaluate the trained WSMCnet on KITTI 2015 test data

```
dataname=k2015-te
datapath=/root_kitti/
arch=WSMCnet
C=4
S=2

python submission.py --arch $arch --maxdisp 192 --C $C --S $S \
               --dataname $dataname --datapath $datapath \
               --loadmodel (finetuned WSMCnet) \
               --dir_save (path for saving result) \
```

You can also see the example in [demos/kitti_submission.sh](demos/kitti_submission.sh) for details.

### Pretrained Model

(1) Pretrained Model for WSMCnet

| Model | KITTI-2015 | SceneFlow | KITTI-2012 |
|---|---|---|---|
| WSMCnet-C1S1 | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ ) | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ )  | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ ) |
| WSMCnet-C4S2 | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ ) | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ )  | [Baidu-pan](https://pan.baidu.com/s/1gURqG2A2s_hHvPswZ-Y4VQ ) |

Extraction code：bwye 



(2) Pretrained Model released from [PSMnet](https://)

※NOTE: The pretrained model were saved in .tar; however, you don't need to untar it. Use torch.load() to load it.

| KITTI-2015 | SceneFlow | KITTI-2012 |
|---|---|---|
| [Google-Drive](https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view?usp=sharing) | [Google-Drive](https://drive.google.com/file/d/1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG/view?usp=sharing) | [Google-Drive](https://drive.google.com/file/d/1p4eJ2xDzvQxaqB20A_MmSP9-KORBX1pZ/view)|
| [Baidu-pan](https://pan.baidu.com/s/1ELkSJ7DPuYliKQ-TZwIKXg ) | [Baidu-pan](https://pan.baidu.com/s/1ELkSJ7DPuYliKQ-TZwIKXg )  | [Baidu-pan](https://pan.baidu.com/s/1ELkSJ7DPuYliKQ-TZwIKXg ) |

Extraction code：wooc 

## Results on [KITTI 2015 leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |
|---|---|---|---|
| [WSMCnet-C4S2]() | 2.63 % | 2.30 % | 0.41 |
| [PSMNet](https://arxiv.org/abs/1803.08669) | 2.32 % | 2.14 % | 0.41 |
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |


## Contacts
wangyf_1991@163.com

Any discussions or concerns are welcomed!
