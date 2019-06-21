# Pytorch Implementation of the Wide Stride Multi-Classification Stereo Matching Network

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

For the stereo matching method based on deep learning, the network architecture is critical for the accuracy of the algorithm, while the efficiency is also an important factor to consider in practical applications. A stereo matching method with spare cost volume in disparity dimension is proposed. The spare cost volume is created by shifting right-view features with a wide stride to reduce greatly the memory and computational resources in three-dimension convolution module. The matching cost is nonlinearly sampled by means of multi-classification in disparity dimension, and model is trained with merging two kind of loss function, so that the accuracy is improved without notably lowering the efficiency. 

## Usage

### Dependencies

- [Python-2.7](https://www.python.org/downloads/)
- [PyTorch-0.4.0](http://pytorch.org)
- [torchvision-0.2.1](http://pytorch.org)
- [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Usage of KITTI and SceneFlow dataset in [stereo/dataloader/README.md](stereo/dataloader/README.md)

### Train
As an example, use the following command to train a WSMCnet on SceneFlow

```
dir_save="./results"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"

python main.py --mode Train --arch WSMCnetEB_S2C3F32 --maxdisp 192 --bn 4 \
               --loadmodel None \
               --datas_train "sf-tr" --dir_datas_train (dir_root_sf) \
               --datas_val "sf-val" --dir_datas_val (dir_root_sf) \
               --crop_width 512 --crop_height 256 \
               --epochs 20 --nloop 1 --freq_print 20 \
               --freq_optim 4 \
               --lr 0.001 --lr_epoch0 16 \
               --lr_stride 10 --lr_delay 0.1 \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"
```

As another example, use the following command to finetune a WSMCnet on KITTI

```
dir_save="./results"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"

python main.py --mode Finetune --arch WSMCnetEB_S2C3F32 --maxdisp 192 --bn 4 \
               --loadmodel (filepath of pretrained weight) \
               --datas_train "k15-tr,k12-tr" --dir_datas_train (dir_root_kitti) \
               --datas_val "k15-val,k12-val" --dir_datas_val (dir_root_kitti) \
               --crop_width 512 --crop_height 256 \
               --epochs 20 --nloop 30 --freq_print 20 \
               --freq_optim 4 \
               --lr 0.005 --lr_epoch0 16 \
               --lr_stride 10 --lr_delay 0.2 \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"

```
You can also see those examples in [demos/train_*.sh] for details.

### Evaluation
Use the following command to evaluate the trained WSMCnet on KITTI 2015 test data

```
dir_save="./results"
LOG="${dir_save}/log_`date +%Y-%m-%d_%H-%M-%S`.txt"
mkdir -p "${dir_save}"

python main.py --mode Submission --arch WSMCnetEB_S2C3F32 --maxdisp 192 --bn 1 \
               --loadmodel (filepath of pretrained weight) \
               --datas_val "k15-te" --dir_datas_val (dir_root_kitti) \
               --freq_print 1 \
               --dir_save $dir_save \
               2>&1 | tee -a "$LOG"
```

You can also see the example in [demos/kitti_submission.sh](demos/kitti_submission.sh) for details.

### Pretrained Model

| Model | SceneFlow | KITTI |
|---|---|---|
| WSMCnet-S1C1 | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | 
| WSMCnetEB-S2C3 | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | 
| WSMCnetEB-S3C3 | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | [Baidu-pan](https://pan.baidu.com/s/1u1PFM4qpirzOlAgjcI2KLA ) | 

Extraction code：rycn 

## Results on [KITTI 2015 leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |Environment|
|---|---|---|---|---|
| [WSMCnetEB-S2C3]() | 2.13 % | 1.85 % | 0.39 |Nvidia GTX 1070 (pytorch) |
| [PSMNet](https://arxiv.org/abs/1803.08669) | 2.32 % | 2.14 % | 0.41 |Nvidia GTX Titan Xp (pytorch)|
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 | Nvidia GTX Titan X (Pascal) (Caffe)|
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 |Nvidia GTX Titan X (TensorFlow)|
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 |Nvidia GTX Titan X (CUDA, Lua/Torch7)|


## Contacts
wangyf_1991@163.com

Any discussions or concerns are welcomed!
