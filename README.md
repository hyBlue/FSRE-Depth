# FSRE-Depth
This is a Python3 / PyTorch implementation of FSRE-Depth, as described in the following paper:

> **Fine-grained Semantics-aware Representation Enhancement for Self-supervisedMonocular Depth Estimation**
>![overview](https://user-images.githubusercontent.com/30494126/136926985-af8c3651-4503-402b-9677-f623f8b0fd95.PNG)
> *Hyunyoung Jung, Eunhyeok Park and Sungjoo Yoo*
>
> ICCV 2021 (oral)
> 
> [arXiv pdf](http://arxiv.org/abs/2108.08829)


The code was implemented based on [Monodepth2](https://github.com/nianticlabs/monodepth2).
## Setup
This code was implemented under torch==1.3.0 and torchvision==0.4.1, using two NVIDIA TITAN Xp gpus with distrutibted training. Different version may produce different results.
```
pip install -r requirements.txt
```
## Dataset
[KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and pre-computed segmentation images (will be uploaded soon) are required for training.

```
KITTI/
    ├── 2011_09_26/             
    ├── 2011_09_28/                    
    ├── 2011_09_29/
    ├── 2011_09_30/
    ├── 2011_10_03/
    └── segmentation/   # pre-computed segmentation images
```

## Training
For trainin the full model, run the command as below:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port YOUR_PORT_NUMBER train_ddp.py --dapa_path YOUR_KITTI_DATA_PATH
```

## Evaluation

## Download Models
will be uploaded soon

## Reference
Please use the following citation when referencing our work:
```
@InProceedings{Jung_2021_ICCV,
    author    = {Jung, Hyunyoung and Park, Eunhyeok and Yoo, Sungjoo},
    title     = {Fine-Grained Semantics-Aware Representation Enhancement for Self-Supervised Monocular Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12642-12652}
}
```
