# mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation
## Paper

This is the implementation for the paper:

[mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation](https://arxiv.org/pdf/2107.09842.pdf)

Accepted to MICCAI 2022

![image](https://github.com/YaoZhang93/mmFormer/blob/main/figs/overview.pdf)

## Usage. 

* Data Preparation

  - Download the data from [MICCAI 2018 BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html).

  - Put `Training` folder in to `./data` 

  - In `./data`, preprocess the data by

  `python3 preprocess.py`

* Train

  - Train the model by

  `python -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py`

* Test

  - inference on the test data by

  `python test`

  - To inference with missing modalities, please 

## Referrence
* [TransBTS](https://github.com/Wenxuan-1119/TransBTS)