# mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation
## Paper

This is the implementation for the paper:

[mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation](https://arxiv.org/pdf/2107.09842.pdf)

Accepted to MICCAI 2022

![image](https://github.com/YaoZhang93/mmFormer/blob/main/figs/overview.png)

## Usage. 

* Data Preparation

  - Download the data from [MICCAI 2018 BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html).

  - Put `Training` folder in  `./data` 

  - In `./data`, preprocess the data by `python preprocess.py`

* Train

  - Train the model by

  `python -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py`

* Test

  - inference on the test data by

  `python test`

  - To inference with missing modalities, please refer to line 201 in [`BraTS.py`](https://github.com/YaoZhang93/mmFormer/blob/main/mmformer/data/BraTS.py)
  
    `missing_modal_list.append(MISSING_MODAL)`
  
    MISSING_MODAL is a list of missing modalities and each modality is denoted by a number.
  
    `0: FLAIR, 1:T1CE, 2:T1, 3:T2`

## Reference
* [TransBTS](https://github.com/Wenxuan-1119/TransBTS)

