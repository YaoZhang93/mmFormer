# mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation
This is the implementation for the paper:

[mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation](https://arxiv.org/abs/2206.02425)

Accepted to MICCAI 2022

## Abstract

Accurate brain tumor segmentation from Magnetic Resonance Imaging (MRI) is desirable to joint learning of multimodal images. However, in clinical practice, it is not always possible to acquire a complete set of MRIs, and the problem of missing modalities causes severe performance degradation in existing multimodal segmentation methods. In this work, we present the first attempt to exploit the Transformer for multimodal brain tumor segmentation that is robust to any combinatorial subset of available modalities. Concretely, we propose a novel multimodal Medical Transformer (mmFormer) for incomplete multimodal learning with three main components: the hybrid modality-specific encoders that bridge a convolutional encoder and an intra-modal Transformer for both local and global context modeling within each modality; an inter-modal Transformer to build and align the long-range correlations across modalities for modality-invariant features with global semantics corresponding to tumor region; a decoder that performs a progressive up-sampling and fusion with the modality-invariant features to generate robust segmentation. Besides, auxiliary regularizers are introduced in both encoder and decoder to further enhance the model's robustness to incomplete modalities. We conduct extensive experiments on the public BraTS 2018 dataset for brain tumor segmentation. The results demonstrate that the proposed mmFormer outperforms the state-of-the-art methods for incomplete multimodal brain tumor segmentation on almost all subsets of incomplete modalities, especially by an average 19.07% improvement of Dice on tumor segmentation with only one available modality. 

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

## Citation

If you find this code and paper useful for your research, please kindly cite our paper.

```
@article{zhang2022mmformer,
  title={mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation},
  author={Zhang, Yao and He, Nanjun and Yang, Jiawei and Li, Yuexiang and Wei, Dong and Huang, Yawen and Zhang, Yang and He, Zhiqiang and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2206.02425},
  year={2022}
}
```

## Reference

* [TransBTS](https://github.com/Wenxuan-1119/TransBTS)

