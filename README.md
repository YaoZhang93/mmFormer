# mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation
This is the implementation for the paper:

[mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation](https://arxiv.org/abs/2206.02425)

Accepted to MICCAI 2022 (Student Travel Award)

## Abstract

Accurate brain tumor segmentation from Magnetic Resonance Imaging (MRI) is desirable to joint learning of multimodal images. However, in clinical practice, it is not always possible to acquire a complete set of MRIs, and the problem of missing modalities causes severe performance degradation in existing multimodal segmentation methods. In this work, we present the first attempt to exploit the Transformer for multimodal brain tumor segmentation that is robust to any combinatorial subset of available modalities. Concretely, we propose a novel multimodal Medical Transformer (mmFormer) for incomplete multimodal learning with three main components: the hybrid modality-specific encoders that bridge a convolutional encoder and an intra-modal Transformer for both local and global context modeling within each modality; an inter-modal Transformer to build and align the long-range correlations across modalities for modality-invariant features with global semantics corresponding to tumor region; a decoder that performs a progressive up-sampling and fusion with the modality-invariant features to generate robust segmentation. Besides, auxiliary regularizers are introduced in both encoder and decoder to further enhance the model's robustness to incomplete modalities. We conduct extensive experiments on the public BraTS 2018 dataset for brain tumor segmentation. The results demonstrate that the proposed mmFormer outperforms the state-of-the-art methods for incomplete multimodal brain tumor segmentation on almost all subsets of incomplete modalities, especially by an average 19.07% improvement of Dice on tumor segmentation with only one available modality. 

![image](https://github.com/YaoZhang93/mmFormer/blob/main/figs/overview.png)

## Usage. 

* Environment Preparation
  * Download the cuda and pytorch from [Google Drive](https://drive.google.com/drive/folders/1x6z7Ot3Xfrg1dokR9cdeoRSKbQJRTpv7?usp=sharing).
  * Set the environment path in `job.sh`.
* Data Preparation
  * Download the data from [MICCAI 2018 BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html).
  * Set the data path in `preprocess.py` and then run `python preprocess.py`.
  * Set the data path in `job.sh`.
* Train
  * Train the model by `sh job.sh`. 

* Test
  * The trained model should be located in `mmFormer/output`. 
  * Uncomment the evaluation command in  `job.sh` and then inference on the test data by `sh job.sh`.
  * The pre-trained [model](https://drive.google.com/file/d/1oKgjXzSfWOG5VT64EE1lfV6rdtjkyC5B/view?usp=sharing) and [log](https://drive.google.com/file/d/165u-MGAiS0_PkExXRkI4KrainRlc_Ibo/view?usp=sharing) are available.

## Citation

If you find this code and paper useful for your research, please kindly cite our paper.

```
@inproceedings{zhang2022mmformer,
  title={mmformer: Multimodal medical transformer for incomplete multimodal learning of brain tumor segmentation},
  author={Zhang, Yao and He, Nanjun and Yang, Jiawei and Li, Yuexiang and Wei, Dong and Huang, Yawen and Zhang, Yang and He, Zhiqiang and Zheng, Yefeng},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022, Proceedings, Part V},
  pages={107--117},
  year={2022},
  organization={Springer}
}
```

## Reference

* [TransBTS](https://github.com/Wenxuan-1119/TransBTS)
* [RFNet](https://github.com/dyh127/RFNet)

