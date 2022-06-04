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

  - Preprocess the data by

  `python3 preprocess.py`

* Train

  - Train the model by

  `python run/run_training.py 3d_fullres MAMLTrainerV2 32 0`

* Test

  - inference on the test data by

  `python inference/predict_simple.py -i INPUT_PATH -o OUTPUT_PATH -t 32 -f 0 -tr MAMLTrainerV2`

 `MAML` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Please refer to it for more usage.

## Citation

If you find this code and paper useful for your research, please kindly cite our paper.



## Acknowledgement

`MAML` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet).