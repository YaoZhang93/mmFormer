#!/bin/bash
pythonname='pytorch_1.2.0a0+8554416-py36tf'

dataname='BRATS2018'
pypath=$pythonname
cudapath=cuda-9.0
datapath=${dataname}_Training_none_npy
savepath=output
 
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib64:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.6
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

$PYTHON train.py --batch_size=8 --datapath $datapath --savepath $savepath --num_epochs 1000 --dataname $dataname

#eval:
#resume=output/model_last.pth
#$PYTHON train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 0 --dataname $dataname --resume $resume