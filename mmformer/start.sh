echo $PWD
pip install setproctitle
# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py > /apdcephfs/share_1290796/medzhang_data/Task/multimodal/mmformer_pad_in_intra_inter_dsv2/log/$(date "+%m%d%H%M%S")_train_job.log 2>&1
python3 test.py > /apdcephfs/share_1290796/medzhang_data/Task/multimodal/mmformer_pad_in_intra_inter_dsv2/log/$(date "+%m%d%H%M%S")_test_job.log 2>&1
# sleep infinity
