#CUDA_VISIBLE_DEVICES=1 \
#python \
#train.py \
#--cfg-path lavis/projects/mix-mome/train/mix_distill.yaml
#-m torch.distributed.launch --master_port 29506  --nproc_per_node=1

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python \
-m torch.distributed.launch \
--master_port `shuf -i 29000-39000 -n1`  \
--nproc_per_node=4 \
train.py \
--cfg-path lavis/projects/mix-mome/train/mix_distill.yaml
