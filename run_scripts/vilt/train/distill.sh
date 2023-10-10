CUDA_VISIBLE_DEVICES=2 \
python \
-m torch.distributed.launch \
--master_port `shuf -i 29000-39000 -n1`  \
--nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/vilt/train/distill.yaml
#-m torch.distributed.launch --master_port 29506  --nproc_per_node=1