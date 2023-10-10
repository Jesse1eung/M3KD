#!/bin/bash
#SBATCH --job-name=sstnovis   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level1
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=3-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=./log/glue/out-%j.out      # output format
#SBATCH --error=./log/glue/error-out-%j.out      # error output file
#SBATCH --account=research


source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis

#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_18.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_20.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_22.pth"

# sst pretrain +distill
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_12.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_14.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_16.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_18.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_20.pth"
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_22.pth"
#
## sst mlm -visual info
##python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
##train.py \
##--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
##--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_8.pth"
#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_10.pth"
##
## sst -mlm
#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1`  --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/glue_sst2_ft.yaml \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc/checkpoint_10.pth"



# CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python -m torch.distributed.run --nproc_per_node=8 --master_port 47770 train.py --cfg-path lavis/projects/albef/train/snli_ve_ft.yaml
#-m torch.distributed.run --nproc_per_node=8