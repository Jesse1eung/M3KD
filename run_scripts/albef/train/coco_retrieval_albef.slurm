#!/bin/bash
#SBATCH --job-name=vlmev14   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level4
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=./log/coco/out-%j.out      # output format
#SBATCH --error=./log/coco/error-out-%j.out      # error output file
#SBATCH --account=research


source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis
#SBATCH -w lambda-hyperplane00

#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230309155_itm_itc_mlm/checkpoint_26.pth"

# pretrain 3distill
python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` \
--nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230306171_itm_itc_mlm_logits_att_hid/checkpoint_14.pth" \
run.amp=False run.evaluate=True run.max_epoch=1 run.batch_size_train=32 model.load_finetuned=True \
model.finetuned="./lavis/output/ALBEF/Retrieval_COCO/202306210212/checkpoint_best.pth"
#model.finetuned="./pretrained_ckpt/albef/mscoco.pth"
#model.pretrained="./pretrained_ckpt/albef/ALBEF_4M.pth" \

#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
#--options run.batch_size_train=128 \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/202304090101_itc_logits_att_hid/checkpoint_4.pth" \
#run.amp=True run.evaluate=False model.load_finetuned=False

#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
#--options run.batch_size_train=128 \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/202304080201_logits_att_hid_itcT/checkpoint_6.pth" \
#run.amp=True run.evaluate=False model.load_finetuned=False
#python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` --nproc_per_node=1 \
#train.py \
#--cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
#--options run.batch_size_train=128 \
#--options model.pretrained="./lavis/output/ALBEF/Pretrain/202304080201_logits_att_hid_itcT/checkpoint_8.pth" \
#run.amp=True run.evaluate=False model.load_finetuned=False


