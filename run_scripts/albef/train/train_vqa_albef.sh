#!/bin/bash
#SBATCH --job-name=vq14   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level4
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --output=./log/vqa/out-%j.out      # output format
#SBATCH --error=./log/vqa/error-out-%j.out      # error output file
#SBATCH --account=research

source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis

#errpath=log/vqa/$1.log
#outpath=log/vqa/out-$1.log
#CUDA_VISIBLE_DEVICES=0,1 \

python -u -m torch.distributed.run \
--nproc_per_node=4 --master_port `shuf -i 29000-49000 -n1` \
train.py --cfg-path lavis/projects/albef/train/vqa_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/20230306171_itm_itc_mlm_logits_att_hid/checkpoint_14.pth" \
run.amp=False run.evaluate=False model.load_finetuned=False \
run.batch_size_train=128 run.max_epoch=8 model.queue_size=65536 \
#>$errpath 2>&1 &
#python -m torch.distributed.run --nproc_per_node=8 train.py \
