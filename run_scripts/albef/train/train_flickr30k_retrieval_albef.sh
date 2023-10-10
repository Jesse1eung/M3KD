#!/bin/bash
#SBATCH --job-name=fkr   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level4
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --output=./log/flickr/out-%j.out      # output format
#SBATCH --error=./log/flickr/error-out-%j.out      # error output file
#SBATCH --account=research

source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis

python -u -m torch.distributed.run --master_port `shuf -i 29000-49000 -n1` \
--nproc_per_node=4 \
train.py \
--cfg-path lavis/projects/albef/train/ret_flickr30k_ft.yaml \
--options model.pretrained="./lavis/output/ALBEF/Pretrain/202306110027_logits_att_hid_itcT/checkpoint_2.pth" \
run.amp=False run.evaluate=False run.max_epoch=10 run.batch_size_train=32 model.load_finetuned=False