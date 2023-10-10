#!/bin/bash
#SBATCH --job-name=c17mocof6-3   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=3-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=./log/out-%j.out      # output format
#SBATCH --error=./log/error-out-%j.out      # error output file


source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis
#nvidia-smi

python -u \
-m torch.distributed.run \
--master_port `shuf -i 29000-49000 -n1`  \
--nproc_per_node=1 \
train.py \
--cfg-path lavis/projects/mix-mome/train/ret_coco_ft.yaml
