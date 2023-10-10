#!/bin/bash
#SBATCH --job-name=ev16   # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level4
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --output=./log/coco/out-%j.out      # output format
#SBATCH --error=./log/coco/error-out-%j.out      # error output file
#SBATCH --account=research


source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis

python -u \
-m torch.distributed.run \
--master_port `shuf -i 29000-49000 -n1`  \
--nproc_per_node=1 \
evaluate.py --cfg-path lavis/projects/albef/eval/ret_coco_eval.yaml

#CUDA_VISIBLE_DEVICES=2 \
#python evaluate.py --cfg-path lavis/projects/albef/eval/ret_coco_eval.yaml
