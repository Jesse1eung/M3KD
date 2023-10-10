#!/bin/bash
#SBATCH --job-name=text  # create a short name for your job
#SBATCH --partition=gpu          # specify the partition name: gpu
#SBATCH --qos=level4
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=20        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8         # number of gpus per node
#SBATCH --output=log/pretrain/out-%j.out      # output format
#SBATCH --error=log/pretrain/error-out-%j.out      # error output file
#SBATCH --account=research

source /scratch/gongzhuocheng/.anaconda3/bin/activate
conda activate lavis
#SBATCH -w lambda-hyperplane04

python \
-m torch.distributed.run \
--master_port `shuf -i 29000-49000 -n1`  \
--nproc_per_node=8 \
train.py --cfg-path lavis/projects/albef/train/pretrain.yaml \
--options model.add_itm=True model.add_itc=True model.add_mlm=False \
model.add_logits=False model.add_att=False model.add_hid=False model.itc_distill=True \
model.caption_distill=False run.amp=False \
model.queue_size=65536 model.alpha=0.4 model.mlm_mask_prob=0.15 run.batch_size_train=256 \
model.has_teacher=False model.has_l_teacher=False model.has_v_teacher=False \
run.accum_grad_iters=1 run.max_epoch=30 run.seed=43 \
run.init_lr=1e-5 \
model.pretrained_stu="./pretrained_ckpt/ck16.pth" \
#run.resume_ckpt_path="./lavis/output/ALBEF/Pretrain/202306150145_logits_att_hid_itcT/checkpoint_3.pth"
#model.pretrained_stu="./lavis/output/ALBEF/Pretrain/202305251209_logits_att_hid_itcT/checkpoint_11.pth"

#python -m torch.distributed.run --nproc_per_node=16 train.py \
# --cfg-path lavis/projects/albef/train/pretrain.yaml
