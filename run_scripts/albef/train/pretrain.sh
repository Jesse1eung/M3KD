errpath=log/pretrain/$1.log
outpath=log/pretrain/out-$1.log
CUDA_VISIBLE_DEVICES=5,7 \
python -u -m torch.distributed.run \
--nproc_per_node=2 --master_port `shuf -i 29000-49000 -n1` \
train.py --cfg-path lavis/projects/albef/train/pretrain.yaml \
--options model.add_itm=False model.add_itc=False model.add_mlm=False \
model.add_logits=True model.add_att=True model.add_hid=True model.itc_distill=True \
model.caption_distill=False \
run.amp=False run.accum_grad_iters=1 model.queue_size=65536 run.batch_size_train=256 \
>$errpath 2>&1 &
#run.init_lr=5e-5 run.min_lr=5e-5 \
#run.resume_ckpt_path="./ptm/ckpt/set0/base_itcT/checkpoint_3.pth" \
#run.init_lr=1.5e-4 run.warmup_lr=5e-7 run.min_lr=5e-7 \
#python -m torch.distributed.run --nproc_per_node=16 train.py \
# --cfg-path lavis/projects/albef/train/pretrain.yaml
