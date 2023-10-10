errpath=log/coco/$1.log
outpath=log/coco/out-$1.log
CUDA_VISIBLE_DEVICES=0 \
python -u -m torch.distributed.run \
--nproc_per_node=1 --master_port `shuf -i 29000-49000 -n1` \
train.py --cfg-path lavis/projects/albef/train/ret_coco_ft.yaml \
--options model.pretrained="./ptm/ckpt/set0/base//checkpoint_12.pth" \
run.amp=False run.evaluate=False model.load_finetuned=False \
run.batch_size_train=128 run.max_epoch=5 model.queue_size=65536 run.init_lr=1e-5 \
>$errpath 2>&1 &
#python -m torch.distributed.run --nproc_per_node=8 train.py \
