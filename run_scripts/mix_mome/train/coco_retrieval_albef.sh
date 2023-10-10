python -m torch.distributed.run --nproc_per_node=1 \
train.py --cfg-path lavis/projects/albef/train/ret_coco_ft.yaml
