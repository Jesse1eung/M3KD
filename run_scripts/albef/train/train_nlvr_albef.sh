python -m torch.distributed.run --nproc_per_node=8 train.py \
--cfg-path lavis/projects/albef/train/nlvr_ft.yaml
