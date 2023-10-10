# python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/albef/eval/coco_retrieval_eval.yaml
errpath=log/coco/err-$1.log
outpath=log/coco/out-$1.log
CUDA_VISIBLE_DEVICES=1 \
python -u evaluate.py \
--cfg-path lavis/projects/albef/eval/ret_coco_eval.yaml \
--options model.pretrained="./ckpt/202303121935_logits_att_hid/checkpoint_12.pth" \
>$errpath 2>&1 &

#CUDA_VISIBLE_DEVICES=2 \
#python evaluate.py --cfg-path lavis/projects/albef/eval/ret_coco_eval.yaml
#"./ckpt/202303121935_logits_att_hid/checkpoint_12.pth"