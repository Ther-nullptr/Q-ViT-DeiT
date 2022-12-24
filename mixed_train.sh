wbits=8
abits=8
lr=2e-04
wd=0.05
epochs=300
lbd=1e-1
budget=21.455
id=4bit_mixed

python -m torch.distributed.launch \
--nproc_per_node=4 --use_env main.py \
--model deit_tiny_patch16_224_mix \
--batch-size 32 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--weight-decay ${wd} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--bitops-scaler ${lbd} \
--budget ${budget} \
--stage-ratio 0.9 \
--dist-eval \
--mixpre \
--head-wise \
--output_dir results/deit_tiny_${id}/${wbits}w${abits}a_bs512_baselr${lr}_wd${wd}_ft${epochs}_lbd${lbd} \
--data-path /data/ImageNet \
--finetune /home/yujin/projects/models/deit_tiny_patch16_224.pth \
--eval
