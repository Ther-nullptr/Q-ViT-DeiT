wbits=4
abits=4
lr=5e-4
epochs=300
id=4bit_uniform

python -m torch.distributed.launch \
--nproc_per_node=4 --use_env main.py \
--model deit_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_tiny_${id}/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs} \
--data-path /data/ImageNet \
--finetune /home/yujin/projects/models/deit_tiny_patch16_224.pth \
--eval