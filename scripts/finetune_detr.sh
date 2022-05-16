cd ./code
time=$(date "+%Y%m%d-%H%M%S")
echo $time
mv ./outputs/detr_stats.txt ./outputs/detr_stats_$time.txt

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=47770 \
finetune_detr.py \
--exp-dir ./outputs \
--epochs 200 \
--batch-size 42 \
--lr 1e-4
# --wd 0.0005 \
# --momentum 0.9