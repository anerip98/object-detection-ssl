cd ./code
python -m torch.distributed.launch --nproc_per_node=6 finetune.py --exp-dir ./outputs --epochs 400 --batch-size 12 --lr 0.005 --wd 0.0005 --momentum 0.9
