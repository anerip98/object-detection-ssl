start_epoch=100
run_epoch=100
end_epoch=`expr ${start_epoch} + ${run_epoch}`
cp -f ./code/outputs/model_$start_epoch.pth ./code/outputs/model.pth
cp -f ./code/outputs/resnet50_$start_epoch.pth ./code/outputs/resnet50.pth

# make sure the continue checkpoint is named model.pth
# for v100
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
./code/main_vicreg.py \
--data-dir ./unlabeled_data \
--exp-dir ./code/outputs \
--arch resnet50 \
--epochs $end_epoch \
--batch-size 1000 \
--base-lr 0.05
