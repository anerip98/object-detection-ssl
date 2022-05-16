###
 # @Created by: Xiang Pan
 # @Date: 2022-04-25 21:51:43
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-25 22:27:08
 # @Email: xiangpan@nyu.edu
 # @Description: 
### 
# rm -rf submissions

mkdir submission
mkdir submission/outputs
cp -f ./vicreg/transforms.py ./submissione/
cp -f ./vicreg/utils.py ./submission/
cp -f ./vicreg/coco_utils.py ./submission/
cp -f ./vicreg/coco_eval.py ./submission/
cp -f ./vicreg/dataset.py ./submission/
cp -f ./vicreg/evaluate.py ./submission/
cp -f ./vicreg/engine.py ./submission/            # engine
cp -f ./vicreg/run_evaluate_hpc.py ./submission/  # evluate
cp -f ./README.md ./submission/