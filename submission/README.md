# NYU_DL_Project
Please use https://github.com/Xiang-Pan/NYU_DL_Project/blob/master/submissions/outputs/model_46.pth to download the model, and put the model in ./submissions/outputs/model_46.pth.

## How to Run
### Main Script
python ./submissions_with_full_code/run_evaluate_local.py


### Run it locally with GPU
Please refer to https://github.com/Xiang-Pan/NYU_DL_Project/blob/master/scripts/evaluate.sh

### GCP
For GCP, please check the slurm.
We do not run in GCP, but test the script in the second submission. Please note the model loading checkpoint part and the data loading part.


## Final Validation
```
 IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.101
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.083
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.050
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.119
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.3011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.157
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.345
```
