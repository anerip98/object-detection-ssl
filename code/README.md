# Training, Finetuning, and Evaluating ResNet-50 for Object Detection

This repository provides code to train a ResNet-50 backbone using VICReg and finetuning this backbone using FasterRCNN/DETR (with hyperparameters as mentioned in the reference paper).

---
To pre-train using VICReg:
```
../scripts/pretrain_vicreg.sh
``` 

To fine-tune using FasterRCNN:
```
../scripts/finetune_fasterrcnn.sh
``` 

To fine-tune using DETR:
```
../scripts/finetune_detr.sh
``` 
--- 

To evaluate:
```
../scripts/evaluate.sh
```
