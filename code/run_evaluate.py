'''
Created by: Xiang Pan
Date: 2022-04-25 21:56:22
LastEditors: Xiang Pan
LastEditTime: 2022-04-25 22:01:04
Email: xiangpan@nyu.edu
FilePath: /NYU_DL_Project/vicreg/run_evaluate.py
Description: 
'''
# import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils
from engine import evaluate
from dataset import LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cuda:0')

    num_classes = 101
    # train_dataset = LabeledDataset(root='./labeled_data', split="training", transforms=get_transform(train=True))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='./labeled_data', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=12, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)
    state_dict = torch.load('./vicreg/outputs/model_199.pth')['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    evaluate(model, valid_loader, device=device)
    print("Evaluate!")

if __name__ == "__main__":
    main()
