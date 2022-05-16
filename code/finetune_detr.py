from pathlib import Path
import argparse
import os
import sys
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T

# from dataset import LabeledDataset
from dataset_detr import LabeledDataset
from engine_detr import train_one_epoch, evaluate
from utils import init_distributed_mode, collate_fn


def get_arguments():
    parser = argparse.ArgumentParser(description="Finetune FasterRCNN", add_help=False)

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Optim
    parser.add_argument("--epochs", type=int, default=20,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=8,
                        help='Batch size')
    parser.add_argument("--lr", type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='Momentum')
    parser.add_argument("--wd", type=float, default=0.0005,
                        help='Weight decay')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrConfig
from torchvision import transforms
import torch.nn as nn

class DetrModel(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        # self.config = DetrConfig.from_pretrained("./config.json")
        self.config = DetrConfig()
        self.config.auxiliary_loss = True
        self.config.num_labels = num_classes
        self.feature_extractor = DetrFeatureExtractor()
        self.od_model = DetrForObjectDetection(self.config)

    def forward(self, images, targets=None):
        tensor2img = torchvision.transforms.ToPILImage()
        image_sizes = [torch.as_tensor(img.shape[-2:]) for img in images]
        image_sizes = torch.stack(image_sizes).to(self.od_model.device)
        images = [tensor2img(img) for img in images]
        labels = targets
        features = self.feature_extractor(images=images, return_tensors="pt")
        for k,v in features.items():
            features[k] = v.to(self.od_model.device)

        if targets is None:
            # for inference
            outputs = self.od_model(**features)
            outputs = self.feature_extractor.post_process(outputs, image_sizes)
        else:
            outputs = self.od_model(**features, labels=labels)


            # outputs = {"logits": model_outputs["logits"].cpu(), "boxes": model_outputs["pred_boxes"].cpu()}
            # print("outputs!!!!!!!!!!!!!!",outputs)

        return outputs

def get_model(args, num_classes, load_epoch=0):
    model = DetrModel(num_classes=num_classes)

    load_resnet50 = False
    if load_resnet50:
        state_dict = torch.load(args.exp_dir / 'model_199.pth')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_name = k[7:]
            new_state_dict[new_name] = state_dict[k]
        model.load_state_dict(new_state_dict, strict=False)
    else:
        load_epoch = 0
        state_dict = torch.load(args.exp_dir / f'detr_model_{load_epoch}.pth')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_name = k[7:]
            new_state_dict[new_name] = state_dict[k]
        model.load_state_dict(new_state_dict, strict=False)
    return model


def train(args, model, optimizer, lr_scheduler, train_loader, valid_loader, device, start_epoch, stats_file, run_evaluate=False):
    start_time = last_logging_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if not run_evaluate:
            train_one_epoch(args, model, optimizer, train_loader, device, epoch, stats_file, start_time, last_logging_time)
        # (model, optimizer, data_loader, device, epoch, print_freq=1):
        # train_one_epoch(model, optimizer, train_loader, device, epoch, last_logging_time)
        # lr_scheduler.step()
 
        if not run_evaluate:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )       
            torch.save(state, args.exp_dir / f"detr_model_{epoch}.pth")

        evaluate(model, valid_loader, device=device)


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    stats_file = open(args.exp_dir / "detr_stats.txt", "w", buffering=1)
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "detr_stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
    
    num_classes = 101
    train_dataset = LabeledDataset(root='../labeled_data', split="training", transforms=get_transform(train=True))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    per_device_batch_size = args.batch_size // args.world_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size, num_workers=args.num_workers, 
                                               pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    valid_dataset = LabeledDataset(root='../labeled_data', split="validation", transforms=get_transform(train=False))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=per_device_batch_size, num_workers=args.num_workers, 
                                               pin_memory=True, sampler=valid_sampler, collate_fn=collate_fn)
    
    model = get_model(args, num_classes).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    lr_scheduler = None
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        start_epoch = 0
        # ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        # start_epoch = ckpt["epoch"]
        # model.load_state_dict(ckpt["model"])
        # optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    train(args, model, optimizer, lr_scheduler, train_loader, valid_loader, gpu, start_epoch, stats_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Finetuning script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
    
