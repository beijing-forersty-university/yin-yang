# import os
# import random
#
# import numpy as np
# from PIL import Image
# # pytoech
# import torch
# import torchvision
# import torch.utils.data
# from torch.optim.lr_scheduler import StepLR
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#
# from datasets import COCODataset
# from datasets.tool import preset_transform
# from engine import train_one_epoch, evaluate
# import crack
# import transforms as T
# # allow downloads
# import ssl
#
# from models import EightTrigrams
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# img_size = 640
#
#
# def get_transform():
#     custom_transforms = [torchvision.transforms.ToTensor(),
#                          torchvision.transforms.Resize(img_size),
#                          torchvision.transforms.Normalize((0.432, 0.432, 0.374), (0.275, 0.273, 0.268))]
#     return torchvision.transforms.Compose(custom_transforms)
#
#
# if __name__ == '__main__':
#
#     # params
#     num_classes = 4  # background and person
#     batch_size = 4
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#     # 1. create train and val dataset objects
#     # train_dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
#     # val_dataset = PennFudanDataset('PennFudanPed', get_transform(train=False))
#
#     train_data_dir = 'data/flower/train'
#     train_coco = 'data/flower/train/_annotations.coco.json'
#     test_data_dir = 'data/flower/test'
#     test_coco = 'data/flower/test/_annotations.coco.json'
#
#     # train_dataset = FlowerDataset(root=train_data_dir,
#     #                               annotation=train_coco,
#     #                               transforms=get_transform()
#     #                               )
#     # val_dataset = FlowerDataset(root=test_data_dir,
#     #                             annotation=test_coco,
#     #                             transforms=get_transform()
#     #                             )
#     train_dataset = COCODataset(train_data_dir, train_coco, "train", preset_transform(train=True))
#     val_dataset = COCODataset(test_data_dir, test_coco, "train", preset_transform(train=True))
#     # 2. train, val split
#     torch.manual_seed(1)
#     indices = torch.randperm(len(train_dataset)).tolist()
#     train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
#     val_dataset = torch.utils.data.Subset(val_dataset, indices[-50:])
#
#     # create training and val data loaders
#     train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1,
#                                                     collate_fn=crack.collate_fn)
#     val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1,
#                                                   collate_fn=crack.collate_fn)
#
#     # get the model using our helper function
#     # model = get_instance_segmentation_model(num_classes)
#     model = EightTrigrams(img_size, batch_size, num_classes)
#     # move model to the right device
#     model.to(device)
#
#     # construct an optimizer
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#
#     # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                    step_size=3,
#                                                    gamma=0.1)
#
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         # train for one epoch, printing every 10 iterations
#         train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
#         # update the learning rate
#         lr_scheduler.step()
#         # evaluate on the test dataset
#         evaluate(model, val_data_loader, device=device)
#


import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from distributed import DistributedSampler, get_rank, reduce_loss_dict, all_gather
from evaluate import evaluate


from datasets import COCODataset, collate_fn
from datasets.tool import preset_transform
from models import EightTrigrams


def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


@torch.no_grad()
def valid(loader, dataset, model, device):


    torch.cuda.empty_cache()

    model.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    preds = {}

    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        pred, _ = model(images.tensors, targets, image_sizes= images.sizes)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)


def train(epoch, loader, model, optimizer, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    for images, targets, _ in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()

        loss = loss_cls + loss_box + loss_center
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_box'].mean().item()
        loss_center = loss_reduced['loss_center'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                    f'box: {loss_box:.4f}; center: {loss_center:.4f}'
                )
            )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

    device = 'cuda'

    train_data_dir = 'data/flower/train'
    train_coco = 'data/flower/train/_annotations.coco.json'
    test_data_dir = 'data/flower/test'
    test_coco = 'data/flower/test/_annotations.coco.json'

    # train_dataset = FlowerDataset(root=train_data_dir,
    #                               annotation=train_coco,
    #                               transforms=get_transform()
    #                               )
    # val_dataset = FlowerDataset(root=test_data_dir,
    #                             annotation=test_coco,
    #                             transforms=get_transform()
    #                             )
    train_dataset = COCODataset(train_data_dir, train_coco, "train", preset_transform(train=True))
    val_dataset = COCODataset(test_data_dir, test_coco, "train", preset_transform(train=True))
    batch_size = 2
    epoch = 1000
    num_classes = 4
    img_size = 640
    model = EightTrigrams(img_size, batch_size, num_classes)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=False),
        num_workers=0,
        collate_fn=collate_fn(32),
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=data_sampler(val_dataset, shuffle=False, distributed=False),
        num_workers=0,
        collate_fn=collate_fn(32),
    )

    for epoch in range(epoch):
        train(epoch, train_loader, model, optimizer, device)
        valid(valid_loader, val_dataset, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/epoch-{epoch + 1}.pt',
            )
