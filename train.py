import math
import os

import torch
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
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

        pred, _ = model(images.tensors, targets, image_sizes=images.sizes)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)


def train(epoch, loader, model, optimizer, scaler, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    for images, targets, _ in pbar:
        # model.zero_grad()

        for param in model.parameters():
            param.grad = None

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            _, loss_dict = model(images.tensors, targets=targets)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()

            loss = loss_cls + loss_box + loss_center
        # loss.backward()
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

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

    # optimizer = optim.Adamax(
    #     model.parameters(),
    #     lr=0.002,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0
    #
    # )
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-04, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
    #                           centered=False)
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[16, 22], gamma=0.1
    # )
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=1e-8,
    #     betas=(0.9, 0.999),
    #     weight_decay=5e-2,
    #     eps=1e-8,
    # )
    lr_rate = 0.1
    lambda1 = lambda epoch: (epoch / 4000) if epoch < 4000 else 0.5 * (
            math.cos((epoch - 4000) / (100 * 1000 - 4000) * math.pi) + 1)
    optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scaler = GradScaler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=False),
        num_workers=0,
        collate_fn=collate_fn(32),
        pin_memory=True
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=data_sampler(val_dataset, shuffle=False, distributed=False),
        num_workers=0,
        collate_fn=collate_fn(32),
        pin_memory=True
    )
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-3,
    #     steps_per_epoch=len(train_loader),
    #     epochs=epoch)
    PATH = 'checkpoint/epoch-93.pt'
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optim'])
    # amp.load_state_dict(checkpoint['amp'])

    for epoch in range(epoch):
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        train(epoch, train_loader, model, optimizer, scaler, device)
        valid(valid_loader, val_dataset, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                {'model': model.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/epoch-{epoch + 1}.pt',
            )
