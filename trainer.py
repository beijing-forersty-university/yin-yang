import math
import os

import torch
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from distributed import DistributedSampler, get_rank, reduce_loss_dict, all_gather
from evaluate import evaluate

from datasets import COCODataset, collate_fn
from datasets.tool import preset_transform
from models import EightTrigrams
from accelerate import Accelerator, find_executable_batch_size


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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


# @find_executable_batch_size(starting_batch_size=16)
def train(epoch, loader, model, optimizer, accelerator, device):
    # nonlocal accelerator  # Ensure they can be used in our context
    # accelerator.free_memory()  # Free all lingering references
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader

    for images, targets, _ in pbar:

        for param in model.parameters():
            param.grad = None
        with accelerator.accumulate(model):
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            optimizer.zero_grad(set_to_none=True)
            _, loss_dict = model(images.tensors, targets=targets)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_box'].mean()
            loss_center = loss_dict['loss_center'].mean()

            loss = loss_cls + loss_box + loss_center

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            scheduler.step()

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
            accelerator.log({"cls": loss_cls, "box": loss_box, "center": loss_center}, step=epoch)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    accelerator = Accelerator(gradient_accumulation_steps=2, log_with="tensorboard", logging_dir="log")
    device = accelerator.device

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
    epoch = 100000
    num_classes = 5
    img_size = 640
    model = EightTrigrams(img_size, batch_size, num_classes)
    model = model.to(device)

    lr_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-5, eps=1e-3, amsgrad=True)
    # 这里可以随便换optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay=1e-4)
    # 定义一个scheduler 参数自己设置
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    # 如果想用带热重启的，可以向下面这样设置
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=10, eta_min=1e-5)

    # Register the LR scheduler
    accelerator.register_for_checkpointing(scheduler)

    # Save the starting state
    # accelerator.save_state("checkpoint/")
    # accelerator.load_state("checkpoint/")
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

    # PATH = 'checkpoint/epoch-93.pt'
    # accelerator.load_state(PATH)

    # ema = EMA(model, 0.999)
    # ema.register()
    config = {
        "num_iterations": epoch,
        "learning_rate": lr_rate,
    }
    accelerator.init_trackers("./log/", config=config)
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    try:
        for epoch in range(config["num_iterations"]):
            torch.backends.cudnn.benchmark = True
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)
            train(epoch, train_loader, model, optimizer, accelerator, device)
            scheduler.step()
            # ema.update()
            # ema.apply_shadow()
            # evaluate

            # ema.restore()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.optimizer.state_dict()  # optimizer is an AcceleratedOptimizer object
            }, "./bundle.pth")
            print(f"save model at ./bundle.pth")
            if epoch % 100 == 0:
                valid(valid_loader, val_dataset, model, device)
    except KeyboardInterrupt:
        print("_-------------------------_")
        print("提前终止")
        accelerator.end_training()

    accelerator.end_training()
