# Batch size
import torch
import torchvision

from datasets.FlowerDataset import FlowerDataset
from torch.utils.data import DataLoader

train_batch_size = 1
train_data_dir = '../data/flower/train'
train_coco = '../data/flower/train/_annotations.coco.json'


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def train_test_loader(train_data_dir, train_coco, transforms):
    # create own Dataset
    my_dataset = FlowerDataset(root=train_data_dir,
                               annotation=train_coco,
                               transforms=transforms
                               )

    return DataLoader(my_dataset,
                      batch_size=train_batch_size,
                      shuffle=True,
                      num_workers=4,
                      collate_fn=collate_fn)
