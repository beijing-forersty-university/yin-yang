# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/22 16:46
# @Author : liumin
# @File : imagenet.py

import glob
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder

"""
    ImageNet
    http://www.image-net.org/
"""

class ImagenetClassification(Dataset):
    def __init__(self,data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(ImagenetClassification, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}

        self._imgs = []
        self._targets = []
        # self.imgs = ImageFolder(root_path)
        if self.stage == 'infer':
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    self._imgs.extend(glob.glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX)))
        else:
            self.cls_label = [d.name for d in os.scandir(data_cfg.IMG_DIR) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    imgs = glob.glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX))
                    self._imgs.extend(imgs)
                    self._targets.extend([self.cls_label.index(fname) for _ in imgs])

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32)
            img_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            _img, _target = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32), self._targets[idx]
            _target = self.encode_map(_target, idx)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_map(self, _target, idx):
        return _target

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/hymenoptera/train'
    dataset = ImagenetClassification(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(0))
    print(len(dataset.cls_label))
    print('finished!')