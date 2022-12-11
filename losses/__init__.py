# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/2 13:21
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from losses.yolov7_loss import Yolov7Loss

# Image Classification


# Semantic Segmentation


# Object Detectiton
from .det import YOLOXLoss

__all__ = [ 'Yolov7Loss']


def build_loss(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'YOLOv7Loss':
        return Yolov7Loss(**loss_cfg)

    else:
        raise NotImplementedError(name)
