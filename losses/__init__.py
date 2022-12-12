# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/2 13:21
# @Author : liumin
# @File : __init__.py

from copy import deepcopy


# from losses.yolov7_loss import YOLOLoss
from losses.yolovp_loss import YolopLoss

__all__ = ['YolopLoss']


def build_loss(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'YOLOv7Loss':
        return YolopLoss(**loss_cfg)

    else:
        raise NotImplementedError(name)
