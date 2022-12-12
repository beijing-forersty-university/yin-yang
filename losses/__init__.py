# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/2 13:21
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from losses.fcos_loss import FCOSLoss


__all__ = ['FCOSLoss']


def build_loss(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'FCOSLoss':
        return FCOSLoss(**loss_cfg)

    else:
        raise NotImplementedError(name)
