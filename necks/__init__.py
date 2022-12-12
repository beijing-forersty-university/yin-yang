# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:00
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from .fcos_fpn import FCOSFPN

__all__ = [

    'FCOSFPN',

]


def build_neck(cfg):
    neck_cfg = deepcopy(cfg)
    name = neck_cfg.pop('name')

    if name == 'FCOSFPN':
        return FCOSFPN(**neck_cfg)

    else:
        raise NotImplementedError
