# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/7 8:45
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from detects.fcos_detect import FCOSDetect
from detects.objectbox_detect import ObjectBoxDetect
from detects.yolov5_detect import YOLOv5Detect
from detects.yolov6_detect import YOLOv6Detect
from detects.yolov7_detect import YOLOv7Detect

__all__ = [
    'FCOSDetect',
    'YOLOv5Detect',
    'YOLOv7Detect',
    'ObjectBoxDetect'
]


def build_detect(cfg):
    detect_cfg = deepcopy(cfg)
    name = detect_cfg.pop('name')

    if name == 'FCOSDetect':
        return FCOSDetect(**detect_cfg)
    elif name == 'YOLOv5Detect':
        return YOLOv5Detect(**detect_cfg)
    elif name == 'YOLOv7Detect':
        return YOLOv7Detect(**detect_cfg)
    elif name == 'ObjectBoxDetect':
        return ObjectBoxDetect(**detect_cfg)
    else:
        raise NotImplementedError