# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/5 14:09
# @Author : liumin
# @File : __init__.py

import copy
import data.transforms.cls_transforms as cls_t
import data.transforms.seg_transforms as seg_t
import data.transforms.det_transforms as det_t
import data.transforms.ins_transforms as ins_t
import data.transforms.keypoint_transforms as kyp_t
import data.transforms.cls_target_transforms as cls_target_t
import data.transforms.seg_target_transforms as seg_target_t
import data.transforms.det_target_transforms as det_target_t
import data.transforms.ins_target_transforms as ins_target_t
import data.transforms.keypoint_target_transforms as kyp_target_t

def build_transforms(dict_name, cfg, mode='train'):
    _params = []

    if dict_name == 'CLS_CLASSES':
        trans = cls_t
    elif dict_name == 'SEG_CLASSES':
        trans = seg_t
    elif dict_name == 'DET_CLASSES':
        trans = det_t
    elif dict_name == 'INS_CLASSES':
        trans = ins_t
    elif dict_name == 'KEYPOINT_CLASSES':
        trans = kyp_t
    elif dict_name == 'FLOWER_CLASSES':
        trans = det_t
    else:
        raise ValueError("Unsupported transforms type: {}".format(dict_name))

    transform_cfg = copy.deepcopy(cfg)
    for t, v in transform_cfg.items():
        t = getattr(trans, t)(**v) if v is not None else getattr(trans, t)()
        if hasattr(t, 'set_load_image_func'):
            t.set_load_image_func()
        _params.append(t)
    return trans.Compose(_params)


def build_targets_transforms(dict_name, cfg, mode='train'):
    _params = []

    if dict_name == 'CLS_CLASSES':
        trans = cls_target_t
    elif dict_name == 'SEG_CLASSES':
        trans = seg_target_t
    elif dict_name == 'DET_CLASSES':
        trans = det_target_t
    elif dict_name == 'INS_CLASSES':
        trans = ins_target_t
    elif dict_name == 'KEYPOINT_CLASSES':
        trans = kyp_target_t
    else:
        raise ValueError("Unsupported transforms type: {}".format(dict_name))

    transform_cfg = copy.deepcopy(cfg)
    for t, v in transform_cfg.items():
        t = getattr(trans, t)(**v) if v is not None else getattr(trans, t)()
        _params.append(t)
    return trans.Compose(_params)