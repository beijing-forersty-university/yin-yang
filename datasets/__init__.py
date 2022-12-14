import copy
import json
import os.path as osp
from typing import Dict

import mmcv
import numpy as np
from mmdet.datasets.api_wrappers import COCO

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.coco import CocoDataset



@DATASETS.register_module()
class FlowerDataset(CocoDataset):
    CLASSES = ('flower',)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos


# pipeline = [dict(type='LoadImageFromFile'), ]
# data = FlowerDataset("../data/flower/train/_annotations.coco.json", pipeline,
#               img_prefix="../data/flower/train").load_annotations("../data/flower/train/_annotations.coco.json")
#
# for i, j in enumerate(data):
#     print(i, j)