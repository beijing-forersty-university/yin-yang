import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from configs import cfg
import os.path as osp
from datasets import FlowerDataset
from mmdet.apis import inference_detector, show_result_pyplot

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

# from mmcv.utils import collect_env as collect_base_env
# from mmcv.utils import get_git_hash
#
# import mmdet
#
#
# def collect_env():
#     """Collect the information of the running environments."""
#     env_info = collect_base_env()
#     env_info['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
#     return env_info
#
#
# if __name__ == '__main__':
#     for name, val in collect_env().items():
#         print(f'{name}: {val}')