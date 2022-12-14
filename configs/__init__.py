from mmdet.apis import set_random_seed
from mmcv import Config

cfg = Config.fromfile('./configs/flower_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'FlowerDataset'
cfg.data_root = 'data/flower/'

cfg.data.test.type = 'FlowerDataset'
cfg.data.test.data_root = '       '
cfg.data.test.ann_file = '_annotations.coco.json'
cfg.data.test.img_prefix = ''

cfg.data.train.type = 'FlowerDataset'
cfg.data.train.data_root = 'data/flower/test/'
cfg.data.train.ann_file = '_annotations.coco.json'
cfg.data.train.img_prefix = ''

cfg.data.val.type = 'FlowerDataset'
cfg.data.val.data_root = 'data/flower/valid/'
cfg.data.val.ann_file = '_annotations.coco.json'
cfg.data.val.img_prefix = ''
cfg.data.workers_per_gpu = 0
cfg.data.samples_per_gpu = 1
# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'bbox'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')
