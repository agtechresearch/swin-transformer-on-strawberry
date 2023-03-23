from mmdet.apis import set_random_seed
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmcv import Config
import shutil
import xml.etree.ElementTree as ET
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
import os.path as osp
import copy
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
import mmcv
import torch

CLASSES = ('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
            
NUM_CLASSES = 7
DEVICE = "cuda"

config = 'configs/strawberry/config.py'
checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'

cfg = Config.fromfile(config)
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

cfg.dataset_type = 'StrawberryDataset'
cfg.data_root = 'dataset/'

cfg.model.roi_head.bbox_head.num_classes = NUM_CLASSES

if "mask_head" in cfg.model.roi_head:
    cfg.model.roi_head.mask_head.num_classes = NUM_CLASSES
cfg.load_from = checkpoint
cfg.work_dir = './tutorial_exps'

cfg.log_config.interval = 50
cfg.evaluation.metric = 'mAP'
cfg.evaluation.interval = 10
cfg.checkpoint_config.interval = 5

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.device = DEVICE
cfg.gpu_ids = range(1)

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# print(f'Config:\n{cfg.pretty_text}')


datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)