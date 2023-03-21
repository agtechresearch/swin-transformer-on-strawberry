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
from mmdet.apis import inference_detector
from mmcv.runner import load_checkpoint
import mmcv
import torch

CLASSES = ('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
            
NUM_CLASSES = 7
DEVICE = "cpu"

config = 'configs/strawberry/config.py'
checkpoint = 'tutorial_exps/epoch_9.pth'

device = DEVICE

config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

model = build_detector(config.model)
checkpoint = load_checkpoint(model, checkpoint, map_location=device)


model.cfg = config
model.roi_head.bbox_head.num_classes = NUM_CLASSES
model.CLASSES = NUM_CLASSES

model.to(device)
model.eval()


# Use the detector to do inference
with open("dataset/test.txt", "r") as f:
    while f:
    #for i in range(10):
        fname = f.readline().rstrip()
        img = mmcv.imread(f"dataset/image/{fname}.png")
        result = inference_detector(model, img)
        print(fname, len(result), result)
        model.show_result(img, result, out_file=f"./dataset/inference/result.jpg")
        input()

# Let's plot the result
# model.show_result(img, result, out_file="./dataset/result.jpg")