from mmdet.datasets import build_dataset
from mmcv import Config
import xml.etree.ElementTree as ET
import os.path as osp
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
import mmcv

CLASSES = ('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
            
NUM_CLASSES = 7
DEVICE = "cpu"

config = 'configs/strawberry/config.py'
checkpoint = 'tutorial_exps/epoch_60.pth'
#checkpoint = 'dataset/epoch_60.pth'

device = DEVICE

cfg = mmcv.Config.fromfile(config)
cfg.model.pretrained = None


# Modify dataset type and path
cfg.dataset_type = 'StrawberryDataset'
cfg.data_root = 'dataset/'

datasets = [build_dataset(cfg.data.test)]
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES
model.cfg = cfg
model.roi_head.bbox_head.num_classes = NUM_CLASSES

checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# if "mask_head" in model.roi_head:
#     model.roi_head.mask_head.num_classes = NUM_CLASSES

model.to(device)
model.eval()

mmcv.mkdir_or_exist(osp.abspath("dataset/inference"))
results=[]
th open("dataset/test.txt", "r") as f:
    while f:
        fname = f.readline().rstrip()
        img = f"/mmdetection/dataset/image/{fname}.png"


        result = inference_detector(model, img)
        results.append(result)
        #print(fname, len(result), result)
        print("\t".join([str(len(r)) for r in result]))
        model.show_result(img, result, 
            out_file=f"./dataset/inference/rs_{fname}.jpg")
