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
EVAL = False
DEVICE = "cuda"


def load_xml_to_dict(
        fname="dataset/label.xml"):

    labels = {}

    anno_doc = ET.parse(fname)
    annoD_root = anno_doc.getroot()
    for items in annoD_root.iter("image"):
        record = {
            "filename": items.attrib["name"],
            "image_id": items.attrib["id"],
            "height": int(items.attrib["height"]),
            "width": int(items.attrib["width"]),
            "bboxes": [],
            "bbox_names": [bbox.attrib["label"]
                           for bidx, bbox in enumerate(items.findall("box"))]
        }

        for bidx, bbox in enumerate(items.findall("box")):
            # px = (float(bbox.attrib["xbr"]), float(bbox.attrib["xtl"]))
            # py = (float(bbox.attrib["ybr"]), float(bbox.attrib["ytl"]))
            # print(px)
            record["bboxes"].append(
            #     [np.min(px), np.max(py), np.max(px), np.min(py)]
                [bbox.attrib["xtl"], bbox.attrib["ytl"], bbox.attrib["xbr"], bbox.attrib["ybr"]]
            )

        labels[record["filename"][:-4]] = record

    return labels


IMG_TYPE = "png"


@DATASETS.register_module()
class StrawberryDataset(CustomDataset):

    CLASSES = ('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
               'Green_small_fruit', 'Receptacle', 'Before_blooming')

    def load_annotations(self, ann_file):
        self.labels = load_xml_to_dict()
        # print("hihi")

        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            data_info = self.labels[image_id]
            bbox_names = data_info["bbox_names"]
            bboxes = data_info["bboxes"]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.int_),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.int_),
            )

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


# config = 'configs/strawberry/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
# checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
config = 'configs/strawberry/config.py'
checkpoint = 'checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'

cfg = Config.fromfile(config)
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'StrawberryDataset'
cfg.data_root = 'dataset/'

cfg.data.test.type = 'StrawberryDataset'
cfg.data.test.data_root = 'dataset/'
cfg.data.test.ann_file = 'test.txt'
cfg.data.test.img_prefix = 'image'
cfg.data.test.classes = CLASSES

cfg.data.train.type = 'StrawberryDataset'
cfg.data.train.data_root = 'dataset/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'image'
cfg.data.train.classes = CLASSES

cfg.data.val.type = 'StrawberryDataset'
cfg.data.val.data_root = 'dataset/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'image'
cfg.data.val.classes = CLASSES


# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = NUM_CLASSES
# if isinstance(cfg.model.roi_head.bbox_head, list):
#     for v in cfg.model.roi_head.bbox_head:
#         v.num_classes = NUM_CLASSES
# else:
#     cfg.model.roi_head.bbox_head.num_classes = NUM_CLASSES

if "mask_head" in cfg.model.roi_head:
    cfg.model.roi_head.mask_head.num_classes = NUM_CLASSES

cfg.load_from = checkpoint
cfg.work_dir = './tutorial_exps'

# cfg.optimizer.lr = 0.02 / 8
# cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.device = DEVICE
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')


datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model)
model.CLASSES = datasets[0].CLASSES

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)