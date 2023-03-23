_base_ = [
    # https://github.com/open-mmlab/mmdetection/issues/1480
    # '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=100)

model = dict(
    
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]



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

            wow =[ float(v)  for v in [bbox.attrib["xtl"], bbox.attrib["ytl"], bbox.attrib["xbr"], bbox.attrib["ybr"]]]

            if wow[0] > wow[2]:
                wow[0], wow[2] = wow[2], wow[0]
            if wow[1] > wow[3]:
                wow[1], wow[3] = wow[3], wow[1]
                
            record["bboxes"].append(wow)

        labels[record["filename"][:-4]] = record

    return labels


import shutil
import xml.etree.ElementTree as ET
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
import os.path as osp
import copy
import mmcv

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



data = dict(
    train=dict(pipeline=train_pipeline,
        type='StrawberryDataset',
         data_root='dataset/',
        ann_file='train.txt',
        img_prefix='image',
        classes=('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
        ),

    val=dict(pipeline=test_pipeline,
        type='StrawberryDataset',
                data_root='dataset/',
        ann_file='val.txt',
        img_prefix='image',
        classes=('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
        ),
    test=dict(pipeline=test_pipeline,
        type='StrawberryDataset',
        data_root='dataset/',
        ann_file='test.txt',
        img_prefix='image',
        classes=('Strawberry_3', 'Strawberry_2', 'Strawberry_1', 'Flower',
            'Green_small_fruit', 'Receptacle', 'Before_blooming')
))
