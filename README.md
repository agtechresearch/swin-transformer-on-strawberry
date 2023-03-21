# swin-transformer-on-strawberry
Object Detection about strawberry using Swin Transformer based on Faster RCNN

- Baseline: https://github.com/open-mmlab/mmdetection/tree/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1


# How to use 

### Install
```
git clone https://github.com/agtechresearch/swin-transformer-on-strawberry
docker build -t mmdetection
docker run --gpus all --shm-size=16g -it -v /{current-absolute-path}/dataset:/mmdetection/dataset mmdetection
```
### Prepare Pretrained Weight File
```
wget https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth 
mv ./mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth ./checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth 
```

### Train & Infer
```
python train.py
python infer.py
```

# Directory of Dataset folder
```
.
|-- image/      // image used in training, inference
|-- inference/          // image saved by model
|-- label.xml
|-- test.txt
|-- train.txt
`-- val.txt

```

# Result
```
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 50/50, 9.9 task/s, elapsed: 5s, ETA:     0s
---------------iou_thr: 0.5---------------
2023-03-21 09:42:24,847 - mmdet - INFO - 
+-------------------+-----+------+--------+-------+
| class             | gts | dets | recall | ap    |
+-------------------+-----+------+--------+-------+
| Strawberry_3      | 5   | 15   | 1.000  | 1.000 |
| Strawberry_2      | 30  | 46   | 0.667  | 0.604 |
| Strawberry_1      | 154 | 7    | 0.032  | 0.023 |
| Flower            | 427 | 289  | 0.468  | 0.460 |
| Green_small_fruit | 157 | 309  | 0.643  | 0.394 |
| Receptacle        | 385 | 455  | 0.574  | 0.503 |
| Before_blooming   | 123 | 3    | 0.000  | 0.000 |
+-------------------+-----+------+--------+-------+
| mAP               |     |      |        | 0.426 |
+-------------------+-----+------+--------+-------+
2023-03-21 09:42:24,890 - mmdet - INFO - Epoch(val) [10][50]	AP50: 0.4260, mAP: 0.4265
```