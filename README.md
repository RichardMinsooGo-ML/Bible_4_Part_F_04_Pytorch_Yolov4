# Engilish
*  **Theory** : [https://wikidocs.net/167699](https://wikidocs.net/167833) <br>
*  **Implementation** : [https://wikidocs.net/167693](https://wikidocs.net/226335)

# 한글
*  **Theory** : [https://wikidocs.net/187967](https://wikidocs.net/163049) <br>
*  **Implementation** : [https://wikidocs.net/167666](https://wikidocs.net/225928)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOv4:

|    Model    |     Backbone    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-------------|-----------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv4-Tiny | CSPDarkNet-Tiny | 1xb16 |  640  |        31.0            |       49.1        |   8.1             |   2.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_t_coco.pth) |
| YOLOv4      | CSPDarkNet-53   | 1xb16 |  640  |        46.6            |       65.8        |   162.7           |   61.5             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth) |

- For training, we train YOLOv4 and YOLOv4-Tiny with 250 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv5](https://github.com/ultralytics/yolov5).
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.
- For YOLOv4's structure, we use decoupled head, following the setting of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_04_Pytorch_Yolov4.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
# Pre-trained weight for coco

! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_t_coco.pth
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov4_coco.pth
```

## Demo
### Detect with Image
```Shell
# Detect with Image

! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 --cuda \
                 -m yolov4 \
                 --weight /content/yolov4_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show

# See /content/det_results/demos/image
```

### Detect with Video
```Shell
# Detect with Video

! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 --cuda \
                 -m yolov4 \
                 --weight /content/yolov4_coco.pth \
                 -size 640 \
                 -vt 0.4 \
                 --gif
                 # --show
# See /content/det_results/demos/video Download and check the results
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolov4 \
#                  --weight /content/yolov4_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show
```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test YOLOv4
Taking testing YOLOv4 on COCO-val as the example,
```Shell
# Test YOLOv4
! python test.py --cuda \
                 -d coco \
                 --data_path /content/dataset \
                 -m yolov4 \
                 --weight /content/yolov4_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show
# See /content/det_results/coco/yolov4
```

## Evaluate YOLOv4
Taking evaluating YOLOv4 on COCO-val as the example,
```Shell
# Evaluate YOLOv4

! python eval.py --cuda \
                 -d coco-val \
                 --data_path /content/dataset \
                 --weight /content/yolov4_coco.pth \
                 -m yolov4
```

# Training test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

## Train YOLOv4
### Single GPU
Taking training YOLOv4 on VOC as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov4_tiny \
                  -bs 16 \
                  --max_epoch 20 \
                  --wp_epoch 1 \
                  --eval_epoch 10 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

### Multi GPU
Taking training YOLOv4 on VOC as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov4 \
                  -bs 16 \
                  --max_epoch 20 \
                  --wp_epoch 1 \
                  --eval_epoch 10 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

