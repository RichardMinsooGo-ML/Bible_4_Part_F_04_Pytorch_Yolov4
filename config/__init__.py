# ------------------ Dataset Config ------------------
from .data_config.dataset_config import dataset_cfg


def build_dataset_config(args):
    if args.dataset in ['coco', 'coco-val', 'coco-test']:
        cfg = dataset_cfg['coco']
    else:
        cfg = dataset_cfg[args.dataset]

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg


# ------------------ Transform Config ------------------
from .data_config.transform_config import (
    # YOLOv5-Style
    yolov5_pico_trans_config,
    yolov5_nano_trans_config,
    yolov5_small_trans_config,
    yolov5_medium_trans_config,
    yolov5_large_trans_config,
    yolov5_huge_trans_config,
    # YOLOX-Style
    yolox_pico_trans_config,
    yolox_nano_trans_config,
    yolox_small_trans_config,
    yolox_medium_trans_config,
    yolox_large_trans_config,
    yolox_huge_trans_config,
    # SSD-Style
    ssd_trans_config,
)

def build_trans_config(trans_config='ssd'):
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
   
    # SSD-style transform 
    if trans_config == 'ssd':
        cfg = ssd_trans_config

    # YOLOv5-style transform 
    elif trans_config == 'yolov5_pico':
        cfg = yolov5_pico_trans_config
    elif trans_config == 'yolov5_nano':
        cfg = yolov5_nano_trans_config
    elif trans_config == 'yolov5_small':
        cfg = yolov5_small_trans_config
    elif trans_config == 'yolov5_medium':
        cfg = yolov5_medium_trans_config
    elif trans_config == 'yolov5_large':
        cfg = yolov5_large_trans_config
    elif trans_config == 'yolov5_huge':
        cfg = yolov5_huge_trans_config
        
    # YOLOX-style transform 
    elif trans_config == 'yolox_pico':
        cfg = yolox_pico_trans_config
    elif trans_config == 'yolox_nano':
        cfg = yolox_nano_trans_config
    elif trans_config == 'yolox_small':
        cfg = yolox_small_trans_config
    elif trans_config == 'yolox_medium':
        cfg = yolox_medium_trans_config
    elif trans_config == 'yolox_large':
        cfg = yolox_large_trans_config
    elif trans_config == 'yolox_huge':
        cfg = yolox_huge_trans_config

    print('Transform Config: {} \n'.format(cfg))

    return cfg


# ------------------ Model Config ------------------
## YOLO series

from .model_config.yolov4_config import yolov4_cfg

def build_model_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    
    # YOLOv4
    if args.model in ['yolov4', 'yolov4_tiny']:
        cfg = yolov4_cfg[args.model]
        
    return cfg

