import os

try:
    from .voc import VOCDataset
    from .data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform
    from .data_augment.yolov5_augment import YOLOv5Augmentation, YOLOv5BaseTransform

except:
    from voc import VOCDataset
    from data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform
    from data_augment.yolov5_augment import YOLOv5Augmentation, YOLOv5BaseTransform


# ------------------------------ Dataset ------------------------------
def build_dataset(args, data_cfg, trans_config, transform, is_train=False):
    # ------------------------- Basic parameters -------------------------
    data_dir = os.path.join(args.data_path, data_cfg['data_name'])
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_indexs': class_indexs
    }

    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    if args.dataset == 'voc':
        image_sets = [('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')]
        dataset = VOCDataset(img_size     = args.img_size,
                             data_dir     = data_dir,
                             image_sets   = image_sets,
                             transform    = transform,
                             trans_config = trans_config,
                             is_train     = is_train,
                             load_cache   = args.load_cache
                             )
    return dataset, dataset_info


# ------------------------------ Transform ------------------------------
def build_transform(args, trans_config, max_stride=32, is_train=False):
    # Modify trans_config
    if is_train:
        ## mosaic prob.
        if args.mosaic is not None:
            trans_config['mosaic_prob']=args.mosaic if is_train else 0.0
        else:
            trans_config['mosaic_prob']=trans_config['mosaic_prob'] if is_train else 0.0
        ## mixup prob.
        if args.mixup is not None:
            trans_config['mixup_prob']=args.mixup if is_train else 0.0
        else:
            trans_config['mixup_prob']=trans_config['mixup_prob']  if is_train else 0.0

    # Transform
    if trans_config['aug_type'] == 'ssd':
        if is_train:
            transform = SSDAugmentation(img_size=args.img_size,)
        else:
            transform = SSDBaseTransform(img_size=args.img_size,)
        trans_config['mosaic_prob'] = 0.0
        trans_config['mixup_prob'] = 0.0

    elif trans_config['aug_type'] == 'yolov5':
        if is_train:
            transform = YOLOv5Augmentation(
                img_size=args.img_size,
                trans_config=trans_config,
                use_ablu=trans_config['use_ablu']
                )
        else:
            transform = YOLOv5BaseTransform(
                img_size=args.img_size,
                max_stride=max_stride
                )

    return transform, trans_config
