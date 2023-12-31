#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
# YOLO series

from .yolov4.build import build_yolov4

# build object detector
def build_model(args, 
                model_cfg,
                device, 
                num_classes=80, 
                trainable=False,
                deploy=False):
    
    # YOLOv4   
    if args.model in ['yolov4', 'yolov4_tiny']:
        model, criterion = build_yolov4(
            args, model_cfg, device, num_classes, trainable, deploy)

    if trainable:
        # Load pretrained weight
        if args.pretrained is not None:
            print('Loading COCO pretrained weight ...')
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        # keep training
        if args.resume and args.resume != "None":
            print('keep training: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)
            del checkpoint, checkpoint_state_dict

        return model, criterion

    else:      
        return model