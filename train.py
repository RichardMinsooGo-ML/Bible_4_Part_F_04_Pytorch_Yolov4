from __future__ import division

import os
import random
import numpy as np
import argparse
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import compute_flops

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Model Components -----------------
from models.detectors import build_model

# ----------------- Train Components -----------------
from engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection LAB')
    # Random seed
    parser.add_argument('--seed', default=42, type=int)

    # GPU
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    
    # Image size
    parser.add_argument('-size', '--img_size', default=640, type=int, help='input image size')
    parser.add_argument('--eval_first', action='store_true', default=False, help='evaluate model before training.')
    
    # Outputs
    parser.add_argument('--tfboard', action='store_true', default=False, help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, help='path to save weight')
    parser.add_argument('--vis_tgt', action="store_true", default=False, help="visualize training data.")
    parser.add_argument('--vis_aux_loss', action="store_true", default=False, help="visualize aux loss.")
    
    # Mixing precision
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False, help="Adopting mix precision training.")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=150, type=int, help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int, help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--no_aug_epoch', default=20, type=int, help='cancel strong augmentation.')

    # Model
    parser.add_argument('-m', '--model', default='yolov1', type=str, help='build yolo')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float, help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int, help='topk candidates dets of each level before NMS')
    parser.add_argument('-p', '--pretrained', default=None, type=str, help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str, help='keep training')
    parser.add_argument('--nms_class_agnostic', action='store_true', default=False, help='Perform NMS operations regardless of category.')

    # Dataset
    parser.add_argument('--data_path', default='/content/dataset', help='data root')
    parser.add_argument('-d', '--dataset', default='voc', help='coco, voc, widerface, crowdhuman')
    parser.add_argument('--load_cache', action='store_true', default=False, help='Path to the cached data.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    
    # Train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='Multi scale')
    parser.add_argument('--ema', action='store_true', default=False, help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float, help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float, help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float, help='mixup augmentation.')
    parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False, help='distributed training')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, help='use sybn.')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode.')

    return parser.parse_args()


def fix_random_seed(args):
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # ---------------------------- Build DDP ----------------------------
    local_rank = local_process_rank = -1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    world_size = distributed_utils.get_world_size()
    print("LOCAL RANK: ", local_rank)
    print("LOCAL_PROCESS_RANL: ", local_process_rank)
    print('WORLD SIZE: {}'.format(world_size))

    # ---------------------------- Build CUDA ----------------------------
    if args.cuda and torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---------------------------- Fix random seed ----------------------------
    fix_random_seed(args)

    # ---------------------------- Build config ----------------------------
    data_cfg  = build_dataset_config(args)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # ---------------------------- Build model ----------------------------
    ## Build model
    model, criterion = build_model(
        args,
        model_cfg,
        device,
        data_cfg['num_classes'],
        True
        )
    model = model.to(device).train()
    model_without_ddp = model
    
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    ## Calcute Params & GFLOPs
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      img_size=args.img_size,
                      device=device)
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # ---------------------------- Build Trainer ----------------------------
    trainer = build_trainer(args, data_cfg, model_cfg, trans_cfg, device, model_without_ddp, criterion, world_size)

    # --------------------------------- Train: Start ---------------------------------
    ## Eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval(model_eval)

    ## Satrt Training
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # Empty cache after train loop
    del trainer
    if args.cuda:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
