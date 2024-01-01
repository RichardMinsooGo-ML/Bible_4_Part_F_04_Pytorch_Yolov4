import os

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

def build_evluator(args, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(args.data_path, data_cfg['data_name'])

    # Evaluator
    ## VOC Evaluator
    if args.dataset == 'voc':
        evaluator = VOCAPIEvaluator(data_dir  = data_dir,
                                    device    = device,
                                    transform = transform
                                    )
    ## COCO Evaluator
    elif args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(data_dir  = data_dir,
                                     device    = device,
                                     transform = transform
                                     )

    return evaluator
