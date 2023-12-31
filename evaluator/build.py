import os

from evaluator.voc_evaluator import VOCAPIEvaluator

def build_evluator(args, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(args.root, data_cfg['data_name'])

    # Evaluator
    ## VOC Evaluator
    if args.dataset == 'voc':
        evaluator = VOCAPIEvaluator(data_dir  = data_dir,
                                    device    = device,
                                    transform = transform
                                    )
    return evaluator
