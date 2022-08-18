# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from engine import evaluate, train_one_epoch
from models import build_model
from uwb_dataset import UWBDataset, detection_collate, detection_collate_val

from custom_lr_scheduler import CosineAnnealingWarmUpRestarts
torch.backends.cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float) # 0.0004
    parser.add_argument('--lr_min', default=2e-5, type=float) # 0.00002
    parser.add_argument('--batch_size', default=128, type=int) # 32
    parser.add_argument('--weight_decay', default=1e-4, type=float) # 0.0001(1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                       help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--frozen_ftr_weights', type=str, default=None,
                        help="Path to the pretrained ftr model. If set, only the mask head will be trained")
                                            
    # * Backbone
    parser.add_argument('--backbone', type=str, default='convnext', choices=('convnext'),
                        help="model_size")
    parser.add_argument('--res4dim', default=18, type=int,
                        help="convnext res4 # of stages")  
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('none', 'sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # * Transformer
    parser.add_argument('--enc_layers', default=0, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=15, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', default=False,
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=10, type=float, 
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=8, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficient
    parser.add_argument('--bbox_loss_coef', default=10, type=float) 
    parser.add_argument('--giou_loss_coef', default=8, type=float) 
    parser.add_argument('--pose_loss_coef', default=10, type=float) 
    parser.add_argument('--feature_loss_coef', default=10, type=float) 
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    parser.add_argument('--output_dir', default='./weights/ver4/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # rf dataset parameter
    parser.add_argument('--cutoff', type=int, default=256, 
            help='cut off the front of the input data, --> length = 1024 - cutoff')  
    parser.add_argument('--stack_num', type=int, default=1,
                help = 'number of frame to stack')
    parser.add_argument('--frame_skip', type=int, default=1, 
                help = 'number of frame to skip when stacked')
    parser.add_argument('--stack_avg', type=int, default=64,
                help='use average filtering')

    # evaluate paramater
    parser.add_argument('--vis', action='store_true', default=False,
                help='visualize the image for debugging')

                
    parser.add_argument('--img_dir', default=None, type=str,
                        help='path where to save evaluation image')
    parser.add_argument('--box_threshold', default=0.5, type=float,
                        help="confidence threshold to print out")
    parser.add_argument('--test_dir', default='9,6,38', 
                        type=lambda s: [int(item) for item in s.split(',')])
    
    # freebies parameter
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                help="lr_scheduler (linear or cosine)")
    parser.add_argument('--drop_prob', default=0.3, type=float,
                        help="drop_path drop probability")
    parser.add_argument('--dropblock_prob', default=0.2, type=float,
                        help="dropblock drop probability")
    parser.add_argument('--drop_size', type=int, default=4,
                help = 'size of drop block')

    # * Pose estimation
    parser.add_argument('--pose',  type=str, default=None,
                help="pose method (dr, heatmap)")
    parser.add_argument('--dr_size', type=int, default=512, 
            help='DR size for pose estimation dr')

    # * feature train
    parser.add_argument('--signal_ratio', default=0.75, type=float, #0.3
                        help="signal : vc ratio")
    parser.add_argument('--box_feature', default='x', type=str, choices=('x','16'),
                        help="get img featuremap for train person detection network")
    parser.add_argument('--soft_nms', action='store_true', default=False,
                help='use soft_nms when postprocessing')          

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("torch version ", torch.__version__)
    if args.frozen_weights is not None:
        assert args.pose is not None, "Frozen training is meant for pose estimation only"

 
    print(args)
    print(torch.backends.cudnn.benchmark)
    
    output_dir = Path(args.output_dir)
    if args.output_dir and not args.eval and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            for arg in vars(args):
                f.write(arg + " = " + str(getattr(args, arg)) + "\n")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)  
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]

    if args.lr_scheduler == 'linear':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.1) # after {lr_drop} epoch, lr drop to 1/10 
    elif args.lr_scheduler =='cosine':
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr_min, weight_decay=args.weight_decay)
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=args.lr,  T_up=10, gamma=0.5)
    
    
    if args.frozen_weights is not None:
        print("Load box model = ", args.frozen_weights)
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.rpet.load_state_dict(checkpoint['model'])
    
    if args.frozen_ftr_weights is not None:
        print("Load ftr_backbone model = ", args.frozen_ftr_weights)
        checkpoint = torch.load(args.frozen_ftr_weights, map_location='cpu')
        model_without_ddp.backbone.load_state_dict(checkpoint['model'])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        for key in list(checkpoint['model'].keys()):
            checkpoint['model'][key.replace('rftr', 'rpet')] = checkpoint['model'].pop(key)

        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    dataset_train = UWBDataset(mode='train', args=args)
    dataset_val = UWBDataset(mode='test', args=args)
    
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=detection_collate, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=detection_collate_val, num_workers=args.num_workers, pin_memory=True)
    

    # Dataset Debug
    if args.eval:
        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, \
                            val_vis=args.vis, is_pose=args.pose, img_dir=args.img_dir, \
                            boxThrs=args.box_threshold, epoch=-1, dr_size=args.dr_size,
                            feature_list=args.box_feature, soft_nms=args.soft_nms,
        )
        return
    print("Start training")
    save_freq = 10 if args.pose is None else 5 
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # train one epoch - engine.py
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, is_pose=args.pose)
            
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            #if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
            if (epoch + 1) % save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
      
        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, \
                                val_vis=args.vis, is_pose=args.pose, epoch=epoch, \
                                    boxThrs=args.box_threshold, dr_size=args.dr_size, \
                                    feature_list=args.box_feature
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RPET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
