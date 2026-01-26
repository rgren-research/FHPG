# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
from typing import NoReturn
import numpy as np
import time
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model   # This error message does not affect execution.
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from utils import NativeScaler_No_Optimizer

from dataset import build_dataset
from engine import train_one_epoch, evaluate, prune_one_shot
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import json

from logger import create_logger
from pruner import prepare_pruning_list, BaseUnitPruner, get_model_complexity_info, pruning_config, Controller

from transformers import (
    ViTConfig,
    ViTModel,
    DeiTConfig,
    DeiTModel
    )
from pruner.fisher import collect_mask_grads
from utils import get_pruning_schedule
# from pruner.search import search_mac, search_latency
from pruner.search import search_optimal_mask, search_optimal_latency
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from evaluate.ViT import test_accuracy
from pruner.rearrange import rearrange_mask
# from pruner.rescale import rescale_mask
from torchvision import transforms
import torch.nn as nn
from models import *
from config import Config
from pruner.fisher import get_ffn2, register_mask, get_mha, register_head_mask


'''
args1：      
--batch-size 32 
--epochs 1 
--data-path  /imagenet

args2：
--batch-size 32
--epochs 100
--model vit-base-patch16-224
--data-path  
--data-set CIFAR


args3,Post-training
--model_name facebook/deit-base-patch16-224 --task_name CIFAR --batch-size 16 --epochs 1 --ckpt_dir /deit-base --constraint 0.5

Pruning + Quantization of execution parameters:
--model_name facebook/deit-base-patch16-224 --quant --ptf --lis --quant-method minmax  --task_name CIFAR --batch-size 16 --epochs 1 --ckpt_dir /deit-base --constraint 0.5
or
--model_name deit_base --quant --ptf --lis --quant-method minmax  --task_name CIFAR --batch-size 16 --epochs 1 --ckpt_dir /deit-base --constraint 0.5
'''

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    # parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    parser.add_argument("--ckpt_dir", type=str, required=True) # a fast ..
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # prunning parameters
    parser.add_argument('--pruning_method', type=str, default='2',
                        help='pruning metric when pruning single component')
    parser.add_argument('--pruning_layers', type=int, default=3,
                        help='which components for pruning, 0(attention head), 1(hidden dimension), 2(embedding dimension), 3(all)')
    parser.add_argument('--pruning_feed_percent', type=float, default=0.1,
                        help='how much data for calculating metric and hessian, 0.1 means 10%')
    parser.add_argument('--pruning_per_iteration', type=int, default=20)
    parser.add_argument('--maximum_pruning_iterations',type=int, default=24000)
    parser.add_argument('--pruning_flops_percentage', type=float, default=0.50,
                        help='set the flops percentage for pruning')
    parser.add_argument('--pruning_flops_threshold', type=float, default=0.001,
                        help='when threshold> abs(pruned_percentage) > 0, exit prune')                    
    parser.add_argument('--need_hessian', action='store_true')
    parser.add_argument('--hessian_embed', type=float, default=6.0,
                        help='used for adjusting the hessian sum of embedding dim')
    
    parser.add_argument('--pruning_momentum', type=float, default=0.9)
    parser.add_argument('--pruning_silent', action='store_true', 
                        help='whether to show some pruning details')
    parser.add_argument('--pruning_pickle_from', type=str, default='',
                        help='load pre-calculated metric and hessian information if you already get')
    parser.add_argument('--pruning_normalize_by_layer', action='store_true',
                        help='whether to normalize the metric')
    parser.add_argument('--pruning_normalize_type', type=int, default=2,
                        help='1(l1 normalization), 2(l2 normalization)')
    parser.add_argument('--pruning_protect', type=bool, default=True,
                        help='to prevent every layer from pruning all neurons')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None, help='used in evaluation')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, 
                        help="distillation loss type")
    parser.add_argument('--distillation-alpha', default=0.5, type=float,
                        help="distillation loss weight")
    parser.add_argument('--distillation-tau', default=1.0, type=float, 
                        help="distillation softmax temperature")

    # * Finetuning params
    parser.add_argument('--finetune', default=None, help='finetune from checkpoint')
    parser.add_argument('--finetune_op', type=int, default=0,
                        help='1(only finetune), 2(only prune)')

    # Dataset parameters
    parser.add_argument('--data-path', default='/data', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--not_imagenet_default_mean_and_std', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_freq', default=50, type=int,
                        help='save checkpoint frequency')                        
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # logging 
    parser.add_argument('--log', type=str, default='same',
                        help='log file path')
    parser.add_argument('--log_period', type=int, default=10,
                        help='log period in training or finetuning')

    # Pruning
    # parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=[
        "CIFAR",
        "IMNET",
        "INAT",
        "INAT19",
    ])
    parser.add_argument("--metric", type=str, choices=[
        "mac",
        "latency",
    ], default="mac")
    parser.add_argument("--constraint", type=float, required=True,
                        help="MAC/latency constraint relative to the original model",
                        )
    parser.add_argument("--mha_lut", type=str,
                        default='/FHPG/FHPG-main/output/mha_lut.pt')
    parser.add_argument("--ffn_lut", type=str,
                        default='/FHPG/FHPG-main/output/ffn_lut.pt')
    parser.add_argument("--num_samples", type=int, default=2048)
    # parser.add_argument("--seed", type=int, default=0)
    # Quantification
    parser.add_argument('--model_name', choices=[
                            'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                            'vit_large', 'swin_tiny', 'swin_small', 'swin_base'], help='model')
    parser.add_argument('--quant', default=False, action='store_true')
    parser.add_argument('--ptf', default=False, action='store_true')
    parser.add_argument('--lis', default=False, action='store_true')
    parser.add_argument('--quant-method', default='minmax', choices=['minmax', 'ema', 'omse', 'percentile'])
    parser.add_argument('--calib-iter', default=10, type=int)
    parser.add_argument('--val-batchsize',
                        default=100,
                        type=int,
                        help='batchsize of validation set')
    parser.add_argument('--print-freq',
                        default=100,
                        type=int,
                        help='print frequency')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.log == 'same':
        args.log = args.output_dir
    log_file = os.path.join(args.log, '{}.log'.format(timestamp))
    logger = create_logger(output_dir=log_file, dist_rank=utils.get_rank())
    logger.warning("GPU {} run on process {}".format(utils.get_rank(),os.getpid()))
    args_txt='----------running configuration----------\n'
    for key, value in vars(args).items():
        args_txt+=('{}: {} \n'.format(key, str(value)))
    logger.info(args_txt)
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
  
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    '''
        sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )
    '''
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # logger.info(f"load  model: {args.model}")
    # model = create_model(
    #     args.model,
    #     pretrained=False,
    #     num_classes=args.nb_classes,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    # )
    logger.info(f"load model: {args.ckpt_dir}")
    # config = ViTConfig.from_pretrained(args.ckpt_dir)
    # model = ViTModel.from_pretrained(args.ckpt_dir, config=config)
    config = DeiTConfig.from_pretrained(args.ckpt_dir)
    model = DeiTModel.from_pretrained(args.ckpt_dir, config=config)
    seq_len = int(config.image_size / config.patch_size) ** 2


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint
        if 'model' in checkpoint_model.keys():
            checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

        if hasattr(model, 'finetune'):
            model.finetune()
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # in order to create optimizer without gate params, which should be freezed during fine-tuning
    for name, m in model.named_parameters():
        if "gate" in name:
            m.requires_grad = False
            logger.info("skipping parameter:{} shape:{}".format(name, m.shape))
    optimizer = create_optimizer(args, model_without_ddp)
    # recover gate gradient calculate
    for name, m in model.named_parameters():
        if "gate" in name:
            m.requires_grad = True

    loss_scaler = NativeScaler()
    loss_scaler_no_optimizer = NativeScaler_No_Optimizer()  # *

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )


    output_dir = Path(args.output_dir)
    if args.resume:
        logger.info('resume training from:{}'.format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        if hasattr(model, 'finetune'):
            model.finetune()
    if args.eval:
        test_stats = evaluate(args, logger, data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

        # save pruning model
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_pruned_00.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)
        
        logger.info('pruned model complexity:')
        # cur_flops = get_model_complexity_info(model_without_ddp, logger)
        original_flops = pruning_settings.flops 
        logger.info('pruning percentage:')
        logger.info('attn percentage:{}'.format((original_flops[0]-cur_flops[0])/original_flops[0]*100))
        logger.info('ffn percentage:{}'.format((original_flops[1]-cur_flops[1])/original_flops[1]*100))
        logger.info('total percentage:{}'.format((original_flops[2]-cur_flops[2])/original_flops[2]*100))
        return

    logger.info(f"Start finetune for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    # finetune
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)
        if args.output_dir:
            if epoch % args.save_freq == 0 and epoch!=0:
                checkpoint_paths = [output_dir / 'checkpoint_{}.pth'.format(epoch)] 
            else:
                checkpoint_paths = [output_dir / 'checkpoint.pth'] 
            # only save pruned model
            if args.finetune_op==2:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                    }, checkpoint_path)    
                break
            # save the whole finetune state
            else:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)


    logger.info("Training completed. Starting post-training pruning...")
    #  Initialize the MHA FFN mask and load the existing model.
    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()
    start = time.time()
    # Search the optimal mask   # Collect gradients or other metrics to guide pruning   **
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        data_loader_val,
    )
    # The main purpose is to set an initial pruning constraint for the teacher model, which is based on a specific constraint objective and two iterative calculations.
    teacher_constraint = get_pruning_schedule(target=args.constraint, num_iter=2)[0]

    # Search for the optimal mask based on the target (e.g., MACs, Latency).
    if args.metric == "mac":
        teacher_head_mask, teacher_neuron_mask = search_optimal_mask(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            teacher_constraint,
        )
        head_mask, neuron_mask = search_optimal_mask(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            args.constraint,

        )
        pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
        logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    elif args.metric == "latency":
        mha_lut = torch.load(args.mha_lut)
        ffn_lut = torch.load(args.ffn_lut)
        teacher_head_mask, teacher_neuron_mask = search_optimal_latency(
            config,
            head_grads,
            neuron_grads,
            teacher_constraint,
            mha_lut,
            ffn_lut,
        )
        head_mask, neuron_mask = search_optimal_latency(
            config,
            head_grads,
            neuron_grads,
            args.constraint,
            mha_lut,
            ffn_lut,
        )
        pruned_latency = estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
        logger.info(f"Pruned Model Latency: {pruned_latency:.2f} ms")
    # Apply pruning mask to the model
    # apply_pruning_masks(model, head_mask, neuron_mask)
    # Rearrange the mask   **
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)
    print(" PSO-GSA algorithm runs successfully.")


    # The device where the first parameter of the model is located.
    model_device = next(model.parameters()).device
    # Devices that check head_mask and neuron_mask
    head_mask_device = head_mask.device
    neuron_mask_device = neuron_mask.device
    # Printing equipment information
    print(f"Model is on device: {model_device}")
    print(f"Head mask is on device: {head_mask_device}")
    print(f"Neuron mask is on device: {neuron_mask_device}")
    # Check if the equipment is consistent
    if model_device == head_mask_device and model_device == neuron_mask_device:
        print("Model and masks are on the same device.")  #
    else:
        print("Model and masks are NOT on the same device.")

    print("head_mask shape:", head_mask.shape)
    print("head_mask:", head_mask)
    print("neuron_mask shape:", neuron_mask.shape)
    print("neuron_mask:", neuron_mask)


# Quantification stage
    def str2model(name):
        # from FHPG-main.models.vit_quant import deit_base_patch16_224
        d = {
            # 'deit_tiny': deit_tiny_patch16_224,
            # 'deit_small': deit_small_patch16_224,
            'deit_base': quant_deit_base_patch16_224,
            # 'vit_base':  vit_base_patch16_224
            # 'vit_large': vit_large_patch16_224,
            # 'swin_tiny': swin_tiny_patch4_window7_224,
            # 'swin_small': swin_small_patch4_window7_224,
            # 'swin_base': swin_base_patch4_window7_224,
        }
        print('Model: %s' % d[name].__name__)
        return d[name]

    def apply_neuron_mask(model, neuron_mask):
        num_hidden_layers = neuron_mask.shape[0]
        # neuron_mask.requires_grad_(True)
        handles = []
        for layer_idx in range(num_hidden_layers):
            ffn2 = get_ffn2(model, layer_idx)
            handle = register_mask(ffn2, neuron_mask[layer_idx])
            handles.append(handle)
        return handles

    def apply_head_mask(model, head_mask):
        num_hidden_layers = head_mask.shape[0]
        num_heads = head_mask.shape[1]
        handles = []
        # Traverse each level
        for layer_idx in range(num_hidden_layers):
            # Get the MHA module of the current layer
            mha = get_mha(model, layer_idx)
            # Registration forward propagation hook
            handle = register_head_mask(mha, head_mask[layer_idx])
            handles.append(handle)
        return handles

    # Application mask
    # apply_masks_to_model(model, head_mask, neuron_mask)
    handles = apply_neuron_mask(model, neuron_mask)
    # apply_head_mask(model, head_mask)
    # Use the model for forward propagation
    for batch in data_loader_val:
        inputs, _ = batch
        inputs = inputs.to("cuda", non_blocking=True)
        outputs = model(inputs)


    # Get the model's state dictionary
    state_dict = model.state_dict()

    # Specify the save path
    save_path = '/FHPG/FHPG-main/output/pruned_model_weights-0717.pth'

    # Save the model weights in .pth format.
    torch.save(state_dict, save_path)
    # torch.save({'model': state_dict}, save_path)
    print("Save the pruned visual model file (.pth).")
    # 模型权重文件结构，显示所有可用的健
    checkpoint = torch.load(save_path, map_location='cpu')

    print("0717 Model File Structure : ", checkpoint.keys())
    # Clean up hooks to prevent excessive memory usage.
    for handle in handles:
        handle.remove()
    neuron_mask.requires_grad_(False)
    '''
    Note: Ensure that the model structure is exactly the same when loading weights as when saving;
    otherwise, shape mismatch errors may occur.
    '''

    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)
    model = str2model(args.model_name)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model_type = args.model_name.split('_')[0]
    if model_type == 'deit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError
    # train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    # val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    # # Data
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # val_dataset = datasets.ImageFolder(valdir, val_transform)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batchsize,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )
    # switch to evaluate mode
    model.eval()
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)
    torch.cuda.empty_cache()  # Release cached GPU memory
    # Add quantization parameters
    if args.quant:
        # train_dataset = datasets.ImageFolder(traindir, train_transform)
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=args.calib_batchsize,
        #     shuffle=True,
        #     num_workers=args.num_workers,
        #     pin_memory=True,
        #     drop_last=True,
        # )
        # Get calibration set.
        '''
        Calibration set preparation

        '''
        image_list = []
        for i, (data, target) in enumerate(data_loader_val):  # First, process the validation data from the pruning process.
            if i == args.calib_iter:
                break
            data = data.to(device)
            image_list.append(data)
        print('Calibrating...')
        model.model_open_calibrate()
        print("Complete data calibration.")
        with torch.no_grad():
            for i, image in enumerate(image_list):
                if i == len(image_list) - 1:
                    # This is used for OMSE method to
                    # calculate minimum quantization error
                    model.model_open_last_calibrate()
                output = model(image)
        model.model_close_calibrate()
        model.model_quant()
    print("After pruning, the model is converted into a quantized model.")

    print('Validating start... ')

    def validate(args, val_loader, model, criterion, device):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        val_start_time = end = time.time()
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1.data.item(), data.size(0))
            top5.update(prec5.data.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                ))
        val_end_time = time.time()
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
              format(top1=top1, top5=top5, time=val_end_time - val_start_time))
        return losses.avg, top1.avg, top5.avg

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    val_loss, val_prec1, val_prec5 = validate(args, data_loader_val, model, criterion, device)
    print('Validate finished. ')

    # Calculate and record the complexity of the pruned model.
    logger.info("Model complexity after pruning:")
    # Print the pruning time
    end = time.time()
    logger.info(f"{args.task_name} Pruning time (s): {end - start}")
    # cur_flops = get_model_complexity_info(model_without_ddp, logger)
    # logger.info(f"Attn percentage: {(original_flops[0] - cur_flops[0]) / original_flops[0] * 100}")
    # logger.info(f"FFN percentage: {(original_flops[1] - cur_flops[1]) / original_flops[1] * 100}")
    # logger.info(f"Total percentage: {(original_flops[2] - cur_flops[2]) / original_flops[2] * 100}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Post Training prune time {}'.format(total_time_str))
    # logger.info('pruned model complexity:')
    # cur_flops = get_model_complexity_info(model_without_ddp, logger)
    # if args.pruning:
    #     original_flops = pruning_settings.flops
    #     logger.info('pruning percentage:')
    #     logger.info('attn percentage:{}'.format((original_flops[0]-cur_flops[0])/original_flops[0]*100))
    #     logger.info('ffn percentage:{}'.format((original_flops[1]-cur_flops[1])/original_flops[1]*100))
    #     logger.info('total percentage:{}'.format((original_flops[2]-cur_flops[2])/original_flops[2]*100))
    # else:
    #     logger.info('prune model finish! ')
    #     sys.exit(0)
    logger.info("Post-training pruning and quant completed and pruning-model saved!!! ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.umask(0)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True, mode=0o777)
    # seq_len =
    IS_CIFAR = "CIFAR" in args.task_name

    main(args)
