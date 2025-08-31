import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.train_lanenet = args.train_lanenet
    cfg.lanenet_ckp = args.lanenet_ckp

    # train or test
    if cfg.train_lanenet: 
        cfg.run = 'train_lanenet'
    elif args.validate: 
        cfg.run = 'val'
    elif args.demo:
        cfg.run = 'demo'
    elif args.test:
        cfg.run = 'test'
    else:
        cfg.run = 'train_CLRlaneNet'

    if not (args.validate or args.test or args.demo) and not cfg.train_lanenet:
        assert cfg.lanenet_ckp is not None, 'Using lanenet as label assignment but a ckeckpoint for lanenet was not given'

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    cudnn.benchmark = True

    runner = Runner(cfg)

    if args.validate:
        runner.validate()
    elif args.test:
        runner.test()
    elif args.demo:
        runner.demo()
    else:
        runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test', 
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='whether to test the checkpoint on demo set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--train_lanenet',
            action='store_true',
            help='whether to train lanenet (true) or to use lanenet as the label assignment method in CLRlaneNet train')
    parser.add_argument('--lanenet_ckp',
            default=None,
            help='The lanenet checkpoint file to load from and use lanenet as the label assignment method')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
