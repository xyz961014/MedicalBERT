import argparse
import os
import sys
import math
import csv
import time
import random
import h5py
import socket
from tqdm import tqdm
from copy import copy
import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter


import dllogger

sys.path.append("..")

from model import MedicalBertForPreTraining, MedicalBertConfig
from model import MedicalBertPretrainingCriterion
from module import PolyWarmUpScheduler
from data import MedicalPretrainingDataset
from utils import is_main_process, get_world_size, get_rank, format_step

def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The MEDICALBERT model config")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    #parser.add_argument("--max_seq_length",
    #                    default=512,
    #                    type=int,
    #                    help="The maximum total input sequence length \n"
    #                         "Sequences longer than this will be truncated, \n"
    #                         "and sequences shorter than this will be padded.")
    #parser.add_argument("--max_predictions_per_seq",
    #                    default=80,
    #                    type=int,
    #                    help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte "
                             "before performing a backward/update pass.")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until "
                             "a model checkpoint is saved to disk.")
    parser.add_argument('--json-summary', type=str, default="dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument('--devices', type=int, default=[0], nargs="+",
                        help='device list')
    parser.add_argument('--distributed', action="store_true",
                        help='enable distributed multiple gpus')

    return parser.parse_args()

def setup_training(args):

    devices = [torch.device("cuda:" + str(i)) for i in args.devices]
    device = devices[args.local_rank]
    torch.cuda.set_device(device)

    if args.distributed:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(devices))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=os.path.join(args.output_dir, args.json_summary)),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    print("device: {} n_gpu: {}, distributed training: {}".format(
        device, len(args.devices), bool(args.distributed)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def main(args):

    assert torch.cuda.is_available()

    # prepare summary writer for Tensorboard
    writer_path = "{}/../log".format(args.output_dir)
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)
    
    # initial training settings
    device, args = setup_training(args)

    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # get model config
    config = MedicalBertConfig.from_json_file(args.config_file)

    # build model
    model = MedicalBertForPreTraining(config)

    # load checkpoint if needed
    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), 
                                    map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")

        model.load_state_dict(checkpoint['model'], strict=False)
        
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)

    # build optimizer and scheduler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                       warmup=args.warmup_proportion, 
                                       total_steps=args.max_steps)

    # load saved optimizer if needed
    if args.resume_from_checkpoint:
        if args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

    # convert model for DDP training
    if args.distributed:
        model = DDP(model, device_ids=[device], dim=0)
    elif len(args.devices) > 1:
        model = torch.nn.DataParallel(model)

    # build training criterion
    criterion = MedicalBertPretrainingCriterion(config.vocab_size)

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})
        dllogger.log(step="PARAMETER", data={"train_start": True})
        dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
        dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

    # prepare for training
    model.train()
    most_recent_ckpts_paths = []
    average_loss = 0.0  # averaged loss every args.log_freq steps
    epoch = 0
    training_steps = 0

    # get data
    restored_dataloader = None
    if checkpoint is not None:
        restored_dataloader = checkpoint.get("dataloader", None)

    if restored_dataloader is None:
        train_dataset = MedicalPretrainingDataset(args.input_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, 
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size * len(args.devices),
                                      num_workers=4, 
                                      worker_init_fn=worker_init,
                                      pin_memory=True)
    else:
        train_dataloader = restored_dataloader

    train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
    
    while True:
        for step, batch in train_iter:
            import ipdb
            ipdb.set_trace()
    pass


def process_fn(rank, args):
    local_args = copy(args)
    local_args.rank = rank
    main(local_args)


if __name__ == "__main__":

    args = parse_args()
    world_size = len(args.devices)

    if args.distributed:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            args.url = url

        mp.spawn(process_fn, args=(args, ), nprocs=world_size)
    else:
        process_fn(0, args)
        


