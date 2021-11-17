import argparse
import os
import sys
import math
import json
import csv
import time
import random
import h5py
import socket
import dill
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

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
sys.path.append(os.path.join(curr_path, "..", "data"))

from model import MedicalBertConfig
from model import MedicalBertForSequenceClassification
from module import PolyWarmUpScheduler
from data import MedicalPretrainingDataset
from utils import is_main_process, get_world_size, get_rank, format_step, format_metric

from data.dataset import MedicalRecommendationDataset
from data import Vocab, PretrainVocab
from utils import sequence_metric, sequence_output_process
from utils import llprint, multi_label_metric, ddi_rate_score


TASKS = {
        "medication_recommendation": {
                                        "dataset": MedicalRecommendationDataset, 
                                        "dataset_args": {"data_prefix": "multi_visit"}
                                     },
        }

def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain data for the task.")
    parser.add_argument("--pretrained_model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The MEDICALBERT model config")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary file MedicalBert was pretrainined on")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=TASKS.keys(),
                        help="The name of the finetune task. Choices: " + " ".join(TASKS.keys()))

    parser.add_argument("--train_batch_size",
                        default=16,
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
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--json-summary', type=str, default="dllogger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument('--devices', type=int, default=[0], nargs="+",
                        help='device list')
    parser.add_argument('--distributed', action="store_true",
                        help='enable distributed multiple gpus')
    parser.add_argument('--log_freq',
                        type=int, default=1,
                        help='frequency of logging loss.')
    parser.add_argument('--display_freq',
                        type=int, default=100,
                        help='frequency of displaying logging.')
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to get model-task performance on the dev and test set by running eval.")
    parser.add_argument("--cpu",
                        action='store_true',
                        help="only use cpu, do not use gpu")

    return parser.parse_args()


def setup_training(args):

    if args.cpu:
        device = torch.device("cpu")
    else:
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
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.DEFAULT, 
                                                       step_format=format_step,
                                                       metric_format=format_metric)])
    else:
        dllogger.init(backends=[])

    print("device: {} n_gpu: {}, distributed training: {}".format(
        device, len(args.devices), bool(args.distributed)))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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

def get_dataset(args):

    if not args.task_name in TASKS.keys():
        raise ValueError("Undefined task {}".format(args.task_name))

    dataset = TASKS[args.task_name]["dataset"](data_path=args.data_dir, **TASKS[args.task_name]["dataset_args"])
    import ipdb
    ipdb.set_trace()

    train_loader, eval_loader, test_loader = dataset.get_dataloader(**TASKS[args.task_name]["dataloader_args"])

    return dataset, train_loader, eval_loader, test_loader


def main(args):

    # prepare summary writer for Tensorboard
    writer_path = "{}/../log".format(args.output_dir)
    os.makedirs(writer_path, exist_ok=True)
    writer = SummaryWriter(writer_path)
    
    # initial training settings
    device, args = setup_training(args)

    dllogger.log(step="PARAMETER", data={"Config": json.dumps(args.__dict__, indent=4)})

    # get vocab
    vocab = dill.load(open(args.vocab_file, "rb"))

    # get task data
    dataset, train_loader, eval_loader, test_loader = get_dataset(args)

    import ipdb
    ipdb.set_trace()

    model = MedicalBertForSequenceClassification.from_pretrained(args.pretrained_model_path)
    pass



def process_fn(rank, args):
    local_args = copy(args)
    local_args.local_rank = rank
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
 
