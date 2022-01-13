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
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
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
                                        "dataset_args": {
                                                            "data_file": "multi_visit_data.pkl",
                                                        },
                                        "dataloader_args": {
                                                            "model_name": "MedicalBert",
                                                            "shuffle": False,
                                                            "history": False,
                                                        },
                                        "task_label": "MED-ATC",
                                     },
        "medication_recommendation_safedrug": {
                                        "dataset": MedicalRecommendationDataset, 
                                        "dataset_args": {
                                                            "data_file": "safedrug_data.pkl",
                                                        },
                                        "dataloader_args": {
                                                            "model_name": "MedicalBert",
                                                            "shuffle": False,
                                                            "history": False,
                                                        },
                                        "task_label": "MED-ATC",
                                     },
        "medication_recommendation_safedrug_lab": {
                                        "dataset": MedicalRecommendationDataset, 
                                        "dataset_args": {
                                                            "data_file": "safedrug_lab_data.pkl",
                                                        },
                                        "dataloader_args": {
                                                            "model_name": "MedicalBert",
                                                            "shuffle": False,
                                                            "history": False,
                                                        },
                                        "task_label": "MED-ATC",
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
    parser.add_argument("--pretrained_model_ckpt",
                        default=None,
                        type=str,
                        help="The MEDICALBERT model ckpt if given, do not use the final model")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary file MedicalBert was pretrained on")
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
    parser.add_argument("--weight_decay", 
                        default=0.0,
                        type=float, 
                        help="weight decay")
    parser.add_argument("--max_epochs",
                        default=5,
                        type=int,
                        help="Total epoch of training steps to perform.")
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
    parser.add_argument("--train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--eval",
                        action='store_true',
                        help="Whether to get model-task performance on the dev and test set by running eval.")
    parser.add_argument('--eval_on_train', 
                        action='store_true',
                        help="eval on train set")
    parser.add_argument('--eval_on_diseases', 
                        action='store_true',
                        help="eval on diseases")
    parser.add_argument('--eval_disease_threshold', type=int, default=100,
                        help='retain diseases occurence > threshold in test set')
    parser.add_argument("--cpu",
                        action='store_true',
                        help="only use cpu, do not use gpu")

    parser.add_argument("--alpha_bce", 
                        type=float, default=0.9,
                        help="multiply factor of bce loss")
    parser.add_argument("--alpha_margin", 
                        type=float, default=0.01,
                        help="multiply factor of margin loss")
    parser.add_argument('--history', 
                        action='store_true',
                        help="load data of admission history")
    parser.add_argument('--shuffle', 
                        action='store_true',
                        help="shuffle data")
    parser.add_argument("--attention_probs_dropout_prob", 
                        type=float, default=0.1,
                        help="dropout rate on attention probabilities")
    parser.add_argument("--hidden_dropout_prob", 
                        type=float, default=0.1,
                        help="dropout rate on hidden layer")
    parser.add_argument('--fix_model', 
                        action='store_true',
                        help="fix model parameter")
    parser.add_argument('--from_scratch', 
                        action='store_true',
                        help="do not load pretrained parameter")
    parser.add_argument('--mean_repr', 
                        action='store_true',
                        help="use mean representation instead of <CLS> to predict")
    parser.add_argument('--use_bert_embedding', 
                        action='store_true',
                        help="use bert embedding to predict first")

    parser.add_argument('--decoder', type=str, default="linear",
                        choices=["linear", "mlp", "gamenet"],
                        help='decoder of Finetuning model. default: linear')
    parser.add_argument('--decoder_mlp_layers', type=int, default=2,
                        help='number of layers of MLP in decoder')
    parser.add_argument('--decoder_mlp_hidden', type=int, default=1024,
                        help='dim of hidden layers of MLP in decoder')
    parser.add_argument('--decoder_gamenet_hidden', type=int, default=256,
                        help='dim of hidden layers of GAMENet in decoder')

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

    if not args.train and not args.eval and not args.eval_on_diseases:
        raise ValueError("At least one of `train` or `eval` or 'eval_on_diseases' must be True.")

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

def get_dataset(args, vocab):

    if not args.task_name in TASKS.keys():
        raise ValueError("Undefined task {}".format(args.task_name))

    dataset = TASKS[args.task_name]["dataset"](data_dir=args.data_dir, 
                                               vocab=vocab,
                                               **TASKS[args.task_name]["dataset_args"])


    TASKS[args.task_name]["dataloader_args"]["history"] = args.history
    TASKS[args.task_name]["dataloader_args"]["shuffle"] = args.shuffle
    if args.decoder == "gamenet":
        TASKS[args.task_name]["dataloader_args"]["return_history_meds"] = True
    train_loader, eval_loader, test_loader = dataset.get_dataloader(**TASKS[args.task_name]["dataloader_args"])
    if args.eval_on_train:
        train_eval_loader = dataset.get_train_eval_loader(**TASKS[args.task_name]["dataloader_args"])
    else:
        train_eval_loader = None

    return dataset, train_loader, eval_loader, test_loader, train_eval_loader


def main(args):

    def evaluate(evalloader):
        
        model.eval()
        y_targets = []
        y_preds = []
        y_pred_probs = []
        y_pred_labels = []
        jaccard, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_count = 0
        visit_count = 0

        for step, data in tqdm(enumerate(evalloader), total=len(evalloader)):

            # get data
            input_ids, segment_ids, y_target = data
            if args.decoder == "gamenet":
                input_ids, history_meds = input_ids
            if len(input_ids) > config.max_position_embeddings:
                input_ids = input_ids[-config.max_position_embeddings:]
                segment_ids = segment_ids[-config.max_position_embeddings:]
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(device)
            input_mask = torch.ones_like(input_ids).to(device)

            # predict
            if args.decoder == "gamenet":
                input_ids = (input_ids, history_meds)
            target_output = model(input_ids=input_ids, 
                                  token_type_ids=segment_ids,
                                  attention_mask=input_mask)
            target_output = torch.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob = target_output
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred = y_pred_tmp
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label = sorted(y_pred_label_tmp)

            med_count += len(y_pred_label)
            visit_count += 1
    
            y_targets.append(y_target)
            y_preds.append(y_pred)
            y_pred_probs.append(y_pred_prob)
            y_pred_labels.append(y_pred_label)

        # ddi rate
        ddi_rate = ddi_rate_score(y_pred_labels,
                                  path=os.path.join(args.data_dir, "ddi_A_final.pkl"))
        (jaccard, 
         prauc, 
         avg_p, 
         avg_r, 
         avg_f1) = multi_label_metric(np.array(y_targets), 
                                      np.array(y_preds), 
                                      np.array(y_pred_probs))

        print("DDI Rate: {:6.4f} | Jaccard: {:10.4f} | PRAUC: {:7.4f}\n"
              "AVG_PRC: {:7.4f} | AVG_RECALL: {:7.4f} | AVG_F1: {:6.4f}\n".format(
              ddi_rate, 
              jaccard, 
              prauc, 
              avg_p, 
              avg_r, 
              avg_f1
        ))

        model.train()
        return ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1
 
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
    dataset, train_loader, eval_loader, test_loader, train_eval_loader = get_dataset(args, vocab)

    # get num_labels
    num_labels = vocab.get_type_vocab_size(TASKS[args.task_name]["task_label"])

    # get model config
    config = MedicalBertConfig.from_json_file(os.path.join(args.pretrained_model_path, "model_config.json"))
    #config.with_pooler = True
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    config.hidden_dropout_prob = args.hidden_dropout_prob
    dllogger.log(step="PARAMETER", data={"Model Config": config.to_json_string()})

    # prepare pretrained model
    if args.pretrained_model_path and args.pretrained_model_ckpt:
        ckpt_file = os.path.join(args.pretrained_model_path, args.pretrained_model_ckpt)
        state_dict = torch.load(ckpt_file, map_location="cpu")["model"]
    else:
        state_dict = None

    if args.use_bert_embedding:
        embedding_index = vocab.get_type_vocab_ids("MED")
    else:
        embedding_index = None

    decoder_params = {}
    if args.decoder.lower() == "linear":
        pass
    elif args.decoder.lower() == "mlp":
        decoder_params = {
                "num_layers": args.decoder_mlp_layers,
                "hidden_dim": args.decoder_mlp_hidden,
                "dropout": args.hidden_dropout_prob
                         }
    elif args.decoder.lower() == "gamenet":

        ehr_adj, ddi_adj = dataset.get_extra_data("GAMENet")
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        decoder_params = {
                "vocab": vocab,
                "ehr_adj": ehr_adj,
                "ddi_adj": ddi_adj,
                "hidden_dim": args.decoder_gamenet_hidden,
                "dropout": args.hidden_dropout_prob,
                "device": device
                         }
        pass

    if not args.from_scratch:
        model = MedicalBertForSequenceClassification.from_pretrained(args.pretrained_model_path, 
                                                                     config=config,
                                                                     state_dict=state_dict,
                                                                     num_labels=num_labels,
                                                                     mean_repr=args.mean_repr,
                                                                     embedding_index=embedding_index,
                                                                     decoder=args.decoder.lower(),
                                                                     decoder_params=decoder_params
                                                                     )
    else:
        model = MedicalBertForSequenceClassification(config=config, 
                                                     num_labels=num_labels, 
                                                     mean_repr=args.mean_repr,
                                                     decoder=args.decoder.lower(),
                                                     decoder_params=decoder_params)

    # load checkpoint if needed
    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
        start_step = 0
        curr_epoch = 1
        epoch = curr_epoch
    else:
        if args.resume_step == -1:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step

        checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), 
                                map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        curr_epoch = checkpoint["epoch"]
        start_step = checkpoint["step"] + 1
        epoch = curr_epoch
        
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)

    # build optimizer and scheduler
    named_params = list(model.named_parameters())
    if args.fix_model:
        param_to_optimize = []
        for name, param in named_params:
            if "classifier" in name:
                param_to_optimize.append((name, param))
    else:
        param_to_optimize = named_params
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_to_optimize if not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_to_optimize if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }]

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                       warmup=args.warmup_proportion, 
                                       total_steps=args.max_epochs * len(train_loader))

    if args.resume_from_checkpoint:
        lr_scheduler.step(global_step)
        optimizer.load_state_dict(checkpoint["optimizer"])

    # convert model for DDP training
    if args.distributed:
        model = DDP(model, device_ids=[device], dim=0)
    elif len(args.devices) > 1:
        model = torch.nn.DataParallel(model)


    if args.train:

        if is_main_process():
            print("=" * 25 + "    Finetuning {}    ".format(args.task_name) + "=" * 25)
            dllogger.log(step="PARAMETER", data={"SEED": args.seed})
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER", data={"batch_size_per_gpu": args.train_batch_size})
            dllogger.log(step="PARAMETER", data={"learning_rate": args.learning_rate})

        # prepare for training
        model.train()
        if checkpoint is None:
            best_step = 0
            best_jaccard = 0
            best_ckp = "final.model"
        else:
            best_step = checkpoint["best"]["best_step"]
            best_jaccard = checkpoint["best"]["best_jaccard"]
            best_ckp = checkpoint["best"]["best_ckp"]
        training_steps = 0

        for epoch in range(curr_epoch, args.max_epochs + 1):
            start_time = time.time()

            model.train()
            loss_record = []
            for step, data in enumerate(train_loader):

                if step < start_step:
                    continue
                else:
                    start_step = 0
    
                # get data
                input_ids, segment_ids, bce_loss_target, margin_loss_target = data
                if args.decoder == "gamenet":
                    input_ids, history_meds = input_ids
                if len(input_ids) > config.max_position_embeddings:
                    input_ids = input_ids[-config.max_position_embeddings:]
                    segment_ids = segment_ids[-config.max_position_embeddings:]
                input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
                segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(device)
                input_mask = torch.ones_like(input_ids).to(device)

                # compute loss
                if args.decoder == "gamenet":
                    output_target, ddi_loss = model(input_ids=(input_ids, history_meds),
                                                    token_type_ids=segment_ids,
                                                    attention_mask=input_mask)
                else:
                    output_target = model(input_ids=input_ids,
                                          token_type_ids=segment_ids,
                                          attention_mask=input_mask)
                bce_loss = F.binary_cross_entropy_with_logits(output_target, 
                                                              torch.FloatTensor(bce_loss_target).to(device))
                margin_loss = F.multilabel_margin_loss(torch.sigmoid(output_target), 
                                                       torch.LongTensor(margin_loss_target).to(device))
                loss = args.alpha_bce * bce_loss + args.alpha_margin * margin_loss

                if args.decoder == "gamenet":
                    loss += ddi_loss

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                loss_record.append(loss.item())
                training_steps += 1

                # optimize the model
                if training_steps % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    optimizer.step()
                    global_step += 1
                    model.zero_grad()
    
                # logging loss
                if training_steps % (args.gradient_accumulation_steps * args.log_freq) == 0:
                    if is_main_process():
                        verbosity = dllogger.Verbosity.VERBOSE
                        dllogger.log(step=(epoch, global_step, ), 
                                     data={"average_loss": np.mean(loss_record),
                                           "step_loss": loss.item() * args.gradient_accumulation_steps,
                                           "lr": optimizer.param_groups[0]['lr']},
                                     verbosity=verbosity)
                        if global_step % args.display_freq == 0:
                            print("\rTrain Epoch: {:3d} | Global Step: {:5d} | Step: {:5d} / {:5d} "
                                    "| LR: {:5.3e} | Loss: {:5.5f}".format(epoch,
                                                           global_step,
                                                           step + 1, 
                                                           len(train_loader),
                                                           optimizer.param_groups[0]['lr'],
                                                           np.mean(loss_record)))

                # save model per args.num_steps_per_checkpoint
                if global_step > 0 and global_step % args.num_steps_per_checkpoint == 0:
                    if is_main_process() and training_steps % args.gradient_accumulation_steps == 0:
                        dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                        # eval on checkpoint
                        if args.eval_on_train:
                            print("-" * 25 + "    Epoch %d Evaluating on Training Set    " % epoch + "-" * 25)
                            evaluate(train_eval_loader)
                        print("-" * 20 + "    Epoch {} Evaluating {} ".format(epoch, args.task_name) + 
                              time.strftime("%Y-%m-%d %H:%M:%S    ", time.localtime()) + "-" * 20)
                        ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1 = evaluate(eval_loader)
                        print("-" * (70 + len(args.task_name)))
                        dllogger.log(step=(epoch, global_step, ), 
                                     data={"jaccard": jaccard,
                                           "ddi_rate": ddi_rate,
                                           "avg_precision": avg_p,
                                           "avg_recall": avg_r,
                                           "avg_f1": avg_f1,
                                           "prauc": prauc},
                                     verbosity=dllogger.Verbosity.VERBOSE)

                        # save checkpoint
                        model_to_save = model.module if hasattr(model, 'module') else model
                        ckp_name = "ckpt_{}.pt".format(global_step)
                        output_save_file = os.path.join(args.output_dir, ckp_name)
                        torch.save({'model': model_to_save.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'epoch': epoch,
                                    'step': step,
                                    'best': {"best_step": best_step, 
                                             "best_jaccard": best_jaccard, 
                                             "best_ckp": best_ckp},
                                    "jaccard": jaccard,
                                    "ddi_rate": ddi_rate}, 
                                    output_save_file)

                        if best_jaccard < jaccard:
                            best_step = global_step
                            best_jaccard = jaccard
                            best_ckp = ckp_name

            mean_loss = np.mean(loss_record)





            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print('Epoch: %d, Loss: %.4f, One Epoch Time: %.2fm, '
                  'Approximate Left Time: %dh%dm\n' % (epoch,
                                                       mean_loss,
                                                       elapsed_time,
                                                       int(elapsed_time * (args.max_epochs - epoch)) // 60,
                                                       int(elapsed_time * (args.max_epochs - epoch)) % 60))


        # save final model
        torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'final.model'), 'wb'))

        print('best_step:', best_step)

    # Evaluation

    if args.eval:
        print("=" * 20 + "    Testing {} ".format(args.task_name) + 
              time.strftime("%Y-%m-%d %H:%M:%S    ", time.localtime()) +  "=" * 20)
        if args.train:
            model.load_state_dict(torch.load(open(os.path.join(args.output_dir, best_ckp), "rb"))["model"])
        ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1 = evaluate(test_loader)

        dllogger.log(step="PARAMETER", data={"test_step": global_step})
        dllogger.log(step=(epoch, global_step, ), 
                     data={"jaccard": jaccard,
                           "ddi_rate": ddi_rate,
                           "avg_precision": avg_p,
                           "avg_recall": avg_r,
                           "avg_f1": avg_f1,
                           "prauc": prauc},
                     verbosity=dllogger.Verbosity.VERBOSE)

    if args.eval_on_diseases:
        test_disease_adms = dataset.data_test["ICD9_CODE"].tolist()
        test_disease_dict = {}
        for adm in test_disease_adms:
            for d in adm:
                if d in test_disease_dict.keys():
                    test_disease_dict[d] += 1
                else:
                    test_disease_dict[d] = 1
        test_diseases = sorted([(k, v) for k, v in test_disease_dict.items() if v > args.eval_disease_threshold],
                                key=lambda x: -x[1])
        disease_dict = {vocab.get_word_id(d, "DIAG"): (vocab.idx2word[vocab.get_word_id(d, "DIAG")],
                                                       vocab.idx2detail[vocab.get_word_id(d, "DIAG")])
                        for d, _ in test_diseases}
        for diag_id in disease_dict.keys():
            TASKS[args.task_name]["dataloader_args"]["diag_id"] = diag_id
            diag_test_loader = dataset.get_diag_test_loader(**TASKS[args.task_name]["dataloader_args"])

            ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1 = evaluate(diag_test_loader)

            dllogger.log(step="PARAMETER", data={"disease_name": disease_dict[diag_id][1]})
            dllogger.log(step=(epoch, global_step, disease_dict[diag_id]), 
                         data={"jaccard": jaccard,
                               "ddi_rate": ddi_rate,
                               "avg_precision": avg_p,
                               "avg_recall": avg_r,
                               "avg_f1": avg_f1,
                               "prauc": prauc},
                         verbosity=dllogger.Verbosity.VERBOSE)



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
 
