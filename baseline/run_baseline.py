import torch
import argparse
import numpy as np
import dill
import time
import os
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from models import GAMENet, Leap
from utils import sequence_metric, sequence_output_process
from utils import llprint, multi_label_metric, ddi_rate_score
import ipdb

BASELINE_MODELS = ["GAMENet", "Leap"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default="1109",
                        help="random seed")
    # mode
    parser.add_argument('--train', action='store_true',
                        help="train mode on train set and eval on eval set")
    parser.add_argument('--eval', action='store_true',
                        help="eval mode on test set")
    # save and load path
    parser.add_argument('--load_path', type=str, default="",
                        help='load checkpoint')
    parser.add_argument("--save_path", type=str, default="/home/xyz/xyz/experiments/medical",
                        help="dir path to save the model")
    parser.add_argument("--data_path", type=str, default="/home/xyz/xyz/Projects/GAMENet/data",
                        help="dir path of processed data")
    # model setting
    parser.add_argument('--model_name', type=str, default="GAMENet", 
                        choices=BASELINE_MODELS,
                        help="baseline model name, choices: {}".format(", ".join(BASELINE_MODELS)))
    parser.add_argument('--ddi', action='store_true', default=False, 
                        help="using ddi for GAMENet")
    parser.add_argument("--target_ddi", type=float, default=0.05,
                        help="target ddi for GAMENet")
    parser.add_argument("--temperature_initial", type=float, default=0.5,
                        help="initial temperature for GAMENet")
    parser.add_argument("--temperature_decay", type=float, default=0.85,
                        help="temperature decay weight for GAMENet")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="embedding dimension")
    # train setting
    parser.add_argument("--epochs", type=int, default=40,
                        help="training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="learning rate")

    return parser.parse_args()

def main(args):

    def evaluate():
        # evaluate
        model.eval()
        smm_record = []
        jaccard, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        case_study = defaultdict(dict)
        med_count = 0
        visit_count = 0
        for step, input in enumerate(data_eval):
            y_gt = []
            y_pred = []
            y_pred_prob = []
            y_pred_label = []
            for adm_idx, adm in enumerate(input):
    
                y_gt_tmp = np.zeros(vocab_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)
    
                if args.model_name == "GAMENet":
                    target_output1 = model(input[:adm_idx+1])
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    y_pred_prob.append(target_output1)
                    y_pred_tmp = target_output1.copy()
                    y_pred_tmp[y_pred_tmp>=0.5] = 1
                    y_pred_tmp[y_pred_tmp<0.5] = 0
                    y_pred.append(y_pred_tmp)
                    y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                    y_pred_label.append(sorted(y_pred_label_tmp))
                    med_count += len(y_pred_label_tmp)
                elif args.model_name == "Leap":
                    output_logits = model(adm)
                    output_logits = output_logits.detach().cpu().numpy()
                    out_list, sorted_predict = sequence_output_process(output_logits, [vocab_size[2], vocab_size[2] + 1])
                    y_pred_label.append(sorted(sorted_predict))
                    y_pred_prob.append(np.mean(output_logits[:, : -2], axis=0))
                    y_pred_tmp = np.zeros(vocab_size[2])
                    y_pred_tmp[out_list] = 1
                    y_pred.append(y_pred_tmp)
                    med_count += len(sorted_predict)

                visit_count += 1
    
    
            smm_record.append(y_pred_label)
            if args.model_name == "Leap":
                (adm_jaccard, 
                 adm_prauc, 
                 adm_avg_p, 
                 adm_avg_r, 
                 adm_avg_f1) = sequence_metric(np.array(y_gt), 
                                               np.array(y_pred), 
                                               np.array(y_pred_prob), 
                                               np.array(y_pred_label))
            else:
                (adm_jaccard, 
                 adm_prauc, 
                 adm_avg_p, 
                 adm_avg_r, 
                 adm_avg_f1) = multi_label_metric(np.array(y_gt), 
                                                  np.array(y_pred), 
                                                  np.array(y_pred_prob))
            #case_study[adm_jaccard] = {'jaccard': adm_jaccard, 'patient': input, 'y_label': y_pred_label}
    
            jaccard.append(adm_jaccard)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            llprint('\rEval Epoch: {:4d} | Step: {:5d} / {:5d}'.format(epoch, step, len(data_eval)))
        print('')
    
        # ddi rate
        ddi_rate = ddi_rate_score(smm_record,
                                  path=os.path.join(args.data_path, "ddi_A_final.pkl"))
    
        llprint("DDI Rate: {:6.4f} | Jaccard: {:10.4f} | PRAUC: {:7.4f}\n"
                "AVG_PRC: {:7.4f} | AVG_RECALL: {:7.4f} | AVG_F1: {:6.4f}\n".format(
                    ddi_rate, 
                    np.mean(jaccard), 
                    np.mean(prauc), 
                    np.mean(avg_p), 
                    np.mean(avg_r), 
                    np.mean(avg_f1)
        ))
        #dill.dump(obj=smm_record, file=open('gamenet_records.pkl', 'wb'))
        #dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))
    
        # print('avg med', med_count / visit_count)
    
        return ddi_rate, np.mean(jaccard), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    
    def train():
        model.train()
        loss_record = []
        if args.model_name == "GAMENet":
            prediction_loss_count = 0
            neg_loss_count = 0
        for step, inputs in enumerate(data_train):
            for idx, adm in enumerate(inputs):
                # process data
                if args.model_name == "GAMENet":
                    seq_inputs = inputs[:idx+1]
                    loss1_target = np.zeros((1, vocab_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, vocab_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item
                elif args.model_name == "Leap":
                    loss_target = adm[2] + [END_TOKEN]
    
                # compute loss
                if args.model_name == "GAMENet":
                    target_output1, batch_neg_loss = model(seq_inputs)
                    loss1 = F.binary_cross_entropy_with_logits(target_output1, 
                                                               torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), 
                                                     torch.LongTensor(loss3_target).to(device))
                    if args.ddi:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]], 
                                                          path=os.path.join(args.data_path, "ddi_A_final.pkl"))
                        if current_ddi_rate <= args.target_ddi:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_count += 1
                        else:
                            rnd = np.exp((args.target_ddi - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_count += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_count += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3
                elif args.model_name == "Leap":
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, 
                                           torch.LongTensor(loss_target).to(device))

                # optimize
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
    
                loss_record.append(loss.item())
    
            
            llprint("\rTrain Epoch: {:3d} | Step: {:5d} / {:5d} "
                    "| Loss: {:5.5f}".format(epoch, step, len(data_train), np.mean(loss_record)))
            #llprint('\rL_p count: %d, L_neg count: %d' % (prediction_loss_count, neg_loss_count))
        print("")
        return np.mean(loss_record)


    # Set random seed

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Create save dir

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_name = "{}_{}".format(args.model_name, time_str)
    save_path = os.path.join(args.save_path, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load Data

    data = dill.load(open(os.path.join(args.data_path, "records_final.pkl"), "rb"))
    vocab = dill.load(open(os.path.join(args.data_path, "voc_final.pkl"), "rb"))
    diag_vocab = vocab['diag_voc']
    proc_vocab = vocab['pro_voc']
    med_vocab = vocab['med_voc']
    vocab_size = (len(diag_vocab.idx2word), len(proc_vocab.idx2word), len(med_vocab.idx2word))
    # split data
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    # special data for GAMENet
    ehr_adj = dill.load(open(os.path.join(args.data_path, "ehr_adj_final.pkl"), "rb"))
    ddi_adj = dill.load(open(os.path.join(args.data_path, "ddi_A_final.pkl"), "rb"))

    # Create model

    if args.model_name == "GAMENet":
        model = GAMENet(vocab_size, ehr_adj, ddi_adj, emb_dim=args.emb_dim, ddi_in_memory=args.ddi, device=device)
    elif args.model_name == "Leap":
        model = Leap(vocab_size, emb_dim=args.emb_dim, device=device)

    # Load checkpoint

    if args.load_path:
        model.load_state_dict(torch.load(open(args.load_path, "rb")))

    model.to(device)

    # Train model

    if args.train:

        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        
        history = defaultdict(list)
        best_epoch = 0
        best_jaccard = 0
        best_ckp = "final.model"
        if args.model_name == "GAMENet":
            T = args.temperature_initial
        elif args.model_name == "Leap":
            END_TOKEN = vocab_size[2] + 1

        print("=" * 25 + "    Training {}    ".format(args.model_name) + "=" * 25)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            mean_loss = train()

            # annealing for GAMENet
            if args.model_name == "GAMENet":
                T *= args.temperature_decay

            print("-" * 25 + "    Evaluating {}    ".format(args.model_name) + "-" * 25)
            ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1 = evaluate()
            print("-" * (68 + len(args.model_name)))

            history['jaccard'].append(jaccard)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('Epoch: %d, Loss: %.4f, One Epoch Time: %.2fm, '
                    'Appro Left Time: %.2fm\n' % (epoch,
                                                  mean_loss,
                                                  elapsed_time,
                                                  elapsed_time * (args.epochs - epoch - 1)))

            ckp_name = 'Epoch_%d_JACCARD_%.4f_DDI_%.4f.model' % (epoch, jaccard, ddi_rate)
            torch.save(model.state_dict(), open(os.path.join(save_path, ckp_name), 'wb'))
            print('')
            if epoch != 0 and best_jaccard < jaccard:
                best_epoch = epoch
                best_jaccard = jaccard
                best_ckp = ckp_name


        dill.dump(history, open(os.path.join(save_path, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(os.path.join(save_path, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)

    # Evaluation

    if args.eval:
        print("=" * 25 + "    Evaluating {}    ".format(args.model_name) + "=" * 25)
        model.load_state_dict(torch.load(open(os.path.join(save_path, best_ckp), "rb")))
        evaluate()


if __name__ == "__main__":
    args = parse_args()
    main(args)


