import torch
import argparse
import numpy as np
import dill
import time
import os
import sys
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
sys.path.append(os.path.join(curr_path, "..", "data"))

from models import GAMENet, Leap, DMNC, MLP, DualMLP, Transformer, DualTransformer, SafeDrugModel
from data.dataset import MedicalRecommendationDataset
from utils import sequence_metric, sequence_output_process
from utils import llprint, multi_label_metric, ddi_rate_score
from utils import buildMPNN

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
from data.build_vocab_and_records import Vocab

import ipdb

BASELINE_MODELS = ["GAMENet", "Leap", "Nearest", "MLP", "DualMLP", 
                   "Transformer", "DualTransformer", "SafeDrug", "DMNC"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=1109,
                        help="random seed")
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="log interval")
    # mode
    parser.add_argument('--train', action='store_true',
                        help="train mode on train set and eval on eval set")
    parser.add_argument('--eval', action='store_true',
                        help="eval mode on test set")
    parser.add_argument('--eval_on_train', action='store_true',
                        help="eval on train set")
    # save and load
    parser.add_argument('--load_path', type=str, default="",
                        help='load checkpoint')
    parser.add_argument("--save_path", type=str, default="/home/xyz/xyz/experiments/medical",
                        help="dir path to save the model")
    parser.add_argument("--data_dir", type=str, default="../data/processed_data",
                        help="dir path of processed data")
    parser.add_argument("--data_prefix", type=str, default="multi_visit",
                        help="prefix of processed data")
    parser.add_argument('--history', action='store_true',
                        help="load data of admission history")
    # model setting
    parser.add_argument('--model_name', type=str, default="GAMENet", 
                        choices=BASELINE_MODELS,
                        help="baseline model name, choices: {}".format(", ".join(BASELINE_MODELS)))
    parser.add_argument('--ddi', action='store_true', default=False, 
                        help="using ddi for GAMENet")
    parser.add_argument("--target_ddi", type=float, default=0.05,
                        help="target ddi for GAMENet")
    parser.add_argument('--kp', type=float, default=0.05, 
                        help='coefficient of P signal')
    parser.add_argument("--temperature_initial", type=float, default=0.5,
                        help="initial temperature for GAMENet")
    parser.add_argument("--temperature_decay", type=float, default=0.85,
                        help="temperature decay weight for GAMENet")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="embedding dimension")
    parser.add_argument("--seq_len", type=int, default=32,
                        help="sequence length")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="size of hidden state in MLP and Transformer")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="number of layers in MLP and Transformer")
    parser.add_argument("--alpha_bce", type=float, default=0.9,
                        help="multiply factor of bce loss")
    parser.add_argument("--alpha_margin", type=float, default=0.01,
                        help="multiply factor of margin loss")
    parser.add_argument("--head_size", type=int, default=16,
                        help="head size in Transformer")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of heads in Transformer")
    parser.add_argument("--filter_size", type=int, default=512,
                        help="FFN filter size in Transformer")
    # train setting
    parser.add_argument("--epochs", type=int, default=40,
                        help="training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="dropout rate")
    parser.add_argument('--shuffle', action='store_true',
                        help="shuffle data when training")

    return parser.parse_args()

def main(args):

    def evaluate(evalloader):
        # evaluate
        if non_trivial:
            model.eval()
        vocab_size = dataset.vocab_size
        y_targets = []
        y_preds = []
        y_pred_probs = []
        y_pred_labels = []
        jaccard, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        case_study = defaultdict(dict)
        med_count = 0
        visit_count = 0
        if args.model_name == "DMNC":
            i1_state, i2_state, i3_state = None, None, None
        # TODO: this is a temp solution, dataloader should be rewritten
        #evalloader = [data for data in evalloader if len(data) > 1]
        #for step, inputs in enumerate(evalloader):
        #    y_target = []
        #    y_pred = []
        #    y_pred_prob = []
        #    y_pred_label = []
        #    for adm_idx, adm in enumerate(inputs):

        #        if adm_idx == 0:
        #            continue
    
        #        y_target_tmp = np.zeros(vocab_size[2])
        #        y_target_tmp[adm[2]] = 1
        #        y_target.append(y_target_tmp)

        for step, data in tqdm(enumerate(evalloader), total=len(evalloader)):
    
            if args.model_name == "GAMENet":
                seq_inputs, y_target = data
                target_output1 = model(seq_inputs)
                target_output1 = torch.sigmoid(target_output1).detach().cpu().numpy()[0]
                y_pred_prob = target_output1
                y_pred_tmp = target_output1.copy()
                y_pred_tmp[y_pred_tmp>=0.5] = 1
                y_pred_tmp[y_pred_tmp<0.5] = 0
                y_pred = y_pred_tmp
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label = sorted(y_pred_label_tmp)
            elif args.model_name == "Leap":
                admission, y_target = data
                output_logits = model(admission)
                output_logits = output_logits.detach().cpu().numpy()
                out_list, sorted_predict = sequence_output_process(output_logits, [vocab_size[2], vocab_size[2] + 1])
                y_pred_label = sorted(sorted_predict)
                y_pred_prob = np.mean(output_logits[:, : -2], axis=0)
                y_pred_tmp = np.zeros(vocab_size[2])
                y_pred_tmp[out_list] = 1
                y_pred = y_pred_tmp
            elif args.model_name == "DMNC":
                admission, y_target = data
                output_logits, i1_state, i2_state, i3_state = model(admission, i1_state, i2_state, i3_state)
                output_logits = output_logits.detach().cpu().numpy()
                out_list, sorted_predict = sequence_output_process(output_logits, [vocab_size[2], vocab_size[2] + 1])
                y_pred_label = sorted(sorted_predict)
                y_pred_prob = np.mean(output_logits[:, : -2], axis=0)
                y_pred_tmp = np.zeros(vocab_size[2])
                y_pred_tmp[out_list] = 1
                y_pred = y_pred_tmp
                pass
            elif args.model_name == "SafeDrug":
                seq_inputs, y_target = data
                target_output1, _ = model(seq_inputs)
                target_output1 = torch.sigmoid(target_output1).detach().cpu().numpy()[0]
                y_pred_prob = target_output1
                y_pred_tmp = target_output1.copy()
                y_pred_tmp[y_pred_tmp>=0.5] = 1
                y_pred_tmp[y_pred_tmp<0.5] = 0
                y_pred = y_pred_tmp
                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label = sorted(y_pred_label_tmp)
            elif args.model_name == "Nearest":
                pred_list, y_target = data
                y_pred_label = sorted(pred_list)
                y_pred_tmp = np.zeros(vocab_size[2])
                y_pred_tmp[pred_list] = 1
                y_pred = y_pred_tmp
                y_pred_prob = y_pred_tmp
            elif args.model_name in ["MLP", "DualMLP", "Transformer", "DualTransformer"]:
                union_inputs, y_target = data
                target_output = model(union_inputs)
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
            #llprint('\rEval Epoch: {:4d} | Step: {:5d} / {:5d}'.format(epoch, step + 1, len(evalloader)))
        #print('')
    
        # ddi rate
        ddi_rate = ddi_rate_score(y_pred_labels,
                                  path=os.path.join(args.data_dir, "ddi_A_final.pkl"))
    
        if args.model_name == "Leap":
            (jaccard, 
             prauc, 
             avg_p, 
             avg_r, 
             avg_f1) = sequence_metric(np.array(y_targets), 
                                       np.array(y_preds), 
                                       np.array(y_pred_probs), 
                                       np.array(y_pred_labels, dtype=object))
        else:
            (jaccard, 
             prauc, 
             avg_p, 
             avg_r, 
             avg_f1) = multi_label_metric(np.array(y_targets), 
                                              np.array(y_preds), 
                                              np.array(y_pred_probs))
        llprint("DDI Rate: {:6.4f} | Jaccard: {:10.4f} | PRAUC: {:7.4f}\n"
                "AVG_PRC: {:7.4f} | AVG_RECALL: {:7.4f} | AVG_F1: {:6.4f}\n".format(
                ddi_rate, 
                jaccard, 
                prauc, 
                avg_p, 
                avg_r, 
                avg_f1
        ))
    
        #dill.dump(obj=y_pred_labels, file=open('gamenet_records.pkl', 'wb'))
        #dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))
    
        # print('avg med', med_count / visit_count)
    
        return ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1
    
    def train():
        model.train()
        loss_record = []
        if args.model_name == "GAMENet":
            prediction_loss_count = 0
            neg_loss_count = 0
        elif args.model_name == "DMNC":
            i1_state, i2_state, i3_state = None, None, None
        #for step, inputs in enumerate(data_train):
        #    for idx, adm in enumerate(inputs):
        #        # process data
        #        if args.model_name == "GAMENet":
        #            seq_inputs = inputs[:idx+1]
        #            loss1_target = np.zeros((1, vocab_size[2]))
        #            loss1_target[:, adm[2]] = 1
        #            loss3_target = np.full((1, vocab_size[2]), -1)
        #            for idx, item in enumerate(adm[2]):
        #                loss3_target[0][idx] = item
        #        elif args.model_name == "Leap":
        #            loss_target = adm[2] + [END_TOKEN]
        for step, data in enumerate(train_loader):
    
            # compute loss
            if args.model_name == "GAMENet":
                seq_inputs, loss1_target, loss3_target = data
                target_output1, batch_neg_loss = model(seq_inputs)
                loss1 = F.binary_cross_entropy_with_logits(target_output1, 
                                                           torch.FloatTensor(loss1_target).to(device))
                loss3 = F.multilabel_margin_loss(torch.sigmoid(target_output1), 
                                                 torch.LongTensor(loss3_target).to(device))
                if args.ddi:
                    target_output1 = torch.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]], 
                                                      path=os.path.join(args.data_dir, "ddi_A_final.pkl"))
                    if current_ddi_rate <= args.target_ddi:
                        loss = args.alpha_bce * loss1 + args.alpha_margin * loss3
                        prediction_loss_count += 1
                    else:
                        rnd = np.exp((args.target_ddi - current_ddi_rate)/T)
                        if np.random.rand(1) < rnd:
                            loss = batch_neg_loss
                            neg_loss_count += 1
                        else:
                            loss = args.alpha_bce * loss1 + args.alpha_margin * loss3
                            prediction_loss_count += 1
                else:
                    loss = args.alpha_bce * loss1 + args.alpha_margin * loss3
            elif args.model_name == "Leap":
                admission, loss_target = data
                output_logits = model(admission)
                loss = F.cross_entropy(output_logits, 
                                       torch.LongTensor(loss_target).to(device))
            elif args.model_name == "DMNC":
                admission, loss_target = data
                output_logits, i1_state, i2_state, i3_state = model(admission, i1_state, i2_state, i3_state)
                loss = F.cross_entropy(output_logits, 
                                       torch.LongTensor(loss_target).to(device))
                pass
            elif args.model_name == "SafeDrug":
                seq_inputs, loss1_target, loss3_target = data
                target_output1, loss_ddi = model(seq_inputs)
                loss1 = F.binary_cross_entropy_with_logits(target_output1, 
                                                           torch.FloatTensor(loss1_target).to(device))
                loss3 = F.multilabel_margin_loss(torch.sigmoid(target_output1), 
                                                 torch.LongTensor(loss3_target).to(device))

                target_output1 = torch.sigmoid(target_output1).detach().cpu().numpy()[0]
                target_output1[target_output1 >= 0.5] = 1
                target_output1[target_output1 < 0.5] = 0
                y_label = np.where(target_output1 == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], 
                                                  path=os.path.join(args.data_dir, "ddi_A_final.pkl"))
                
                if current_ddi_rate <= args.target_ddi:
                    loss = args.alpha_bce * loss1 + args.alpha_margin * loss3 + loss_ddi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = beta * (args.alpha_bce * loss1 + args.alpha_margin * loss3) + (1 - beta) * loss_ddi
            elif args.model_name in ["MLP", "DualMLP", "Transformer", "DualTransformer"]:
                inputs, bce_loss_target, margin_loss_target = data
                output_target = model(inputs)
                bce_loss = F.binary_cross_entropy_with_logits(output_target, 
                                                              torch.FloatTensor(bce_loss_target).to(device))
                margin_loss = F.multilabel_margin_loss(torch.sigmoid(output_target), 
                                                       torch.LongTensor(margin_loss_target).to(device))
                loss = args.alpha_bce * bce_loss + args.alpha_margin * margin_loss


            # optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    
            loss_record.append(loss.item())
    
            if step > 0 and step % args.log_interval == 0: 
                print("\rTrain Epoch: {:3d} | Step: {:5d} / {:5d} "
                      "| Loss: {:5.5f}".format(epoch, step, len(train_loader), np.mean(loss_record)))
            #llprint("\rTrain Epoch: {:3d} | Step: {:5d} / {:5d} "
            #        "| Loss: {:5.5f}".format(epoch, step + 1, len(train_loader), np.mean(loss_record)))
            #llprint('\rL_p count: %d, L_neg count: %d' % (prediction_loss_count, neg_loss_count))
        #print("")
        return np.mean(loss_record)


    # Set random seed
    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    non_trivial = True
    epoch = 0

    # Create save dir

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_name = "{}_{}".format(args.model_name, time_str)
    save_path = os.path.join(args.save_path, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load Data

    #data = dill.load(open(os.path.join(args.data_dir, "records_final.pkl"), "rb"))
    #vocab = dill.load(open(os.path.join(args.data_dir, "voc_final.pkl"), "rb"))
    #diag_vocab = vocab['diag_voc']
    #proc_vocab = vocab['pro_voc']
    #med_vocab = vocab['med_voc']
    #vocab_size = (len(diag_vocab.idx2word), len(proc_vocab.idx2word), len(med_vocab.idx2word))
    ## split data
    #split_point = int(len(data) * 2 / 3)
    #data_train = data[:split_point]
    #eval_len = int(len(data[split_point:]) / 2)
    #data_test = data[split_point:split_point + eval_len]
    #data_eval = data[split_point+eval_len:]
    ## special data for GAMENet
    #ehr_adj = dill.load(open(os.path.join(args.data_dir, "ehr_adj_final.pkl"), "rb"))
    #ddi_adj = dill.load(open(os.path.join(args.data_dir, "ddi_A_final.pkl"), "rb"))

    # load vocab
    vocab = dill.load(open(os.path.join(args.data_dir, "{}_vocab.pkl".format(args.data_prefix)), "rb"))

    dataset = MedicalRecommendationDataset(data_dir=args.data_dir, 
                                           data_file="{}_data.pkl".format(args.data_prefix),
                                           vocab=vocab
                                           )
    train_loader, eval_loader, test_loader = dataset.get_dataloader(args.model_name, 
                                                                    shuffle=args.shuffle, 
                                                                    history=args.history)
    if args.eval_on_train:
        train_eval_loader = dataset.get_train_eval_loader(args.model_name, history=args.history)

    if args.model_name == "GAMENet":
        ehr_adj, ddi_adj = dataset.get_extra_data(args.model_name)
    elif args.model_name == "SafeDrug":
        ddi_adj, ddi_mask_H, molecule = dataset.get_extra_data(args.model_name)

    # Create model

    if args.model_name == "GAMENet":
        model = GAMENet(dataset.vocab_size, ehr_adj, ddi_adj, 
                        emb_dim=args.emb_dim, 
                        ddi_in_memory=args.ddi, 
                        dropout=args.dropout,
                        device=device)
    elif args.model_name == "SafeDrug":
        MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, dataset.med_vocab.idx2word, 2, device)
        model = SafeDrugModel(dataset.vocab_size,
                              ddi_adj,
                              ddi_mask_H,
                              MPNNSet,
                              N_fingerprint,
                              average_projection,
                              emb_dim=args.emb_dim,
                              device=device)
    elif args.model_name == "DMNC":
        model = DMNC(dataset.vocab_size, 
                     emb_dim=args.emb_dim,
                     dropout=args.dropout,
                     device=device)
    elif args.model_name == "Leap":
        model = Leap(dataset.vocab_size, 
                     emb_dim=args.emb_dim, 
                     dropout=args.dropout,
                     device=device)
    elif args.model_name == "MLP":
        model = MLP(dataset.vocab_size,
                    emb_dim=args.emb_dim,
                    seq_len=args.seq_len,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    history=args.history,
                    device=device)
    elif args.model_name == "DualMLP":
        model = DualMLP(dataset.vocab_size,
                        emb_dim=args.emb_dim,
                        seq_len=args.seq_len,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        history=args.history,
                        device=device)
    elif args.model_name == "Transformer":
        model = Transformer(dataset.vocab_size,
                            hidden_size=args.hidden_size,
                            head_size=args.head_size,
                            num_heads=args.num_heads,
                            filter_size=args.filter_size,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            history=args.history,
                            device=device)
    elif args.model_name == "DualTransformer":
        model = DualTransformer(dataset.vocab_size,
                                hidden_size=args.hidden_size,
                                head_size=args.head_size,
                                num_heads=args.num_heads,
                                filter_size=args.filter_size,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                history=args.history,
                                device=device)
    elif args.model_name == "Nearest":
        non_trivial = False

    # Load checkpoint

    if non_trivial:
        if args.load_path:
            model.load_state_dict(torch.load(open(args.load_path, "rb")))

        model.to(device)

    # Train model

    if args.train and non_trivial:

        optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        history = defaultdict(list)
        best_epoch = 0
        best_jaccard = 0
        best_ckp = "final.model"
        if args.model_name == "GAMENet":
            T = args.temperature_initial

        print("=" * 25 + "    Training {}    ".format(args.model_name) + "=" * 25)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            mean_loss = train()

            # annealing for GAMENet
            if args.model_name == "GAMENet":
                T *= args.temperature_decay

            if args.eval_on_train:
                print("-" * 25 + "    Epoch %d Evaluating on Training Set    " % epoch + "-" * 25)
                evaluate(train_eval_loader)
            print("-" * 20 + "    Epoch {} Evaluating {} ".format(epoch, args.model_name) + 
                  time.strftime("%Y-%m-%d %H:%M:%S    ", time.localtime()) + "-" * 20)
            ddi_rate, jaccard, prauc, avg_p, avg_r, avg_f1 = evaluate(eval_loader)
            print("-" * (70 + len(args.model_name)))

            history['jaccard'].append(jaccard)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('Epoch: %d, Loss: %.4f, One Epoch Time: %.2fm, '
                    'Approximate Left Time: %dh%dm\n' % (epoch,
                                                         mean_loss,
                                                         elapsed_time,
                                                         int(elapsed_time * (args.epochs - epoch)) // 60,
                                                         int(elapsed_time * (args.epochs - epoch)) % 60))

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
        print("=" * 20 + "    Testing {} ".format(args.model_name) + 
              time.strftime("%Y-%m-%d %H:%M:%S    ", time.localtime()) +  "=" * 20)
        if args.train and non_trivial:
            model.load_state_dict(torch.load(open(os.path.join(save_path, best_ckp), "rb")))
        evaluate(test_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)


