from torchtext import data

from torch import nn
from torch.autograd import Variable
from torch import optim
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from model import *
from tqdm import tqdm
from util import *
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import copy
import os
import sys
import argparse
plt.switch_backend('agg')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def multi_label_metrics_transfer(y, label_num):
    """
    Multi-label classify. 0 is placeholder.
    tensor([[1, 1, 1, 2, 1, 1, 1, 1, 1],
            [2, 3, 2, 1, 0, 2, 2, 0, 2],
            [0, 0, 3, 0, 0, 0, 0, 0, 0]])
    output one-hot multi-label matrix
    tensor([[1., 1., 0.],
            [1., 0., 1.],
            [1., 1., 1.],
            [1., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.]])
    """
    return torch.zeros(
        y.shape[1],
        label_num,
        dtype=torch.float,
        device=y.device,
    ).scatter_(1, y.T, 
        torch.ones(
            y.shape[1],
            label_num,
            dtype=torch.float,
            device=y.device,
        ),
    )[:, 1:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_func', dest='loss_func', type=str, help='Loss function', default="bce")
    parser.add_argument('--gpu', dest='gpu', type=str, help="GPU ID", default="0")
    args = parser.parse_args()
    loss_func = args.loss_func
    gpu_id = args.gpu

    n_label = 11
    run_times = 1    
    epochs = 2000

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("working on gpu")
    else:
        device = torch.device("cpu")
        print("working on cpu")
    
    # label_name=['Have','example',"logical",'PerStory','Foot',"Credibility","P-inqu","Emotional",'none',"t-inqu","info"]
    label_name=[
        'provide-org-facts',
        'PerStory',
        "task-related-inquiry",
        'provide-donation-procedure',
        'None',
        "have-you-heard-of-the-org",
        "logical-appeal",
        "foot-in-the-door",
        'personal-related-inquiry',
        "emotion-appeal",
        "example-donation",
    ]
    
    data_path = "./data/"
    
    acc_dict = {}
    prec_dict = {}

    feat_list = {'add_char':False,'add_turn':True,'add_senti':False,'add_his':True}
    feat = 'all'
    his_mode = 'rnn'

    acc_dict[feat] = []
    prec_dict[feat] = []

    bertmulticls = BERTMultiCls(dropout=0.1, turn_emb_dim=12, num_class=11)
    bertmulticls.to(device=device)
    print('The BERTCls model has', bertmulticls.count_params(), 'trainable parameters.')

    train_iter, valid_iter, test_iter = load_data_bertmulticls(num_classes=n_label, f_path=data_path)
            
    bertmulticls.optimizer = optim.Adam(params=[{"params": bertmulticls.cls.parameters()}, {"params": bertmulticls.turn_embeddings.parameters()}], lr=1e-3)
    bertmulticls.lr_scheduler = optim.lr_scheduler.StepLR(bertmulticls.optimizer, step_size=100, gamma=0.95)
    bertmulticls.loss_func = nn.BCEWithLogitsLoss(size_average=False)
    best_acc = 0

    for epoch in range(epochs):
        print("---------- Epoch {} ----------".format(epoch+1))
        for phase in ('train', 'val'):
            accs = AverageMeter()
            acc_ss = AverageMeter()
            acc_ms = AverageMeter()
            losses = AverageMeter()
            recalls = AverageMeter()
            # f1s = AverageMeter()
            if phase == 'train':
                bertmulticls.train(True)
                phrase_iter=train_iter
            else:
                bertmulticls.eval()
                # print("running valid.....")
                phrase_iter=valid_iter
            end = time.time()
            for l in phrase_iter: 
                (xs,x_len),(hs,hs_len),turn,ys,index=l

                # xs=xs.to(device)
                # hs=hs.to(device)
                # ys=ys.to(device)
                # turn=turn.to(device)

                # x_len=x_len.cuda().float().view(-1,1)
                # hs_len=hs_len.cuda().float().view(-1,1)

                bertmulticls.optimizer.zero_grad() #clear the gradient 
                logits = bertmulticls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device))
                gold_label_metric = multi_label_metrics_transfer(ys, n_label + 1).to(device) # <pad>

                loss = bertmulticls.loss_func(logits[:, 1:], gold_label_metric)
                probs = torch.sigmoid(logits)
                pred_metric = torch.where(probs > 0.5, 
                                          torch.ones(probs.shape[0], probs.shape[1], device=probs.device), 
                                          torch.zeros(probs.shape[0], probs.shape[1], device=probs.device)
                                          )[:, 1:]
                
                y_p = pred_metric.tolist()
                y_t = gold_label_metric.tolist()
                n_single = 0
                n_multi = 0
                correct_single = 0
                correct_multi = 0
                acc_single = 0.0
                acc_multi = 0.0
                for i, j in zip(y_p, y_t):
                    if (sum(j) == 1):
                        n_single += 1
                        if (i == j):
                            correct_single += 1
                    else:
                        n_multi += 1
                        # if (i == j):
                        #     correct_multi += 1
                        if sum([i[k] == 1 and i[k] == j[k] for k in range(len(i))]) >= 1:
                            correct_multi += 1

                nsample = n_single + n_multi

                acc = (correct_single + correct_multi) / (n_single + n_multi)
                accs.update(acc, nsample)
                if n_single != 0:
                    acc_single = correct_single / n_single
                    acc_ss.update(acc_single, n_single)
                if n_multi != 0:
                    acc_multi = correct_multi / n_multi
                    acc_ms.update(acc_multi, n_multi)
                # f1s.update(f1, nsample)


                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(bertmulticls.parameters(), 10)
                    bertmulticls.optimizer.step()
                    bertmulticls.lr_scheduler.step()
                
                losses.update(loss.item(), nsample)

            elapsed_time = time.time() - end

            print('[{}]\tEpoch: {}/{}\tAcc: {:.2%}\tAcc-s: {:.2%}\tAcc-m:{:.2%}\tLoss: {:.3f}\tTime: {:.3f}'.format(
            phase, epoch+1, epochs, accs.avg, acc_ss.avg, acc_ms.avg, losses.avg, elapsed_time))

            if phase == 'val' and accs.avg > best_acc:
                best_acc = accs.avg
                best_epoch = epoch
                best_model_state = bertmulticls.state_dict()
                preds=None
                targets=None
                
                test_accs = AverageMeter()
                test_acc_ss = AverageMeter()
                test_acc_ms = AverageMeter()
                # test_f1s = AverageMeter()

                for l in test_iter:
                    bertmulticls.eval()
                    (xs,x_len),(hs,hs_len),turn,ys,index=l
                    # xs=xs.to(device)
                    # hs=hs.to(device)
                    # ys=ys.to(device)
                    # turn=turn.to(device)
                    # x_len=x_len.cuda().float().view(-1,1)
                    # hs_len=hs_len.cuda().float().view(-1,1)
                    bertmulticls.optimizer.zero_grad() #clear the gradient 

                    logits = bertmulticls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device))
                    gold_label_metric = multi_label_metrics_transfer(ys, n_label + 1).to(device) # <pad>
                    probs = torch.sigmoid(logits)
                    pred_metric = torch.where(probs > 0.5, 
                                            torch.ones(probs.shape[0], probs.shape[1], device=probs.device), 
                                            torch.zeros(probs.shape[0], probs.shape[1], device=probs.device)
                                            )[:, 1:]
                    
                    y_p = pred_metric.tolist()
                    y_t = gold_label_metric.tolist()
                    print("Pred: ", y_p)
                    print("Truth:", y_t)
                    n_single = 0
                    n_multi = 0
                    correct_single = 0
                    correct_multi = 0
                    for i, j in zip(y_p, y_t):
                        if (sum(j) == 1):
                            n_single += 1
                            if (i == j):
                                correct_single += 1
                        else:
                            n_multi += 1
                            # if (i == j):
                            #     correct_multi += 1
                            if sum([i[k] == 1 and i[k] == j[k] for k in range(len(i))]) >= 1:
                                correct_multi += 1

                    nsample = n_single + n_multi

                    acc = (correct_single + correct_multi) / (n_single + n_multi)
                    test_accs.update(acc, nsample)
                    if n_single != 0:
                        acc_single = correct_single / n_single
                        test_acc_ss.update(acc_single, n_single)
                    if n_multi != 0:
                        acc_multi = correct_multi / n_multi
                        test_acc_ms.update(acc_multi, n_multi)

                print('[test]\tEpoch: {}/{}\tAcc: {:.2%}\tAcc-s: {:.2%}\tAcc-m: {:.2%}\tTime: {:.3f}'.format(
                    epoch+1, epochs, test_accs.avg, test_acc_ss.avg, test_acc_ms.avg, elapsed_time))

    print('[Info] best valid acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
    torch.save(best_model_state, './models/policy_classifier_{}.pkl'.format(loss_func))
    print('Test Acc: {:.2%}\tAcc-s: {:.2%}\tAcc-m: {:.2%}\t'.format(
                    test_accs.avg, test_acc_ss.avg, test_acc_ms.avg))
    acc_dict[feat].append(test_accs.avg)

    res_dict={}
    res_dict['acc']=acc_dict
    with open('final.json','w') as f:
        json.dump(res_dict,f)
