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

def _max(a, b):
    if a > b:
        return a
    else:
        return b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_func', dest='loss_func', type=str, help='Loss function', default="ce")
    parser.add_argument('--this_turn', dest='this_turn', type=bool, help='If Use This Turn', default=False)
    parser.add_argument('--no_nonepolicy', dest='no_nonepolicy', type=bool, help='If include None policy', default=False)
    parser.add_argument('--gpu', dest='gpu', type=str, help="GPU ID", default="0")
    args = parser.parse_args()
    loss_func = args.loss_func
    this_turn = args.this_turn
    no_nonepolicy = args.no_nonepolicy
    gpu_id = args.gpu

    if no_nonepolicy:
        n_label = 10
        label_name=[
            'provide-org-facts',
            'PerStory',
            "task-related-inquiry",
            'provide-donation-procedure',
            'example-donation',
            "have-you-heard-of-the-org",
            "logical-appeal",
            "foot-in-the-door",
            'personal-related-inquiry',
            "emotion-appeal",
        ]
    else:
        n_label = 11
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

    epochs = 300

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("working on gpu")
    else:
        device = torch.device("cpu")
        print("working on cpu")

    data_path = "./data/"
    best_acc = 0
    best_macro_f1 = 0
    best_micro_f1 = 0
    best_all = 0

    bertcls = BERTCls(dropout=0.1, num_class=n_label, turn_emb_dim=12, this_turn=this_turn)
    bertcls.to(device=device)
    print('The BERTCls model has', bertcls.count_params(), 'trainable parameters.')

    train_iter, valid_iter, test_iter = load_data_bertcls(num_classes=n_label, f_path=data_path, this_turn=this_turn)
            
    bertcls.optimizer = optim.Adam(params=[{"params": bertcls.cls.parameters()}, {"params": bertcls.turn_embeddings.parameters()}], lr=1e-3)
    bertcls.lr_scheduler = optim.lr_scheduler.StepLR(bertcls.optimizer, step_size=100, gamma=0.95)
    if loss_func == "ce" or loss_func == "CE":
        bertcls.loss_func = nn.CrossEntropyLoss()
    elif loss_func == "focal" or loss_func == "Focal":
        alpha = torch.zeros(n_label, 1)
        for l in train_iter: 
            if this_turn:
                (xs,x_len),(hs,hs_len),turn,ys,index,(tt,tt_len)=l
            else:
                (xs,x_len),(hs,hs_len),turn,ys,index=l
            for y in ys:
                alpha[int(y)][0] += 1
        alpha = 1000 / alpha
        print(alpha)
        bertcls.loss_func = FocalLoss(class_num=n_label, alpha=alpha)

    for epoch in range(epochs):
        print("---------- Epoch {} ----------".format(epoch+1))
        for phase in ('train', 'val'):
            accs=AverageMeter()
            losses= AverageMeter()
            recalls=AverageMeter()
            macro_f1s=AverageMeter()
            micro_f1s=AverageMeter()
            if phase == 'train':
                bertcls.train(True)
                phrase_iter=train_iter

            else:
                bertcls.eval()
                # print("running valid.....")
                phrase_iter=valid_iter
            end = time.time()
            for l in phrase_iter: 
                if this_turn:
                    (xs,x_len),(hs,hs_len),turn,ys,index,(tt,tt_len)=l
                else:
                    (xs,x_len),(hs,hs_len),turn,ys,index=l

                # xs=xs.to(device)
                # hs=hs.to(device)
                # ys=ys.to(device)
                # turn=turn.to(device)

                # x_len=x_len.cuda().float().view(-1,1)
                # hs_len=hs_len.cuda().float().view(-1,1)

                bertcls.optimizer.zero_grad() #clear the gradient 
                if this_turn:
                    logits = bertcls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device), this_turn_ids=tt.to(device))
                else:
                    logits = bertcls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device))

                loss = bertcls.loss_func(logits, ys.to(device).data.long())
                acc,macro_f1,micro_f1 = eval_(logits, labels=ys.to(device).data.long())

                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(bertcls.parameters(), 10)
                    bertcls.optimizer.step()
                    bertcls.lr_scheduler.step()

                nsample = xs.size(0)
                accs.update(acc, nsample)
                macro_f1s.update(macro_f1, nsample)
                micro_f1s.update(micro_f1, nsample)
                losses.update(loss.item(), nsample)

            elapsed_time = time.time() - end

            print('[{}]\tEpoch: {}/{}\tAcc: {:.2%}\tMacro_F1: {:.2%}\tMicro_F1: {:.2%}\tLoss: {:.3f}\tTime: {:.3f}'.format(
            phase, epoch+1, epochs, accs.avg, macro_f1s.avg, micro_f1s.avg, losses.avg, elapsed_time))

            if phase == 'val' and (macro_f1s.avg > best_macro_f1 or micro_f1s.avg > best_micro_f1 or accs.avg > best_acc):
                best_macro_f1 = _max(macro_f1s.avg, best_macro_f1)
                best_micro_f1 = _max(micro_f1s.avg, best_micro_f1)
                best_acc = _max(accs.avg, best_acc)
                best_epoch = epoch
                if macro_f1s.avg + micro_f1s.avg + accs.avg > best_all:
                    best_model_state = bertcls.state_dict()
                    best_all = macro_f1s.avg + micro_f1s.avg + accs.avg
                preds=None
                targets=None
                
                test_accs=AverageMeter()
                test_macro_f1s=AverageMeter()
                test_micro_f1s=AverageMeter()
                y_true=None
                y_pred=None

                for l in test_iter:
                    bertcls.eval()
                    if this_turn:
                        (xs,x_len),(hs,hs_len),turn,ys,index,(tt,tt_len)=l
                    else:
                        (xs,x_len),(hs,hs_len),turn,ys,index=l
                    # xs=xs.to(device)
                    # hs=hs.to(device)
                    # ys=ys.to(device)
                    # turn=turn.to(device)
                    # x_len=x_len.cuda().float().view(-1,1)
                    # hs_len=hs_len.cuda().float().view(-1,1)
                    bertcls.optimizer.zero_grad() #clear the gradient 

                    if this_turn:
                        logits = bertcls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device), this_turn_ids=tt.to(device))
                    else:
                        logits = bertcls(x_ids=xs.to(device), turn=turn.to(device), his_ids=hs.to(device))

                    output=logits
                    l_n=logits.data.cpu().numpy()
                    nsample = xs.size(0)

                    acc,macro_f1,micro_f1=eval_(output, labels=ys.to(device).data.long())
                    _, predicted = torch.max(logits.cpu().data, 1)
                    test_accs.update(acc, nsample)
                    
                    test_macro_f1s.update(macro_f1, nsample)
                    test_micro_f1s.update(micro_f1, nsample)
                    if y_true is None:
                        y_true=ys.to(device).data.cpu().numpy()
                        y_pred=l_n.argmax(axis=1)
                    else:
                        y_true=np.hstack([y_true,ys.to(device).data.cpu().numpy()])
                        y_pred=np.hstack([y_pred,l_n.argmax(axis=1)])

                print('[test]\tEpoch: {}/{}\tAcc: {:.2%}\tMacro_F1: {:.2%}\tMicro_F1: {:.2%}\tTime: {:.3f}'.format(
                    epoch+1, epochs, test_accs.avg, test_macro_f1s.avg, test_micro_f1s.avg, elapsed_time))
                from sklearn.metrics import confusion_matrix
                cm=confusion_matrix(y_true, y_pred)
                # print(cm.shape)
                print_cm(cm, label_name)

    print('[Info] best valid acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
    torch.save(best_model_state, './models/policy_classifier_{}.pkl'.format(loss_func))
    print('Test Acc: {:.2%}\tMacro F1: {:.2%}\tMicro F1: {:.2%}'.format(
                    test_accs.avg, test_macro_f1s.avg, test_micro_f1s.avg))

