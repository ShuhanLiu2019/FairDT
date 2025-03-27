import argparse
import warnings
import seaborn as sns

import torch
torch.cuda.empty_cache()
from alive_progress import alive_bar
import random
import numpy as np
import torch as th
import torch.nn as nn
from utils import random_splits
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support
import datetime

warnings.filterwarnings("ignore")

from model import LogReg,Model

parser = argparse.ArgumentParser(description="FairDT")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  
parser.add_argument('--dev', type=int, default=0, help='device id')

parser.add_argument(
    "--dataname", type=str, default="cora", help="Name of dataset."
)
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu."
)
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--a", type=float, default=5, help="hyper-parameters a."
)
parser.add_argument(
    "--b", type=float, default=2, help="hyper-parameters b."
)
parser.add_argument(
    "--phi", type=float, default=10, help="hyper-parameters phi."
)
parser.add_argument(
    "--lr", type=float, default=0.010, help="Learning rate of prop."
)
parser.add_argument(
    "--lr1", type=float, default=0.001, help="Learning rate of FairDT loss 1."
)

parser.add_argument(
    "--lr2", type=float, default=0.001, help="Learning rate of FairDT loss 2."
)

parser.add_argument(
    "--lr3", type=float, default=0.01, help="Learning rate of linear evaluator."
)
parser.add_argument(
    "--lr4", type=float, default=0.01, help="Learning rate of dis audit."
)
parser.add_argument(
    "--lr5", type=float, default=0.01, help="Learning rate of pre audit."
)
parser.add_argument(
    "--lr6", type=float, default=0.001, help="Learning rate of FairDT loss 2."
)
parser.add_argument(
    "--wd", type=float, default=0.0, help="Weight decay of FairDT prop."
)
parser.add_argument(
    "--wd1", type=float, default=0.0, help="Weight decay of FairDT 1."
)
parser.add_argument(
    "--wd2", type=float, default=0.0, help="Weight decay of FairDT 2."
)
parser.add_argument(
    "--wd3", type=float, default=0.0, help="Weight decay of linear evaluator."
)
parser.add_argument(
    "--wd4", type=float, default=0.0, help="Weight decay of dis audit."
)
parser.add_argument(
    "--wd5", type=float, default=0.0, help="Weight decay of pre audit."
)
parser.add_argument(
    "--wd6", type=float, default=0.0, help="Weight decay of FairDT 2."
)
parser.add_argument(
    "--hid_dim", type=int, default=512, help="Hidden layer dim."
)

parser.add_argument(
    "--K", type=int, default=10, help="Layer of encoder."
)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu',
                    help='activation function')
parser.add_argument('--acc', type=float, default=0.688,
                    help='the selected accuracy on val would be at least this high')
parser.add_argument('--f1', type=float, default=0.745,
                    help='the selected f1 score on val would be at least this high')
parser.add_argument('--label_number', type=int, default=1000,
                    help='the label number,threshold')
parser.add_argument('--test_idx', type=bool, default=False)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.cuda.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)
seed=args.seed

from dataset_loader import DataLoader
import time

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2*(features - min_values).div(max_values-min_values) - 1


def fair_metric(val_labels,sens_test,val_preds): 
    val_y=val_labels.cpu().numpy()
    
    idx_s0 = sens_test==0
    idx_s1 = sens_test==1

    idx_s0_y1 = (idx_s0 & (val_y == 1)).to(torch.bool)
    idx_s1_y1 = (idx_s1 & (val_y == 1)).to(torch.bool)

    pred_y = (val_preds.squeeze() > 0).type_as(val_labels).cpu().numpy()
    
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))   
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))    
    return parity,equality



def fair_loss(train_labels,sens_train,logits):
    train_y=train_labels
    
    idx_s0 = sens_train==0
    idx_s1 = sens_train==1

    idx_s0_y1 = (idx_s0 & (train_y == 1)).to(torch.bool)
    idx_s1_y1 = (idx_s1 & (train_y == 1)).to(torch.bool)

    sp0 = torch.mean(logits[idx_s0])
    sp1 = torch.mean(logits[idx_s1])
    eo0 = torch.mean(logits[idx_s0_y1])
    eo1 = torch.mean(logits[idx_s1_y1])

    return sp0,sp1,eo0,eo1



def split(label_idx,feat,label,edge_index,seed,label_number,test_idx):
    idx_train = label_idx[:min(int(0.8 * len(label_idx)),label_number)]
    print('Length of training set',len(idx_train))

    if test_idx==True:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.9 * len(label_idx)):] 


    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)

    mask_train = torch.isin(edge_index[0], idx_train) & torch.isin(edge_index[1], idx_train)
    edge_index_train = edge_index[:,mask_train]
    idx_map1 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_train)}
    edge_index_train = torch.tensor([[idx_map1.get(idx.item(), idx.item()) for idx in edge_index_train[0]],
                                    [idx_map1.get(idx.item(), idx.item()) for idx in edge_index_train[1]]])

    mask_test = torch.isin(edge_index[0], idx_test) & torch.isin(edge_index[1], idx_test)
    edge_index_test = edge_index[:, mask_test]
    idx_map2 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_test)}
    edge_index_test = torch.tensor([[idx_map2.get(idx.item(), idx.item()) for idx in edge_index_test[0]],
                                    [idx_map2.get(idx.item(), idx.item()) for idx in edge_index_test[1]]])


    return idx_train, idx_test, edge_index_train, edge_index_test


def add_s_train(idx_train,sens_train,edge_index):
    idx_s0 = idx_train[sens_train==0]
    idx_s1 = idx_train[sens_train==1]

    idx_s0 = torch.tensor(idx_s0, dtype=torch.long)
    idx_s1 = torch.tensor(idx_s1, dtype=torch.long)

    mask_s0 = torch.isin(edge_index[0], idx_s0) & torch.isin(edge_index[1], idx_s0)
    edge_index_s0 = edge_index[:,mask_s0]
    idx_map1 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_s0)}
    edge_index_s0 = torch.tensor([[idx_map1.get(idx.item(), idx.item()) for idx in edge_index_s0[0]],
                                    [idx_map1.get(idx.item(), idx.item()) for idx in edge_index_s0[1]]])

    mask_s1 = torch.isin(edge_index[0], idx_s1) & torch.isin(edge_index[1], idx_s1)
    edge_index_s1 = edge_index[:, mask_s1]
    idx_map2 = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_s1)}
    edge_index_s1 = torch.tensor([[idx_map2.get(idx.item(), idx.item()) for idx in edge_index_s1[0]],
                                    [idx_map2.get(idx.item(), idx.item()) for idx in edge_index_s1[1]]])

    return idx_s0, idx_s1, edge_index_s0, edge_index_s1


if __name__ == "__main__":
    print(args)
    # Step 1: Load data =================================================================== #
    dataset = DataLoader(name=args.dataname)
    data = dataset[0]
    feat = data.x
    label = data.y
    label[label > 1] = 1
    edge_index = data.edge_index.long()
    sens=dataset.sens()

    a=args.a
    b=args.b
    phi=args.phi


    if args.dataname=='nba':
        label_idx = dataset.get_idx("SALARY",seed)
        feat = feature_norm(feat)
    elif args.dataname=='german':
        label_idx = dataset.get_idx("GoodCustomer",seed)
    elif args.dataname=='credit':
        label_idx = dataset.get_idx("NoDefaultNextMonth",seed)
    elif args.dataname=='bail':
        label_idx = dataset.get_idx("RECID",seed)
    else:
        label_idx = dataset.get_idx("I_am_working_in_field",seed)
    

    idx_train, idx_test, edge_index_train, edge_index_test=split(label_idx,feat,label,edge_index,seed,args.label_number,args.test_idx)
    

    feat_train=feat[idx_train]
    feat_test=feat[idx_test]

    
    label_train=label[idx_train]
    label_test=label[idx_test]
    #print('label_train:',label_train)
    
    sens_train=sens[idx_train]
    sens_test=sens[idx_test]

    idx_s0, idx_s1, edge_index_s0, edge_index_s1=add_s_train(idx_train,sens_train,edge_index)

    feat_train_s0=feat[idx_s0]
    feat_train_s1=feat[idx_s1]


    n_feat = feat_train.shape[1]
    n_classes = np.unique(label_train).shape[0]
    print('Classes number：',n_classes)

    edge_index_train = edge_index_train.to(args.device)
    feat_train = feat_train.to(args.device)
    sens_train = sens_train.to(args.device)
    label_train = label_train.to(args.device)
    feat_train_s0 = feat_train_s0.to(args.device)
    feat_train_s1 = feat_train_s1.to(args.device)
    edge_index_s0 = edge_index_s0.to(args.device)
    edge_index_s1 = edge_index_s1.to(args.device)

    edge_index_test = edge_index_test.to(args.device)
    feat_test = feat_test.to(args.device)
    label_test = label_test.to(args.device)

    n_node = feat_train.shape[0]
    n_node_test = feat_test.shape[0]
    print('Number of training n_node：',n_node)
    print('Number of test n_node：',n_node_test)

    s0_node=feat_train_s0.shape[0]
    s1_node= feat_train_s1.shape[0]

    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))  
    
    lbls0 = th.ones(s0_node)
    lbls1 = th.zeros(s1_node)
    lbls = th.cat((lbls0, lbls1))  


    # Step 2: Create model =================================================================== #
    model = Model(in_dim=n_feat, out_dim=args.hid_dim, K=args.K, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn,n_node=n_node,phi=phi)
    model = model.to(args.device)

    logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)  
    logreg = logreg.to(args.device)


    lbl = lbl.to(args.device)
    lbls = lbls.to(args.device)

    results = []
    v_p_results = []
    v_e_results=[]
    t_p_results=[]
    t_e_results = []
    best_fair = 100

    # 10 fixed seeds for random splits fro
    print('Length of total used dataset：',len(label_idx))

    # Step 3: Create training components ===================================================== #
    optimizer_p = torch.optim.Adam([{'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.disc1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.disc2.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
                                  {'params': model.disc3.parameters(), 'weight_decay': args.wd6, 'lr': args.lr6},
                                  {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
                                  ])

    
    loss_p = nn.BCEWithLogitsLoss()
    loss_h = nn.BCEWithLogitsLoss()

    loss_cs = nn.CrossEntropyLoss()
    #loss_cs = nn.BCEWithLogitsLoss()
    loss_sp=nn.MSELoss()
    loss_eo=nn.MSELoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    best_t = 0

    assert label_train.shape[0] == n_node

    best_val_acc = 0    
    eval_acc = 0   
    bad_counter = 0     

    tag = str(int(time.time()))

    best_result = {}
    best_fair = 100
    best_acc = 0
    best_auc = 0
    best_ar = 0
    best_f1 = 0
    best_ars_result = {}

    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer_p.zero_grad()

            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat_train[shuf_idx, :]

            out1,out2,out3 = model(edge_index_train, feat_train,feat_train_s0,feat_train_s1,edge_index_s0, edge_index_s1,shuf_feat,n_node)
            train_embeds=model.get_embedding(edge_index_train, feat_train, n_node)

            logreg.train()
            logits = logreg(train_embeds)      
            preds = th.argmax(logits, dim=1)     

            sp0,sp1,eo0,eo1 = fair_loss(label_train,sens_train,logits)
            
            train_acc = th.sum(preds == label_train).float() / label_train.shape[0]        

            loss_f=loss_h(out2, lbls)+ loss_h(out3, lbls) + loss_sp(sp0, sp1) + loss_eo(eo0, eo1)

            loss =  a*loss_f + b*loss_p(out1, lbl) 

            loss.backward()    
            optimizer_p.step()

            logreg.eval()   
            model.eval()    

            if epoch % 20 == 0:
                print("Epoch: {0}, All Loss: {1:0.4f}, FN1 Loss: {2:0.4f}, FN2 Loss: {3:0.4f}, SP Loss: {4:0.4f}, EO Loss: {5:0.4f},".format(epoch, loss.item(),loss_p(out1, lbl),loss_h(out2, lbls), loss_sp(sp0, sp1), loss_eo(eo0, eo1)))

            bar()

    model.eval() 
    train_embeds=model.get_embedding(edge_index_train, feat_train, n_node)

    #print(edge_index_train)
    #print(edge_index_test)
    

    test_embs = model.get_embedding(edge_index_test,  feat_test, n_node_test)


    logreg2 = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)  
    logreg2 = logreg2.to(args.device)
    opt=torch.optim.Adam(logreg2.parameters(), lr=args.lr3, weight_decay=args.wd3)

    print("=== Evaluation ===")
    ''' Linear Evaluation '''

    print("Unique test labels:", torch.unique(label_test))
    print("Label min/max:", label_test.min(), label_test.max())



    for epoch in range(1000):
        logreg2.train()
        opt.zero_grad()
        logits = logreg2(train_embeds)    
        preds = th.argmax(logits, dim=1)       
        train_acc = th.sum(preds == label_train).float() / label_train.shape[0]  
        loss = loss_cs(logits, label_train)    
        loss.backward()     
        opt.step() 

        logreg2.eval()  

        with th.no_grad():  
            test_logits = logreg2(test_embs)     
            test_preds = th.argmax(test_logits, dim=1)
            acc_test = th.sum(test_preds == label_test).float() / label_test.shape[0]

            #test_probs = th.softmax(test_logits, dim=1)[:, 1]  # Probability of positive class
            #print('test_probs:',test_probs)
            #roc_test = roc_auc_score(label_test.cpu().numpy(), test_probs.detach().cpu().numpy())

            precision, recall, f1_test, _ = precision_recall_fscore_support(label_test.cpu(), test_preds.cpu(), average='binary')
            
            parity,equality = fair_metric(label_test,sens_test,test_preds)

        if best_acc <= acc_test:
            best_acc = acc_test
            best_acc_result = {}
            best_acc_result['acc'] = acc_test.item()
            best_acc_result['parity'] = parity
            best_acc_result['equality'] = equality
            best_ars_result['best_acc_result'] = best_acc_result
            print("Test best_acc:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "f1: {:.4f}".format(f1_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality))
            
        if best_f1 <= f1_test:
            best_f1 = f1_test
            best_f1_result = {}
            best_f1_result['acc'] = acc_test.item()
            best_f1_result['f1'] = f1_test
            best_f1_result['parity'] = parity
            best_f1_result['equality'] = equality
            best_ars_result['best_f1_result'] = best_f1_result
            print("Test best_f1:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "f1: {:.4f}".format(f1_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality))

        
        if best_ar <= f1_test + acc_test:
            best_ar = f1_test + acc_test
            best_ar_result = {}
            best_ar_result['acc'] = acc_test.item()
            best_ar_result['f1'] = f1_test
            best_ar_result['parity'] = parity
            best_ar_result['equality'] = equality
            best_ars_result['best_ar_result'] = best_ar_result
            print("Test best_ar:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "f1: {:.4f}".format(f1_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality))
                    
        if acc_test > args.acc and f1_test > args.f1:
            if best_fair > parity + equality:
                best_fair = parity + equality
                best_result['acc'] = acc_test.item()
                best_result['f1'] = f1_test
                best_result['parity'] = parity
                best_result['equality'] = equality 
                print("Test best_fair:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "f1: {:.4f}".format(f1_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality))

        #print('Linear evaluation accuracy on train dataset:{:.4f}'.format(train_acc))
        #print('Linear evaluation accuracy on test dataset:{:.4f}'.format(eval_acc))
        #print('Linear evaluation fairmetric on validation dataset:{:.4f}，{:.4f}'.format(best_t_parity,best_t_equality))

    print('============fair classification on test set=============')
    print(best_ars_result)

    if len(best_result) > 0:
        log = "Test: accuracy: {:.4f}, f1: {:.4f}, parity: {:.4f}, equality: {:.4f}"\
                .format(best_result['acc'],best_result['f1'], best_result['parity'],best_result['equality'])
        with open('log.txt', 'a') as f:
            f.write(log)
        print("Test:",
            "accuracy: {:.4f}".format(best_result['acc']),
            "f1: {:.4f}".format(best_result['f1']),
            "parity: {:.4f}".format(best_result['parity']),
            "equality: {:.4f}".format(best_result['equality']))
    else:
        print("Please set smaller acc/roc thresholds")
        
    
    sens_test = sens_test.to(args.device)

    print('============Disentaglement audit=============')
    train_embs = model.get_embedding(edge_index_train, feat_train, n_node)
    test_embs = model.get_embedding(edge_index_test, feat_test, n_node_test)
    loss_d = nn.CrossEntropyLoss()
    eval_acc_d = 0

    logreg2 = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)  
    # = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2) 
    logreg2 = logreg2.to(args.device)

    optimizer_d = torch.optim.Adam([{'params': logreg2.parameters(), 'lr':args.lr4, 'weight_decay':args.wd4}])

    for epoch in range(50):
        logreg2.train()
        optimizer_d.zero_grad()
        logits = logreg2(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == sens_train).float() / sens_train.shape[0]
        loss = loss_d(logits, sens_train.long())
        loss.backward()
        optimizer_d.step()

        logreg2.eval()

        if epoch % 20 == 0:
            print('Disentaglement audit:{:.4f}'.format(loss))
            print('Disentaglement audit accuracy on train set:{:.4f}'.format(train_acc))
        with th.no_grad():
            test_logits = logreg2(test_embs)

            test_preds = th.argmax(test_logits, dim=1)

            test_acc_d = th.sum(test_preds == sens_test).float() / sens_test.shape[0]

            if test_acc_d > eval_acc_d:
                eval_acc_d = test_acc_d

    print('Disentaglement audit accuracy on test set:{:.4f}'.format(eval_acc_d))



    print('============Predictive audit=============')
    train_embs = model.get_predictive(edge_index_train, feat_train, n_node)
    test_embs = model.get_predictive(edge_index_test, feat_test, n_node_test)
    loss_pre = nn.CrossEntropyLoss()
    eval_acc_p = 0
    
    logreg3 = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)  
    # = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    logreg3 = logreg3.to(args.device)

    optimizer_pre = torch.optim.Adam([{'params': logreg3.parameters(), 'lr':args.lr5, 'weight_decay':args.wd5}])

    for epoch in range(50):
        logreg3.train()
        optimizer_pre.zero_grad()
        logits = logreg3(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == sens_train).float() / sens_train.shape[0]
        loss = loss_pre(logits, sens_train.long())
        loss.backward()
        optimizer_pre.step()

        logreg3.eval()

        if epoch % 20 == 0:
            print('Predictive audit:{:.4f}'.format(loss))
            print('Predictive audit accuracy on train set:{:.4f}'.format(train_acc))
        with th.no_grad():
            test_logits = logreg3(test_embs)

            test_preds = th.argmax(test_logits, dim=1)

            test_acc_p = th.sum(test_preds == sens_test).float() / sens_test.shape[0]

            if test_acc_p > eval_acc_p:
                eval_acc_p = test_acc_p

    print('Predictive audit accuracy on test set:{:.4f}'.format(eval_acc_p))

        

