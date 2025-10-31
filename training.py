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


printcnt = 0

def fair_metric(val_labels,sens_test,sens2_test,val_preds):
    global printcnt
    printcnt += 1;
    PrintcntSH = 1000

    val_y=val_labels.cpu().numpy()

    # list 4 sub-sensitive group sa&sb
    idxs_subsens = [];
    for sa in range(0,2):
        for sb in range(0,2):
            idxs_subsens.append((sens_test == sa) & (sens2_test == sb))

    # sa&sb & y = 1
    idxs_subsens_y1 = [(idx_ssub & (val_y == 1)).to(torch.bool) for idx_ssub in idxs_subsens]

    # y^ = 1
    pred_y = (val_preds.squeeze() > 0).type_as(val_labels).cpu().numpy()

    # pre calculate
    sumss = [sum(idx_ssub) for idx_ssub in idxs_subsens]
    sumss_py1 = [sum(pred_y[idx_ssub]) for idx_ssub in idxs_subsens]
    sumss_y1 = [sum(idx_sysub) for idx_sysub in idxs_subsens_y1]
    sumss_y1_py1 = [sum(pred_y[idx_sysub]) for idx_sysub in idxs_subsens_y1]

    # calculate SP
    SP_subs = [(1 if sumss[x] == 0 else sumss_py1[x]/sumss[x]) for x in range(0,4)]
    SPd = max(SP_subs) - min(SP_subs)
    SPv = np.std(SP_subs)
    if (printcnt % PrintcntSH == 0):
        print("\tSP", SP_subs)
        for x in range(0,4):
            print("\tSP sub-sens:",sumss_py1[x],":",sumss[x])

    # calculate EO
    EO_subs = [(1 if sumss_y1[x] == 0 else sumss_y1_py1[x]/sumss_y1[x]) for x in range(0,4)]
    EOd = max(EO_subs) - min(EO_subs)
    EOv = np.std(EO_subs)
    if (printcnt % PrintcntSH == 0):
        print("\tEO", EO_subs)
        for x in range(0,4):
            print("\tEO sub-sens:",sumss_y1_py1[x],":",sumss_y1[x])

    # calculate UC
    UC_subs = [(1 if sumss_py1[x] == 0 else sumss_y1_py1[x]/sumss_py1[x]) for x in range(0,4)]
    UCd = max(UC_subs) - min(UC_subs)
    UCv = np.std(UC_subs)
    if (printcnt % PrintcntSH == 0):
        print("\tUC", UC_subs)
        for x in range(0,4):
            print("\tUC sub-sens:",sumss_y1_py1[x],":",sumss_py1[x])

    # old : sp = abs(sum(pred_y[idx1_s0])/sum(idx1_s0)-sum(pred_y[idx1_s1])/sum(idx1_s1))
    # old : eo = abs(sum(pred_y[idx1_s0_y1])/sum(idx1s0_y1)-sum(pred_y[idx1_s1_y1])/sum(idx_s1_y1))

    # return SP_subs, EO_subs, UC_subs
    return SPd, SPv, EOd, EOv, UCd, UCv



def fair_loss(train_labels,sens_train,sens2_train,logits):
    train_y=train_labels
    
    idx_s0s0 = (sens_train==0) & (sens2_train==0)
    idx_s1s0 = (sens_train==1) & (sens2_train==0)
    idx_s0s1 = (sens_train==0) & (sens2_train==1)
    idx_s1s1 = (sens_train==1) & (sens2_train==1)

    idx_s0s0_y1 = (idx_s0s0 & (train_y == 1)).to(torch.bool)
    idx_s0s1_y1 = (idx_s0s1 & (train_y == 1)).to(torch.bool)
    idx_s1s0_y1 = (idx_s1s0 & (train_y == 1)).to(torch.bool)
    idx_s1s1_y1 = (idx_s1s1 & (train_y == 1)).to(torch.bool)


    sp00 = torch.mean(logits[idx_s0s0])
    sp10 = torch.mean(logits[idx_s1s0])
    sp01 = torch.mean(logits[idx_s0s1])
    sp11 = torch.mean(logits[idx_s1s1])
    sp_max = torch.max(torch.stack([sp00, sp10, sp01, sp11]))
    sp_min = torch.min(torch.stack([sp00, sp10, sp01, sp11]))

    eo00 = torch.mean(logits[idx_s0s0_y1])
    eo10 = torch.mean(logits[idx_s0s1_y1])
    eo01 = torch.mean(logits[idx_s1s0_y1])
    eo11 = torch.mean(logits[idx_s1s1_y1])
    eo_max = torch.max(torch.stack([eo00, eo10, eo01, eo11]))
    eo_min = torch.min(torch.stack([eo00, eo10, eo01, eo11]))


    uc00 = eo00 / sp00 / sum(idx_s0s0) * (idx_s0s0_y1)
    uc10 = eo10 / sp10 / sum(idx_s1s0) * (idx_s1s0_y1)
    uc01 = eo01 / sp01 / sum(idx_s0s1) * (idx_s0s1_y1)
    uc11 = eo11 / sp11 / sum(idx_s1s1) * (idx_s1s1_y1)
    uc_max = torch.max(torch.stack([uc00, uc10, uc01, uc11]))
    uc_min = torch.min(torch.stack([uc00, uc10, uc01, uc11]))

    return sp_max,sp_min,eo_max,eo_min,uc_max,uc_min



def split(label_idx,feat,label,edge_index,seed,label_number,test_idx):
    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    print('Length of training set',len(idx_train))

    if test_idx==True:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):] 


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
    sens,sens2=dataset.sens()

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
    sens2_train=sens2[idx_train]
    sens_test=sens[idx_test]
    sens2_test=sens2[idx_test]

    idx_s0, idx_s1, edge_index_s0, edge_index_s1=add_s_train(idx_train,sens_train,edge_index)

    feat_train_s0=feat[idx_s0]
    feat_train_s1=feat[idx_s1]


    n_feat = feat_train.shape[1]
    n_classes = np.unique(label_train).shape[0]
    print('Classes number：',n_classes)

    edge_index_train = edge_index_train.to(args.device)
    feat_train = feat_train.to(args.device)
    sens_train = sens_train.to(args.device)
    sens2_train = sens2_train.to(args.device)
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
                                {'params': model.weight_params, 'weight_decay': args.wd, 'lr': args.lr}
                                ])

    
    loss_p = nn.BCEWithLogitsLoss()
    loss_h = nn.BCEWithLogitsLoss()

    loss_cs = nn.CrossEntropyLoss()
    #loss_cs = nn.BCEWithLogitsLoss()
    loss_sp=nn.MSELoss()
    loss_eo=nn.MSELoss()
    loss_uc = nn.MSELoss()

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

            sp_max,sp_min,eo_max,eo_min,uc_max,uc_min = fair_loss(label_train,sens_train,sens2_train,logits)
            
            train_acc = th.sum(preds == label_train).float() / label_train.shape[0]        

            loss_f=loss_h(out2, lbls)+ loss_h(out3, lbls) + loss_sp(sp_max,sp_min) + loss_eo(eo_max,eo_min)+loss_uc(uc_max,uc_min)

        
            loss =  a*loss_f + b*loss_p(out1, lbl) 

            loss.backward()    
            optimizer_p.step()

            logreg.eval()   
            model.eval()    

            if epoch % 20 == 0:
                print("Epoch: {0}, All Loss: {1:0.4f}, FN1 Loss: {2:0.4f}, FN2 Loss: {3:0.4f}, SP Loss: {4:0.4f}, EO Loss: {5:0.4f},UC Loss: {6:0.4f}.".format(epoch, loss.item(),loss_p(out1, lbl),loss_h(out2, lbls), loss_sp(sp_max,sp_min), loss_eo(eo_max,eo_min),loss_uc(uc_max,uc_min)))

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

    def recode_an_dprint_result(best_ars_result, result_type, acc_test, f1_test, SPd, SPv, EOd, EOv, UCd, UCv):
        best__result = {}
        best__result['acc'] = acc_test.item()
        best__result['f1'] = f1_test
        best__result['SPd'] = SPd
        best__result['SPvar'] = SPv
        best__result['EOd'] = EOd
        best__result['EOvar'] = EOv
        best__result['UCd'] = UCd
        best__result['UCvar'] = UCv
        best_ars_result[f'{result_type}_result'] = best__result
        print(f"Test {result_type}:",
              "accuracy: {:.4f}".format(acc_test.item()),
              "f1: {:.4f}".format(f1_test),
              "SPd: {:.4f}".format(SPd),"SPvar: {:.4f}".format(SPv),"EOd: {:.4f}".format(EOd),"EOvar: {:.4f}".format(EOv),"UCd: {:.4f}".format(UCd),"UCvar: {:.4f}".format(UCv))
        return best__result


    # Update_Threshold = 0.00005
    for epoch in range(1000):
        print(f"\r{str(epoch).zfill(4)}/1000", end = ": ", flush=True)
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
            
            SPd,SPv,EOd,EOv,UCd,UCv = fair_metric(label_test,sens_test,sens2_test,test_preds)

        if best_acc <= acc_test:
            best_acc = acc_test
            best_acc_result = recode_an_dprint_result(best_ars_result, "best_acc", acc_test, f1_test, SPd,SPv,EOd,EOv,UCd,UCv)
            # best_acc_result = {}
            # best_acc_result['acc'] = acc_test.item()
            # best_acc_result['parity'] = parity
            # best_acc_result['equality'] = equality
            # best_ars_result['best_acc_result'] = best_acc_result
            # print("Test best_acc:",
            #         "accuracy: {:.4f}".format(acc_test.item()),
            #         "f1: {:.4f}".format(f1_test),
            #         "parity: {:.4f}".format(parity),
            #         "equality: {:.4f}".format(equality))
            
        if best_f1 <= f1_test:
            best_f1 = f1_test
            best_f1_result = recode_an_dprint_result(best_ars_result, "best_f1", acc_test, f1_test, SPd,SPv,EOd,EOv,UCd,UCv)
            # best_f1_result = {}
            # best_f1_result['acc'] = acc_test.item()
            # best_f1_result['f1'] = f1_test
            # best_f1_result['parity'] = parity
            # best_f1_result['equality'] = equality
            # best_ars_result['best_f1_result'] = best_f1_result
            # print("Test best_f1:",
            #         "accuracy: {:.4f}".format(acc_test.item()),
            #         "f1: {:.4f}".format(f1_test),
            #         "parity: {:.4f}".format(parity),
            #         "equality: {:.4f}".format(equality))

        
        if best_ar <= f1_test + acc_test:
            best_ar = f1_test + acc_test
            best_ar_result = recode_an_dprint_result(best_ars_result, "best_ar", acc_test, f1_test, SPd,SPv,EOd,EOv,UCd,UCv)
            # best_ar_result = {}
            # best_ar_result['acc'] = acc_test.item()
            # best_ar_result['f1'] = f1_test
            # best_ar_result['parity'] = parity
            # best_ar_result['equality'] = equality
            # best_ars_result['best_ar_result'] = best_ar_result
            # print("Test best_ar:",
            #         "accuracy: {:.4f}".format(acc_test.item()),
            #         "f1: {:.4f}".format(f1_test),
            #         "parity: {:.4f}".format(parity),
            #         "equality: {:.4f}".format(equality))
                    
        if acc_test > args.acc and f1_test > args.f1:
            if best_fair > SPd + EOd + UCd:
                best_fair = SPd + EOd + UCd
                best_result = recode_an_dprint_result(best_ars_result, "best_fair", acc_test, f1_test, SPd,SPv,EOd,EOv,UCd,UCv)
                # best_result['acc'] = acc_test.item()
                # best_result['f1'] = f1_test
                # best_result['parity'] = parity
                # best_result['equality'] = equality 
                # print("Test best_fair:",
                #     "accuracy: {:.4f}".format(acc_test.item()),
                #     "f1: {:.4f}".format(f1_test),
                #     "parity: {:.4f}".format(parity),
                #     "equality: {:.4f}".format(equality))

        #print('Linear evaluation accuracy on train dataset:{:.4f}'.format(train_acc))
        #print('Linear evaluation accuracy on test dataset:{:.4f}'.format(eval_acc))
        #print('Linear evaluation fairmetric on validation dataset:{:.4f}，{:.4f}'.format(best_t_parity,best_t_equality))

    print('============fair classification on test set=============')
    print(best_ars_result)

    if len(best_result) > 0:
        log = "Test: accuracy: {:.4f}, f1: {:.4f}, SPd: {:.4f}, SPvar: {:.4f}, EOd: {:.4f}, EOvar: {:.4f}, UCd: {:.4f}, UCvar: {:.4f}"\
                .format(best_result['acc'],best_result['f1'], best_result['SPd'],best_result['SPvar'], best_result['EOd'],best_result['EOvar'], best_result['UCd'],best_result['UCvar'])
        with open('log.txt', 'a') as f:
            f.write(log)
        print(log)
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

        

