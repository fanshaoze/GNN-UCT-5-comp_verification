import pprint

import torch
import torch.nn.functional as F

import numpy as np
import math

from easydict import EasyDict
from sklearn.metrics import roc_curve, auc
from model_different_gnn_encoders import  PT_GNN, Serial_GNN, LOOP_GNN
import copy

from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
import torch

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from decimal import Decimal
from reward_fn import compute_reward

import torch_geometric


def rse(y,yt):

    assert(y.shape==yt.shape)

    if len(y)==0:
        return 0,0
    var=0
    m_yt=yt.mean()
#    print(yt,m_yt)
    for i in range(len(yt)):
        var+=(yt[i]-m_yt)**2
#    print("len(y)",len(y))
    var = var/len(y)
    mse=0
    for i in range(len(yt)):
        mse+=(y[i]-yt[i])**2
    mse = mse/len(y)
    # print("var: ", var)
    # print("mse: ",mse)
    rse=mse/(var+0.0000001)

    rmse=math.sqrt(mse/len(yt))

#    print(rmse)

    return rse,mse


def compute_roc(preds, ground_truth,th_true):
    """
    Generate TPR, FPR points under different thresholds for the ROC curve.
    Reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: {threshold: {'TPR': true positive rate, 'FPR': false positive rate}}
    """
    preds, ground_truth = np.array(preds), np.array(ground_truth)

    for i in range(len(ground_truth)):

        if ground_truth[i] <= th_true:
            ground_truth[i] = 0
        else:
            ground_truth[i] = 1


    fpr, tpr, threshold = roc_curve(ground_truth, preds,pos_label=1)
    AUC = auc(fpr, tpr)
    return fpr, tpr, AUC


def initialize_model(model_index, gnn_nodes, gnn_layers, pred_nodes, nf_size, ef_size, device, output_size=1):
    #初始化模型
    args = EasyDict()
    args.len_hidden = gnn_nodes
    args.len_hidden_predictor = pred_nodes
    args.len_node_attr = nf_size
    args.len_edge_attr = ef_size
    args.gnn_layers = gnn_layers
    args.use_gpu = False
    args.dropout = 0
    args.output_size = output_size

    if model_index == 1:
        model = PT_GNN(args).to(device)
        return model
    elif model_index == 2:
        model = Serial_GNN(args).to(device)
        return model
    elif model_index == 3:
        model = LOOP_GNN(args).to(device)
        return model
    else:
        assert ("Invalid model")

    #选择model

def train(train_loader, val_loader, model, n_epoch, batch_size, num_node, device, model_index, optimizer,gnn_layers):
    train_perform=[]

    min_val_loss=100

    for epoch in range(n_epoch):
    
    ########### Training #################
        
        train_loss=0
        n_batch_train=0
    
        model.train()

        for i, data in enumerate(train_loader):
                 data.to(device)
                 L=data.node_attr.shape[0]
                 B=int(L/num_node)
#                 print(L,B,data.node_attr)
                 node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                 if model_index == 0:
                     edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                 else:
                     edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                     edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])
 
                 adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                 y=data.label
                 n_batch_train=n_batch_train+1
                 optimizer.zero_grad()
                 if model_index == 0:
                      out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                 else:
                      out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device), gnn_layers))
 
                 out=out.reshape(y.shape)
                 assert(out.shape == y.shape)
                 loss=F.mse_loss(out, y.float())

#                 loss=F.binary_cross_entropy(out, y.float())
                 loss.backward()
                 optimizer.step()
        
                 train_loss += out.shape[0] * loss.item()
        
        if epoch % 1 == 0:
#                 print('%d epoch training loss: %.3f' % (epoch, train_loss/n_batch_train/batch_size))
                 if epoch % 5 == 0:
                    print("epoch: ", epoch)
                 n_batch_val=0
                 val_loss=0

#                epoch_min=0
                 model.eval()

                 for data in val_loader:

                     n_batch_val+=1

                     data.to(device)
                     L=data.node_attr.shape[0]
                     B=int(L/num_node)
                     node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
                     if model_index == 0:
                         edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
                     else:
                         edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                         edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])

                     adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
                     y=data.label

                     n_batch_train=n_batch_train+1
                     optimizer.zero_grad()
                     if model_index == 0:
                          out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device)))
                     else:
                          out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device),gnn_layers))

                     out=out.reshape(y.shape)
                     assert(out.shape == y.shape)
#                     loss=F.binary_cross_entropy(out, y.float())
                     loss=F.mse_loss(out,y.float())
                     val_loss += out.shape[0] * loss.item()
                 val_loss_ave=val_loss/n_batch_val/batch_size
                 

                 if val_loss_ave<min_val_loss:
                    model_copy=copy.deepcopy(model)
 #                   print('lowest val loss', val_loss_ave)
                    epoch_min=epoch
                    min_val_loss=val_loss_ave

                 if epoch-epoch_min>5:
                    #print("training loss:",train_perform)
 #                   print("training loss minimum value:", min(train_perform))
 #                   print("training loss average value:", np.mean(train_perform))

                    return model_copy, min(train_perform), np.mean(train_perform)


        train_perform.append(train_loss/n_batch_train/batch_size)

    return model, min(train_perform), np.mean(train_perform)


def test(test_loader, model, num_node, model_index, device,gnn_layers, TN, FN):
        model.eval()
        accuracy=0
        n_batch_test=0
        gold_list=[]
        out_list=[]
        analytic_list = []


        TPR=0
        FPR=0
        count=0
        for data in test_loader:
             data.to(device)
             L=data.node_attr.shape[0]
             B=int(L/num_node)
             node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
             if model_index == 0:
                 edge_attr=torch.reshape(data.edge0_attr,[B,int(L/B),int(L/B),-1])
             else:
                 edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
                 edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])

             adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
             y=data.label.cpu().detach().numpy()

             n_batch_test=n_batch_test+1
             if model_index==0:
                  out=model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
             else:
                  out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device),gnn_layers)).cpu().detach().numpy()
             out=out.reshape(y.shape)
             assert(out.shape == y.shape)
             out=np.array([x for x in out])
             # Shun: the following needs to be disabled for reg_both
             # It shouldn't affect reg_eff, reg_vout, etc.
             #gold=np.array(y.reshape(-1))
             gold=np.array([x for x in y])

             gold_list.extend(gold)
             out_list.extend(out)

             L=len(gold)
             np.set_printoptions(precision=2,suppress=True)

#             result_bins = compute_errors_by_bins(np.reshape(out_list,-1),np.reshape(gold_list,-1),[(-0.1,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.1)])

        final_rse, final_mse = rse(np.reshape(out_list,-1),np.reshape(gold_list,-1))
#        print("Final RSE:", final_rse)
        fpr, tpr, auc = compute_roc(np.reshape(out_list,-1),np.reshape(gold_list,-1),0.5)
 
        gold_list.extend([0]*TN)
        gold_list.extend([1]*FN)

        out_list.extend([0]*TN)
        out_list.extend([0]*FN)


        final_rse_filter, final_mse_filter = rse(np.reshape(out_list,-1),np.reshape(gold_list,-1))
#        print("Final RSE:", final_rse)
        fpr_filter, tpr_filter, auc_filter = compute_roc(np.reshape(out_list,-1),np.reshape(gold_list,-1),0.5)
        return final_rse, final_mse, fpr, tpr, auc, final_rse_filter,final_mse_filter, fpr_filter, tpr_filter, auc_filter


def compute_errors_by_bins(pred_y:np.array, true_y:np.array, bins):
    """
    Divide data by true_y into bins, report their rse separately
    :param pred_y: model predictions (of the test data)
    :param true: true labels (of the test data)
    :param bins: a list of ranges where errors in these ranges are computed separately
                 e.g. [(0, 0.33), (0.33, 0.66), (0.66, 1)]
    :return: a list of rses by bins
    """
    results = []

    for range_from, range_to in bins:
        # get indices of data in this range
        indices = np.nonzero(np.logical_and(range_from <= true_y, true_y < range_to))

        if len(indices) > 0:
            temp_rse, temp_mse = rse(pred_y[indices], true_y[indices])
            results.append(math.sqrt(temp_mse))
            # print('data between ' + str(range_from) + ' ' + str(range_to))
            # pprint.pprint(list(zip(pred_y[indices], true_y[indices])))
        else:
            print('empty bin in the range of ' + str(range_from) + ' ' + str(range_to))

    return results


def split_balance_data(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    # print("train_ratio", train_ratio)
    # print("val_ratio", val_ratio)
    # print("test_ratio", test_ratio)

    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_train = int(dataset_size * train_ratio)
    n_val = int(dataset_size * val_ratio)
    n_test = int(dataset_size * test_ratio)

    train_indices, val_indices, test_indices = indices[:n_train], indices[n_train + 1:n_train + n_val], indices[
                                                                                                        n_train + n_val + 1:n_train + n_val + n_test]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader

def dataset_balance_indices(dataset,indices,reward_th,flag_extend):
    ind = 0
    ind_positive = []
    ind_negative = []
    
    for data in dataset:
        if ind in indices:
            flag_cls = data['label'].tolist()[0]
            if flag_cls > reward_th:
                ind_positive.append(ind)
            else:
                ind_negative.append(ind)
        ind+=1
    ind_new = ind_negative

    print('#Good topo: ', len(ind_positive))
    print('#Bad topo: ', len(ind_negative))
    
    if flag_extend == 1:
        for i in range(int(len(ind_negative)/(len(ind_positive)+1))):
            ind_new.extend(ind_positive)
    else:
        ind_new.extend(ind_positive)


    return ind_new, int(len(ind_negative)/(len(ind_positive)+1))

def filter_by_groundtruth(dataset, cls_th, true_th):
    
    indices=[]
    dataset_size = len(dataset)
 
    ind = 0
    P = 0
    N = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0



    for data in dataset:
        flag_cls = data['label'].tolist()[0]

        if flag_cls >= true_th:
                P+=1
        else:
                N+=1

        if flag_cls >= cls_th:
            indices.append(ind)
            if flag_cls >= true_th:
                TP+=1
        else:
            TN+=1

        ind+=1

    FN=P-TP
    FP=N-TN
    
    return indices,TP, FP, TN, FN

def filter_by_model(dataset, device, num_node, gnn_layers, model, cls_th, true_th):
    
    indices=[]
    dataset_size = len(dataset)

    Batch_Size=1024

    test_loader = DataLoader(dataset, batch_size=Batch_Size, )


    n_batch = 0
    P = 0
    N = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for data in test_loader:
             print(n_batch)
             data.to(device)
             L=data.node_attr.shape[0]
             B=int(L/num_node)
             node_attr=torch.reshape(data.node_attr,[B,int(L/B),-1])
             edge_attr1=torch.reshape(data.edge1_attr,[B,int(L/B),int(L/B),-1])
             edge_attr2=torch.reshape(data.edge2_attr,[B,int(L/B),int(L/B),-1])

             adj=torch.reshape(data.adj,[B,int(L/B),int(L/B)])
             y=data.label.cpu().detach().numpy()

             out=model(input=(node_attr.to(device), edge_attr1.to(device),edge_attr2.to(device), adj.to(device),gnn_layers)).cpu().detach().numpy()
             out=out.reshape(y.shape)

             for i in range(len(y)):
                  if y[i] >= true_th:
                      P+=1
                      if out[i] >= cls_th:
                          ind=n_batch*Batch_Size+i
                          indices.append(ind)
                          TP+=1
                      else:
                          FN+=1
                  else:
                      if out[i] >= cls_th:
                          ind=n_batch*Batch_Size+i
                          indices.append(ind)
                          FP+=1
                      else:
                          TN+=1

             n_batch+=1
   
    return indices, TP, FP, TN, FN

def split_data_by_filter(dataset, batch_size, rtrain, rval, rtest, device, num_node, gnn_layers, model,cls_th,true_th):
    shuffle_dataset = True
    random_seed = 42

    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    print("train_ratio", train_ratio)
    print("val_ratio", val_ratio)
    print("test_ratio", test_ratio)


    dataset_size = len(dataset)

    if model is None:
        print('Filter by groundtruth')
        indices,TP, FP, TN, FN = filter_by_groundtruth(dataset,cls_th,true_th)
    else:
        print('Filter by model')
        indices,TP, FP, TN, FN = filter_by_model(dataset,device,num_node,gnn_layers,model,cls_th,true_th)

    print("TP,FP,TN,FN:",[TP,FP,TN,FN])

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    dataset_size = len(indices)
    print(dataset_size)

    n_train=int(dataset_size * train_ratio)
    n_val=int(dataset_size * val_ratio)
    n_test=int(dataset_size * test_ratio)
    
    train_indices = indices[0:n_train]
    val_indices = indices[n_train+1:n_train+n_val]
    test_indices = indices[n_train+n_val+1:n_train+n_val+n_test]
    
    train_indices_new,np_ratio = dataset_balance_indices(dataset,train_indices,true_th,1)
    val_indices_new,np_ratio = dataset_balance_indices(dataset,val_indices,true_th,1)
    test_indices_new,np_ratio = dataset_balance_indices(dataset,test_indices,true_th,0)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices_new)
        np.random.shuffle(val_indices_new)
        np.random.shuffle(test_indices_new)


    train_sampler = SubsetRandomSampler(train_indices_new)
    valid_sampler = SubsetRandomSampler(val_indices_new)
    test_sampler = SubsetRandomSampler(test_indices_new)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader, TP, FP, TN, FN

def split_imbalance_data_reward(dataset, batch_size, rtrain, rval, rtest):
    shuffle_dataset = True
    random_seed = 42
    reward_th = 0.5
    bad_th = -1

    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    print("train_ratio", train_ratio)
    print("val_ratio", val_ratio)
    print("test_ratio", test_ratio)


    dataset_size = len(dataset)
    indices,TN,FN = filter_by_groundtruth(dataset,bad_th)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    dataset_size = len(indices)

    n_train=int(dataset_size * train_ratio)
    n_val=int(dataset_size * val_ratio)
    n_test=int(dataset_size * test_ratio)
    
    train_indices = indices[0:n_train]
    val_indices = indices[n_train+1:n_train+n_val]
    test_indices = indices[n_train+n_val+1:n_train+n_val+n_test]
    
    train_indices_new,np_ratio = dataset_balance_indices(dataset,train_indices,reward_th,1)
    val_indices_new,np_ratio = dataset_balance_indices(dataset,val_indices,reward_th,1)
    test_indices_new,np_ratio = dataset_balance_indices(dataset,test_indices,reward_th,0)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices_new)
        np.random.shuffle(val_indices_new)
        np.random.shuffle(test_indices_new)


    train_sampler = SubsetRandomSampler(train_indices_new)
    valid_sampler = SubsetRandomSampler(val_indices_new)
    test_sampler = SubsetRandomSampler(test_indices_new)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader, TP, FP, TN, FN
