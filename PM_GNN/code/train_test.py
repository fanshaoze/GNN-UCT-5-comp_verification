import csv
import os
from datetime import datetime
import numpy as np
import torch
from torch.nn import Linear, MSELoss
from torch.utils.data.sampler import SubsetRandomSampler

from topo_data import *

from ml_utils import *
import argparse


if __name__ == '__main__':

# ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-y_select', type=str, default='reg_reward', help='define target label')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=40, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=10, help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=4, help='number of layer')
    parser.add_argument('-model_index', type=int, default=3, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')
    parser.add_argument('-ncomp', type=int, default=5, help='# components')
    parser.add_argument('-train_ratio', type=float, default=0.7, help='# components')
    parser.add_argument('-val_ratio', type=float, default=0.1, help='# components')
 
    parser.add_argument('-retrain', type=int, default=1, help='force retrain model')
#    parser.add_argument('-seed', type=int, default=0, help='random seed')
    parser.add_argument('-seedrange', type=int, default=1, help='random seed')
#----------------------------------------------------------------------------------------------------#
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='weight decay')
    #parser.add_argument('-loops_num',type=int, default=6,help='loops number for edge attr encoder')
#----------------------------------------------------------------------------------------------------#

    args = parser.parse_args()
#    print("\nargs: ",args)


    ncomp=args.ncomp
    train_ratio=args.train_ratio
    val_ratio=args.val_ratio
    path=args.path
    y_select=args.y_select
    gnn_layers=args.gnn_layers
    gnn_nodes=args.gnn_nodes
    data_folder='../2_dataset/'+y_select+'_'+str(ncomp)
    batch_size=args.batch_size
    n_epoch=args.n_epoch
    th=args.threshold
    model_index=args.model_index
    retrain=args.retrain


    lr = args.lr
    weight_decay = args.weight_decay
    seedrange = args.seedrange

    output_file = datetime.now().strftime(y_select + '-'  + str(ncomp))
    result_metric = []


# ======================== Data & Model ==========================#

    dataset = Autopo(data_folder,path,y_select,ncomp)

    print('\n # data point:\n', len(dataset))

    if y_select=='cls_buck':
                train_loader, val_loader, test_loader = split_imbalance_data_cls(dataset, batch_size,train_rate,0.05,0.05)
    elif y_select=='reg_reward':
                 print('reward prediction')
                 train_loader, val_loader, test_loader,TN,FN = split_imbalance_data_reward(dataset, batch_size,train_ratio, val_ratio,1-train_ratio-val_ratio)
 
    else:
                train_loader, val_loader, test_loader = split_balance_data(dataset, batch_size,train_rate,0.1,0.3)

    # # set random seed for training
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    for seed in range(seedrange):
        # set random seed for training
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print("seed: ", seed)

        nf_size=4
        ef_size=3
        nnode=8
        # if args.model_index==0:
        #     ef_size=6

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = dataset[0].to(device)
#        print("data: ", data)
#        print("data size:", data.size())

        model = initialize_model(model_index=args.model_index,
                                 gnn_nodes=args.gnn_nodes,
                                 gnn_layers=args.gnn_layers,
                                 pred_nodes=args.predictor_nodes,
                                 nf_size=nf_size,
                                 ef_size=ef_size,
                                 device=device,
                                 output_size=2 if y_select == 'reg_both' else 1)
        #print("model: ",model)

        postfix = str(ncomp) if device.type == 'cuda' else '_cpu'
#        pt_filename = y_select + postfix + 'model' + str(model_index) + \
#                    str(gnn_layers) + 'layers' + str(gnn_nodes) + 'nodes' + \
#                      str(ncomp) + 'comp' + '.pt'
        pt_filename = y_select + '-' + str(model_index) + '-' + str(ncomp) + '-' + str(seed)+'.pt'
#        print("pt filename",pt_filename)



        if os.path.exists(pt_filename) and retrain==0:
            print('loading model from pt file')

            model_state_dict, _ = torch.load(pt_filename)
            model.load_state_dict(model_state_dict)
        else:
            print('training')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = MSELoss(reduction='mean').to(device)
            model, min_loss, mean_loss = train(train_loader=train_loader,
                          val_loader=val_loader,
                          model=model,
                          n_epoch=n_epoch,
                          batch_size=batch_size,
                          num_node=nnode,
                          device=device,
                          model_index=args.model_index,
                          optimizer=optimizer,
                          gnn_layers=gnn_layers)

            # save model and test data
            torch.save((model.state_dict(), test_loader), './pt/'+y_select + '-' + str(ncomp) +  '-' + str(seed) +'.pt')

        rse, mse, fpr, tpr, auc, rse_filter, mse_filter, fpr_filter, tpr_filter, auc_filter = test(test_loader=test_loader, model=model, num_node=nnode, model_index=args.model_index,device=device,gnn_layers=args.gnn_layers, TN=TN, FN=FN)
        print("rse: ", rse)
        print("mse: ", mse)
        print("auc: ",auc)
        print("rse_final: ", rse_filter)
        print("mse_final: ", mse_filter)
        print("auc_final: ",auc_filter)

        result_ROC = zip(fpr,tpr,fpr_filter,tpr_filter)

        result_metric.append([rse,mse,auc,rse_filter,mse_filter,auc_filter])

        with open('./csv/ROC/ROC-'+output_file + '-' + str(ncomp) + '-' + str(train_ratio) +'-' + str(seed) + '.csv','w') as f:
             csv_writer = csv.writer(f) 

             header = ['FPR','TPR','FPR(filter)','TPR(filter)']
             csv_writer.writerow(header)

             csv_writer.writerows(result_ROC)
 
    with open('./csv/ROC/Summary-'+output_file + '-' + str(ncomp) + '-' + str(train_ratio) +'.csv','w') as f:

             csv_writer = csv.writer(f)

             header = ['RSE','MSE','AUC','RSE(filter)','MSE(filter)','AUC(filter)']
             csv_writer.writerow(header)

             csv_writer.writerows(result_metric)
        
#
#    with open('./log/result-'+output_file + '.csv','w') as f:
#        csv_writer = csv.writer(f)
#        header = ['model_index','n_comp','y_select','gnn_layers','gnn_nodes',
#                  'min_loss','mean_loss','final_rse','mse[0-0.3]','mse[0.3-0.7]','mse[0.7-1]']
#        csv_writer.writerow(header)
#
#        csv_writer.writerows(final_result)
#
#
#
#
#    with open('./log/result-'+output_file + '.csv','w') as f:
#        csv_writer = csv.writer(f)
#        header = ['model_index','n_comp','y_select','gnn_layers','gnn_nodes',
#                  'min_loss','mean_loss','final_rse','mse[0-0.3]','mse[0.3-0.7]','mse[0.7-1]']
#        csv_writer.writerow(header)
#
#        csv_writer.writerows(final_result)




