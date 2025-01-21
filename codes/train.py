import os

from collections import OrderedDict
import json
import shutil
import time
import copy
import pickle
from itertools import chain
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch

from dataset import GraphDataset
from model import GNN, Predictor

from args import argparser
from data_process import (create_train_test_dataset,
                            getDrugMolecularGraph, getTargetMolecularGraph,
                            refineAdj,
                            getSimMatrix)
from logs import Logger
from metrics import model_evaluate
from utils import infer, train

def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch

# main program
def main():
    args = argparser()
    
    log_path = './logs/{}/{}'.format(args.dataset, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # record results
    writer = SummaryWriter(log_dir=log_path)
    # record some codes for recurrence
    shutil.copyfile('./codes/model.py', log_path + '/model.py')
    shutil.copyfile('./codes/data_process.py', log_path + '/data_process.py')
    os.mkdir(f'{log_path}/models')
    # write logs
    logger = Logger(log_path + '/log.txt')
    
    print("dataset:", args.dataset)
    print("split mode:", args.split_mode)
    print("cuda id:", args.cuda_id)
    print("epochs:", args.num_epochs)
    print("batch size:", args.batch_size)
    print("learning rate:", args.lr)
    print("skip:", args.skip)
    print("edge ratio:", args.ratio)
    print('refine epoch:', args.refine_epoch)
    print('sim_d:', args.sim_d)
    print('sim_t:', args.sim_t)
    print("mode:{}".format('train_test') if args.fold == -100 else f"5-CV({args.fold})")
    print("model params: {}".format(sum(p.numel() for p in GNN().parameters())))
    # print("model archetecture: ")
    # print(GNN())

    # logger.reset() # stop write log

    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')

    print("create train test dataloader ...")
    # create train and test
    train_data, test_data, prior_data, affinity_graph, refine_adj = create_train_test_dataset(args.dataset, args.fold, args.ratio, args.split_mode)
    drug_sim, target_sim = getSimMatrix(args.dataset, args.split_mode, args.sim_d, args.sim_t) # sim matrix
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    prior_loader = torch.utils.data.DataLoader(prior_data, batch_size=args.batch_size, shuffle=False)

    print("create graphs_dataloader ...")
    drug_graphs_dict = getDrugMolecularGraph(json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    target_graphs_dict = getTargetMolecularGraph(json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node1s)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node2s)

    drug_graphs_DataLoader = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graphs_DataLoader = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))

    # model
    gnn = GNN(edge_dropout_rate=args.dropedge_rate, skip=args.skip).to(device)
    predictor = Predictor(skip=args.skip).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, chain(gnn.parameters(), predictor.parameters())), lr=args.lr, weight_decay=0)

    best_result = [1000]
    best_loss = 50 # record total minimum train loss
    best_train_loss = 50 # record interval minimum train loss
    best_epoch = None
    best_architecture = None
    best_predictor = None
    print("start training ...")
    epoch = 0
    iter_num = 0

    # real affinity matrix
    true_affinity = pickle.load(open(f'data/{args.dataset}/affinities', 'rb'), encoding='latin1')
    if args.dataset == 'davis':
        true_affinity = [-np.log10(y / 1e9) for y in true_affinity]
    true_affinity = np.asarray(true_affinity)

    # 将remove变为0
    prior_idx = json.load(open(f'data/{args.dataset}/{args.split_mode}_prior_index.txt'))
    rows, cols = np.where(np.isnan(true_affinity) == False)
    prior_rows, prior_cols = rows[prior_idx], cols[prior_idx]
    true_affinity[prior_rows, prior_cols] = 0

    true_affinity = np.nan_to_num(true_affinity) # turn nan to 0

    while epoch < args.num_epochs:
        train_loss = train(gnn, predictor, optimizer, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, epoch, args.batch_size, affinity_graph, writer, iter_num)
        G, P = infer(gnn, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_sim, target_sim, args.split_mode)
        result = model_evaluate(G, P, args.dataset, epoch, writer, iter_num)
        print("reslut:", result)

        # record minimum train loss to refine adjacency matrix
        if train_loss < best_train_loss:
        # if False:
            best_train_loss = train_loss
            best_epoch = epoch
            # record best checkpoint
            best_architecture, best_predictor = copy.deepcopy(gnn), copy.deepcopy(predictor)

        # refine adjacency matrix
        if epoch in args.refine_epoch and best_train_loss < best_loss:
        # if epoch in args.refine_epoch:
        # # if False:
            iter_num += 1

            best_loss = best_train_loss
            best_train_loss = 50

            affinity_graph, refine_adj = refineAdj(args.dataset, best_architecture, best_predictor, device, test_loader, prior_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, args.ratio, drug_sim, target_sim, args.split_mode)
            pickle.dump(refine_adj, open(f'{log_path}/refineAdj_epoch{best_epoch}_iter{iter_num}.pkl', 'wb')) # save adjacency matrix

            # calculate refine error
            refine_adj[prior_rows, prior_cols] = 0
            err = (true_affinity - refine_adj) ** 2
            print(f'$$ refine epoch: {best_epoch}')
            print(f'$$ refine adj test mse: {np.sum(err) / np.sum(err > 0)}')

            # reset model parameters
            gnn = GNN(edge_dropout_rate=args.dropedge_rate, skip=args.skip).to(device)
            predictor = Predictor(skip=args.skip).to(device)

            # reset optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, chain(gnn.parameters(), predictor.parameters())), lr=args.lr, weight_decay=0)

            args.refine_epoch[args.refine_epoch.index(epoch)] = -1
            epoch = 0

        # # only save refineAdj
        # if epoch > 0 and epoch % 100 == 0:
        #     _, refine_adj = refineAdj(args.dataset, best_architecture, best_predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, args.ratio, drug_sim, target_sim, args.split_mode)
        #     pickle.dump(refine_adj, open(f'{log_path}/refineAdj_epoch{best_epoch}_iter{iter_num}.pkl', 'wb')) # save adjacency matrix

        #     # calculate refine error
        #     err = (true_affinity - refine_adj) ** 2
        #     print(f'$$ refine epoch: {best_epoch}')
        #     print(f'$$ refine adj test mse: {np.sum(err) / np.sum(err > 0)}')
            
        # save all model
        if epoch % 10 == 0:
            checkpoint_path = f"{log_path}/models/model_{epoch}.pkl"
            torch.save({
                        'gnn': gnn.state_dict(),
                        'predictor': predictor.state_dict(),
                        'affinitygraph': np.array(affinity_graph.adj.cpu())
                        }, checkpoint_path, _use_new_zipfile_serialization=False)

        # save checkpoint
        if result[0] < best_result[0]:
            best_result = result
            checkpoint_path = f"{log_path}/model.pkl"
            torch.save({
                        'gnn': gnn.state_dict(),
                        'predictor': predictor.state_dict(),
                        'affinitygraph': np.array(affinity_graph.adj.cpu())
                        }, checkpoint_path, _use_new_zipfile_serialization=False)
        
        epoch += 1

    print("best reslut:", best_result)
    writer.close()
    
if __name__ == '__main__':
    start = time.time()
    main()
    print('running time: {}s'.format(time.time() - start))
