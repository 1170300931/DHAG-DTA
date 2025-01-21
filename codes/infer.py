import os

from collections import OrderedDict
import json
import shutil
import time
import pickle
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch

from dataset import GraphDataset
from model import GNN, Predictor

from args import argparser
from data_process import (create_test_dataset,
                            getDrugMolecularGraph, getTargetMolecularGraph,
                            getSimMatrix)
from logs import Logger
from metrics import model_evaluate
from utils import infer
import torch_geometric.data as DATA


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
    print("skip:", args.skip)
    print("edge ratio:", args.ratio)
    print('sim_d:', args.sim_d)
    print('sim_t:', args.sim_t)
    print("mode:{}".format('train_test') if args.fold == -100 else f"5-CV({args.fold})")
    print("model params: {}".format(sum(p.numel() for p in GNN().parameters())))
    # print("model archetecture: ")
    # print(GNN())

    # logger.reset() # stop write log

    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')

    # load checkpoint
    checkpoint = torch.load(f'checkpoints/{args.dataset}_{args.split_mode}.pkl')
    gnn_param = checkpoint['gnn']
    predictor_param = checkpoint['predictor']
    adj = checkpoint['affinitygraph']

    print("create test dataloader ...")
    # create test
    test_data, affinity_graph = create_test_dataset(args.dataset, args.fold, args.ratio, args.split_mode, adj)
    drug_sim, target_sim = getSimMatrix(args.dataset, args.split_mode, args.sim_d, args.sim_t) # sim matrix
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

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
    gnn.load_state_dict(gnn_param, strict=False)
    predictor = Predictor(skip=args.skip).to(device)
    predictor.load_state_dict(predictor_param, strict=False)

    print("start infering ...")

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

    G, P = infer(gnn, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_sim, target_sim, args.split_mode)
    result = model_evaluate(G, P, args.dataset, 0, writer, 0)
    print('MSE: ', result[0])
    print('CI: ', result[1])
    print('rm2: ', result[2])
    print('PCC: ', result[3])
    print('AUPR: ', result[4])

    writer.close()
    
if __name__ == '__main__':
    start = time.time()
    main()
    print('running time: {}s'.format(time.time() - start))