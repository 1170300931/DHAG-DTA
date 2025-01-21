import copy
import os
import json
from collections import OrderedDict
import pickle
from collections import Counter
from multiprocessing import Pool
import itertools

import numpy as np

from rdkit import Chem
import networkx as nx

import torch
from torch_geometric import data as DATA
import torch.nn.functional as F

from dataset import DTADataset


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# Maps inputs not in the allowable set to the last element.
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index

# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return feature

# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)

    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index

# create molecule graph
def getDrugMolecularGraph(ligands):
    smile_graph = OrderedDict()
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smile_graph[d] = smile_to_graph(lg)
    
    # count the number of proteins with aln and contact files
    print('effective drugs:', len(smile_graph))
    if len(smile_graph) == 0:
        raise Exception('no drug, run the script for datasets preparation.')

    return smile_graph

def getTargetMolecularGraph(proteins, dataset):
    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'

    # create protein graph
    target_graph = OrderedDict()
    for t in proteins.keys():
        g = target_to_graph(t, proteins[t], contac_path, msa_path)
        target_graph[t] = g

    # count the number of proteins with aln and contact files
    print('effective protein:', len(target_graph))
    if len(target_graph) == 0:
        raise Exception('no protein, run the script for datasets preparation.')

    return target_graph

def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y

# choose topk neighbor
def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]

    return refine_adj

# calculate drug-drug similarity and target-target similarity using interactive network (l2-norm)
def getSimMatrix(dataset, split_mode, top_d, top_t):
    # 使用分子本身相似度
    drug_sim = np.loadtxt(f"data/{dataset}/drug-drug-sim.txt", delimiter=",")
    target_sim = np.loadtxt(f"data/{dataset}/target-target-sim.txt", delimiter=",")

    # 去掉unknown节点内部的相似度，确保unknown节点投影到了known空间中(mask set包含test和remove)
    unknown_drug_idx = []
    unknown_target_idx = []
    if split_mode == 'S1':
        return None, None

    if split_mode == 'S2':
        unknown_drug_idx = json.load(open(f"data/{dataset}/{split_mode}_drug_mask_set.txt"))
        for i in unknown_drug_idx:
            for j in unknown_drug_idx:
                drug_sim[i][j] = 0

    if split_mode == 'S3':    
        unknown_target_idx = json.load(open(f"data/{dataset}/{split_mode}_target_mask_set.txt"))
        for i in unknown_target_idx:
            for j in unknown_target_idx:
                target_sim[i][j] = 0
    # S4
    if split_mode == 'S4':
        unknown_drug_idx = json.load(open(f"data/{dataset}/{split_mode}_drug_mask_set.txt"))
        unknown_target_idx = json.load(open(f"data/{dataset}/{split_mode}_target_mask_set.txt"))
        for i in unknown_drug_idx:
            for j in unknown_drug_idx:
                drug_sim[i][j] = 0
        for i in unknown_target_idx:
            for j in unknown_target_idx:
                target_sim[i][j] = 0
    
    drug_sim = denseAffinityRefine(drug_sim, top_d)
    target_sim = denseAffinityRefine(target_sim, top_t)

    # normalize by row
    drug_sim = drug_sim / np.array([np.sum(drug_sim, axis=1)]).T
    target_sim = target_sim / np.array([np.sum(target_sim, axis=1)]).T

    # remove nan: nan means no share neighbor
    drug_sim = np.nan_to_num(drug_sim)
    target_sim = np.nan_to_num(target_sim)

    return drug_sim, target_sim

# transform affinity matrix into undirected graph adjacency matrix
def getAffinityGraph(dataset, affinity_matrix, ratio):
    num_drugs = affinity_matrix.shape[0]
    num_targets = affinity_matrix.shape[1]

    if dataset == "davis":
        affinity_matrix[affinity_matrix < 5] = 5 # make sure no negative num
        affinity_matrix[affinity_matrix != 0] -= 5 # remove 5

        topd, topt = search_param(copy.deepcopy(affinity_matrix), ratio, dataset)
        # topd, topt = 31, 182
        print(f'topd: {topd}\ttopt: {topt}')
        affinity_matrix = denseAffinityRefine(affinity_matrix.T, topd)
        affinity_matrix = denseAffinityRefine(affinity_matrix.T, topt)
        
        norm_matrix = minMaxNormalize(affinity_matrix, 0) # normalize matrix

    elif dataset == "kiba":
        topd, topt = search_param(copy.deepcopy(affinity_matrix), ratio, dataset)
        # topd, topt = 231, 34
        print(f'topd: {topd}\ttopt: {topt}')
        affinity_matrix = denseAffinityRefine(affinity_matrix.T, topd)
        affinity_matrix = denseAffinityRefine(affinity_matrix.T, topt)
        norm_matrix = minMaxNormalize(affinity_matrix, 0)

    bool_matrix = copy.deepcopy(norm_matrix)
    bool_matrix[bool_matrix > 0] = 1
    Ed, Et, Ea = entropy(list(sum(bool_matrix.T))), entropy(list(sum(bool_matrix))), entropy(list(sum(bool_matrix.T)) + list(sum(bool_matrix)))

    # transform into graph adj
    graph_adj = np.concatenate((
        np.concatenate((np.zeros([num_drugs, num_drugs]), norm_matrix), 1), 
        np.concatenate((norm_matrix.T, np.zeros([num_targets, num_targets])), 1)
    ), 0)

    # G = nx.from_numpy_matrix(adj_all)
    # print(nx.is_connected(G))

    train_raw_ids, train_col_ids = np.where(graph_adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_raw_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)

    affinity_graph = DATA.Data(x=None, adj=torch.Tensor(graph_adj), edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("num_node1s", num_drugs)
    affinity_graph.__setitem__("num_node2s", num_targets)

    return affinity_graph

def refineAdj(dataset, best_architecture, best_predictor, device, test_loader, prior_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, ratio, drug_sim, target_sim, split_mode='S1'):
    # real affinity matrix
    affinity = pickle.load(open(f'data/{dataset}/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    # load dataset
    dataset_path = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(dataset_path + f'{split_mode}_train_index.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        train_folds += train_fold_origin[i]

    rows, cols = np.where(np.isnan(affinity) == False)

    train_rows, train_cols = rows[train_folds], cols[train_folds]
    train_Y = affinity[train_rows, train_cols]

    train_adj = np.zeros_like(affinity)
    train_adj[train_rows, train_cols] = train_Y

    best_architecture.eval()
    best_predictor.eval()

    affinity_graph = affinity_graph.to(device)
    drug_sim, target_sim = torch.Tensor(drug_sim).to(device), torch.Tensor(target_sim).to(device)

    print('refining ……')
    with torch.no_grad():
        for data in test_loader:
            drug_idx, target_idx, aff = data
            # aff = aff.to(device)

            if split_mode == 'S1':
                drug_embedding, target_embedding = best_architecture(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader)
            else:
                # weighted sum
                drug_embedding, target_embedding = best_architecture(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader, split_mode, drug_sim, target_sim)
            
            output = best_predictor(drug_idx, target_idx, drug_embedding, target_embedding)

            # fill matrix
            train_adj[drug_idx, target_idx] = output.view(-1).cpu()
        
        # fill prior knowledge
        for data in prior_loader:
            drug_idx, target_idx, aff = data
            # aff = aff.to(device)

            # weighted sum
            drug_embedding, target_embedding = best_architecture(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader, split_mode, drug_sim, target_sim)
            output = best_predictor(drug_idx, target_idx, drug_embedding, target_embedding)

            # fill matrix
            train_adj[drug_idx, target_idx] = output.view(-1).cpu()
    
    # print(np.sum(affinity[train_rows, train_cols] - train_adj[train_rows, train_cols]))

    affinity_graph = getAffinityGraph(dataset, np.array(train_adj), ratio)
    return affinity_graph, train_adj

'''
calculate entropy of degree distribution

param:
    degree_list: degree of each node
return:
    entropy of degree distribution
'''
def entropy(degree_list):
    count = Counter(degree_list)
    distribute = np.array([c/len(degree_list) for c in count.values()])
    distribute = - distribute * np.log10(distribute)
    return np.sum(distribute)

# multiprocess
def core(param):
    affinity, topd, topt, rate_interval = param

    affinity = denseAffinityRefine(affinity.T, topd)
    affinity = denseAffinityRefine(affinity.T, topt)
    affinity_norm = minMaxNormalize(affinity, 0)
    affinity_norm[affinity_norm > 0] = 1

    graph_adj = np.concatenate((
        np.concatenate((np.zeros([affinity.shape[0], affinity.shape[0]]), affinity_norm), 1), 
        np.concatenate((affinity_norm.T, np.zeros([affinity.shape[1], affinity.shape[1]])), 1)
    ), 0)

    rate = np.sum(graph_adj > 0) / (affinity.shape[0]+affinity.shape[1]) / (affinity.shape[0]+affinity.shape[1])

    # skip outside interval
    if rate < rate_interval[0] or rate > rate_interval[1]:
        return None

    degree_d = list(sum(affinity_norm.T))
    degree_t = list(sum(affinity_norm))

    del affinity

    return (topd, topt, rate, entropy(degree_d), entropy(degree_t), entropy(degree_d+degree_t))

'''
search the best TopK through max entropy in ratio interval

param:
    affinity_matrix: complete adjacency matrix
    ratio: edge density ratio
    dataset: data name
return:
    best topK
'''
def search_param(affinity_matrix, ratio, dataset):
    worker = Pool(40)
    print('searching best param ……')
    
    if dataset == 'davis':
        epsilon = 0.001 # interval range
        top_grid = itertools.product(list(range(1, 69, 1)), list(range(1, 443, 1)))
        rate_interval = [ratio - epsilon, ratio + epsilon]
        # rate_interval = [0.03, 0.06]

    if dataset == 'kiba':
        epsilon = 0.0005 # interval range
        top_grid = itertools.product(list(range(1, 2112, 10)), list(range(1, 229, 3)))
        # top_grid = itertools.product(list(range(1, 2112, 15)), list(range(1, 229, 10))) # for debug
        rate_interval = [ratio - epsilon, ratio + epsilon]
    
    # use all edge
    graph_adj = np.concatenate((
        np.concatenate((np.zeros([affinity_matrix.shape[0], affinity_matrix.shape[0]]), affinity_matrix), 1), 
        np.concatenate((affinity_matrix.T, np.zeros([affinity_matrix.shape[1], affinity_matrix.shape[1]])), 1)
    ), 0)

    rate = np.sum(graph_adj > 0) / (affinity_matrix.shape[0]+affinity_matrix.shape[1]) / (affinity_matrix.shape[0]+affinity_matrix.shape[1])
    if ratio == -1 or ratio >= rate:
        top_grid = [list(top_grid)[-1]]
        rate_interval = [0, 1]

    param = []
    # top_grid = [(931, 19), (391, 22), (281, 25), (231, 34), (181, 67)]
    for (topd, topt) in list(top_grid):
        param.append((copy.deepcopy(affinity_matrix), topd, topt, rate_interval))
    
    result = list(filter(None, worker.map(core, param)))
    worker.close()
    
    result = sorted(result, key=lambda x:x[5], reverse=True)

    # for item in result[:10]:
    #     print(f'{item}')
    
    print(f'best param: ({result[0][0]}, {result[0][1]})')
    print('rate: {:.4f}'.format(result[0][2]))
    print('Ed: {:.4f}\tEt: {:.4f}\tEa: {:.4f}'.format(result[0][3], result[0][4], result[0][5]))

    return result[0][0], result[0][1]

def create_test_dataset(dataset, fold, ratio, split_mode='S1', adj=None):
    dataset_path = f'data/{dataset}/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    test_fold = json.load(open(dataset_path + f'{split_mode}_test_index.txt')) if fold == -100 else train_fold_origin[fold]

    rows, cols = np.where(np.isnan(affinity) == False)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_idx=test_rows, target_idx=test_cols, aff=test_Y)

    # create affinity graph
    train_raw_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_raw_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    affinity_graph = DATA.Data(x=None, adj=torch.Tensor(adj), edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("num_node1s", affinity.shape[0])
    affinity_graph.__setitem__("num_node2s", affinity.shape[1])

    return test_dataset, affinity_graph

# data split
def create_train_test_dataset(dataset, fold, ratio, split_mode='S1'):
    dataset_path = f'data/{dataset}/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    train_fold_origin = json.load(open(dataset_path + f'{split_mode}_train_index.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    test_fold = json.load(open(dataset_path + f'{split_mode}_test_index.txt')) if fold == -100 else train_fold_origin[fold]

    rows, cols = np.where(np.isnan(affinity) == False)

    train_rows, train_cols = rows[train_folds], cols[train_folds]
    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_idx=train_rows, target_idx=train_cols, aff=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_idx=test_rows, target_idx=test_cols, aff=test_Y)

    prior_fold = json.load(open(dataset_path + f'{split_mode}_prior_index.txt'))
    prior_rows, prior_cols = rows[prior_fold], cols[prior_fold]
    prior_Y = np.zeros_like(prior_rows) # unknown affinity value
    prior_dataset = DTADataset(drug_idx=prior_rows, target_idx=prior_cols, aff=prior_Y)

    aff = np.zeros_like(affinity)
    aff[train_rows, train_cols] = train_Y # train affinity
    affinity_graph = getAffinityGraph(dataset, aff, ratio) # train graph

    return train_dataset, test_dataset, prior_dataset, affinity_graph, copy.deepcopy(aff)