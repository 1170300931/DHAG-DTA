import torch
from torch.utils.data import Dataset
from torch_geometric import data as DATA

class DTADataset(Dataset):
    '''
    Dataset for train and test.

    Param: 
        drug_idx: train or test drug index.
        drug_idx: train or test target index.
        aff: train or test affinity value.        
    '''
    def __init__(self, drug_idx, target_idx, aff):
        super(DTADataset, self).__init__()
        self.drug_idx, self.target_idx, self.aff = drug_idx, target_idx, aff

    def __len__(self):
        return len(self.aff)

    def __getitem__(self, idx):
        return self.drug_idx[idx], self.target_idx[idx], self.aff[idx]

class GraphDataset(Dataset):
    '''
    Dataset for all graph data, align order with index in DTADataset.

    Param: 
        graph_dict: graph dictionary with some attributes, size, features, edge_index
    '''
    def __init__(self, graphs_dict):
        super(GraphDataset, self).__init__()
        self.data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            self.data_list.append(GCNData)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]