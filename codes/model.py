import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, DenseGCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
import torch_geometric.data as DATA

class LinearBlock(torch.nn.Module):
    '''
    Construct mlp model

    param:
        linear_layers_dim: # of neurons per layer
        dropout_rate
        relu_layers_index: layer index followed by relu
        dropout_layers_index: layer index followed by dropout
    '''
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        for layer_index in range(len(self.layers)):
            # linear
            x = self.layers[layer_index](x)

            # relu
            if layer_index in self.relu_layers_index:
                x = self.relu(x)

            # dropout
            if layer_index in self.dropout_layers_index:
                x = self.dropout(x)
        return x

class DenseGCNModel(torch.nn.Module):
    '''
    GCN model for affinity graph
    '''
    def __init__(self, layers_dim, edge_dropout_rate):
        super(DenseGCNModel, self).__init__()
        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1

        self.conv_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv_layer_input = layers_dim[i]
            conv_layer = DenseGCNConv(conv_layer_input, layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(0.1)
        self.relu_layers_index = range(self.num_layers)
        self.dropout_layers_index = range(self.num_layers)

    def forward(self, graph):
        x, adj, num_node1s, num_node2s = graph.x, graph.adj, graph.num_node1s, graph.num_node2s
        
        # with dropout
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs], p=self.edge_dropout_rate, force_undirected=True, num_nodes=num_node1s + num_node2s, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout
        
        # # without dropout
        # adj_dropout = adj

        for conv_layer_index in range(len(self.conv_layers)):
            # GCN
            x = self.conv_layers[conv_layer_index](x, adj_dropout, add_loop=False) # without self-loop

            # output = self.conv_layers[conv_layer_index](output, adj, add_loop=True) # with self-loop

            # relu
            if conv_layer_index in self.relu_layers_index:
                x = self.relu(x)

            # dropout
            if conv_layer_index in self.dropout_layers_index:
                x = self.dropout(x)
            
        output = torch.squeeze(x, dim=0)

        return output

class GCNModel(torch.nn.Module):
    '''
    GCN model for molecular graph
    '''
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.conv_layers = torch.nn.ModuleList()

        # molecular GCN * 3
        for i in range(self.num_layers):
            conv_layer_input = layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.relu_layers_index = range(self.num_layers)

    def forward(self, graph_batchs):
        embeddings = []
        for graph in graph_batchs:
            x, edge_index, batch = graph.x, graph.edge_index, graph.batch
            for conv_layer_index in range(len(self.conv_layers)):
                # GCN
                x = self.conv_layers[conv_layer_index](x, edge_index)
                # relu
                if conv_layer_index in self.relu_layers_index:
                    x = self.relu(x)

            embeddings.append(gep(x, batch))

        return torch.cat(embeddings, dim=0)

class GNN(torch.nn.Module):
    '''
    gnn architecture
    param:
        edge_dropout_rate: affinity graph edge dropout rate
        skip: skip connection
        graph_node_dim: affinity graph node input dim
        drug_atom_dim: drug graph node input dim
        target_amino_dim: target graph node input dim
    return:
        drug and target embedding
    '''
    def __init__(self, edge_dropout_rate=0, skip=True, graph_node_dim=256, drug_atom_dim=78, target_amino_dim=54, embedding_dim=128):
        super(GNN, self).__init__()

        # molecular representation learning
        drug_graph_dims = [drug_atom_dim, drug_atom_dim, drug_atom_dim * 2, drug_atom_dim * 4]
        target_graph_dims = [target_amino_dim, target_amino_dim, target_amino_dim * 2, target_amino_dim * 4]
        
        drug_output_dims = [drug_graph_dims[-1], 512, 256]
        target_output_dims = [target_graph_dims[-1], 512, 256]
        
        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)
        
        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        
        # self.drug_output_batchnorm = BatchNorm1d(drug_output_dims[-1])
        # self.target_output_batchnorm = BatchNorm1d(target_output_dims[-1])

        # affinity graph node representation learning
        affinity_graph_dims = [graph_node_dim, 256]
        # affinity_graph_dims = [graph_node_dim, 256, 256, 256] # 在S2上扩大感受野，使得unknown节点有更多卷集邻居
        drug_transform_dims = [affinity_graph_dims[-1], 512, 128]
        target_transform_dims = [affinity_graph_dims[-1], 512, 128]
        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, edge_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        # self.drug_transform_batchnorm = BatchNorm1d(drug_transform_dims[-1])
        # self.target_transform_batchnorm = BatchNorm1d(target_transform_dims[-1])

        self.skip = skip

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, split_mode=None, sim_matrix_d=None, sim_matrix_t=None):
        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        # molecular representation learning
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)

        drug_output_embedding = self.drug_output_linear(drug_graph_embedding)
        target_output_embedding = self.target_output_linear(target_graph_embedding)

        # return drug_output_embedding, target_output_embedding

        # affinity graph node representation learning
        affinity_graph_feature = torch.cat((drug_output_embedding, target_output_embedding), dim=0)
        affinity_graph = DATA.Data(x=affinity_graph_feature, adj=affinity_graph.adj, edge_index=affinity_graph.edge_index)
        affinity_graph.__setitem__("num_node1s", num_node1s)
        affinity_graph.__setitem__("num_node2s", num_node2s)
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)

        drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])
        target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])

        # for unkown node, use weighted similarity matrix get feature
        if split_mode == 'S2':
            drug_sim_output_embedding = torch.mm(sim_matrix_d, drug_output_embedding)
            drug_sim_transform_embedding = torch.mm(sim_matrix_d, drug_transform_embedding)

            # # no sim
            # drug_embedding, target_embedding = torch.cat((drug_output_embedding, drug_transform_embedding), dim=1), \
            #                                     torch.cat((target_output_embedding, target_transform_embedding), dim=1)

            # # davis: only sim
            # drug_embedding, target_embedding = torch.cat((drug_sim_output_embedding, drug_sim_transform_embedding), dim=1), \
            #                                     torch.cat((target_output_embedding, target_transform_embedding), dim=1)

            # kiba: all
            # with skip connection
            drug_embedding, target_embedding = torch.cat(((drug_sim_output_embedding + drug_output_embedding) / 2, (drug_sim_transform_embedding + drug_transform_embedding) / 2), dim=1), \
                                                torch.cat((target_output_embedding, target_transform_embedding), dim=1)

            return drug_embedding, target_embedding
        
        if split_mode == 'S3':
            target_sim_output_embedding = torch.mm(sim_matrix_t, target_output_embedding)
            target_sim_transform_embedding = torch.mm(sim_matrix_t, target_transform_embedding)
            
            # # only sim
            # drug_embedding, target_embedding = torch.cat((drug_output_embedding, drug_transform_embedding), dim=1), \
            #                                     torch.cat((target_sim_output_embedding, target_sim_transform_embedding), dim=1)

            drug_embedding, target_embedding = torch.cat((drug_output_embedding, drug_transform_embedding), dim=1), \
                                                torch.cat(((target_sim_output_embedding + target_output_embedding) / 2, (target_sim_transform_embedding + target_transform_embedding) / 2), dim=1)

            return drug_embedding, target_embedding
        
        if split_mode == 'S4':
            drug_sim_output_embedding = torch.mm(sim_matrix_d, drug_output_embedding)
            drug_sim_transform_embedding = torch.mm(sim_matrix_d, drug_transform_embedding)

            target_sim_output_embedding = torch.mm(sim_matrix_t, target_output_embedding)
            target_sim_transform_embedding = torch.mm(sim_matrix_t, target_transform_embedding)

            # only sim
            # drug_embedding, target_embedding = torch.cat((drug_sim_output_embedding, drug_sim_transform_embedding), dim=1), \
            #                                     torch.cat((target_sim_output_embedding, target_sim_transform_embedding), dim=1)
            
            # no sim
            # drug_embedding, target_embedding = torch.cat((drug_output_embedding, drug_transform_embedding), dim=1), \
            #                                     torch.cat((target_output_embedding, target_transform_embedding), dim=1)

            # all
            drug_embedding, target_embedding = torch.cat(((drug_sim_output_embedding + drug_output_embedding) / 2, (drug_sim_transform_embedding + drug_transform_embedding) / 2), dim=1), \
                                                torch.cat(((target_sim_output_embedding + target_output_embedding) / 2, (target_sim_transform_embedding + target_transform_embedding) / 2), dim=1)
            
            return drug_embedding, target_embedding

        # add skip
        if not self.skip:
            drug_embedding, target_embedding = drug_transform_embedding, target_transform_embedding
        else:
            drug_embedding, target_embedding = torch.cat((drug_output_embedding, drug_transform_embedding), dim=1), torch.cat((target_output_embedding, target_transform_embedding), dim=1)

        return drug_embedding, target_embedding

class Predictor(torch.nn.Module):
    '''
    predictor architecture: prediction affinity from feature
    param:
        embedding_dim: drug_embedding_dim + target_embedding_dim
        output_dim=1
    return:
        affinity
    '''
    def __init__(self, output_dim=1, skip=True):
        super(Predictor, self).__init__()

        if not skip:
            mlp_layers_dim = [128 * 2, 512, 256, output_dim]
        else:
            mlp_layers_dim = [(128+256) * 2, 512, 256, output_dim]
        
        # mlp_layers_dim = [256 * 2, 512, 256, output_dim] # no affinity

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, drug_idx, target_idx, drug_embedding, target_embedding):
        drug_feature = drug_embedding[drug_idx.int().numpy()]
        target_feature = target_embedding[target_idx.int().numpy()]
        concat_feature = torch.cat((drug_feature, target_feature), -1)

        out = self.mlp(concat_feature)

        return out

# import pickle
# import numpy as np
# if __name__=='__main__':
#     affinity = pickle.load(open('/home/wangcheng/project/DTA/MSGraphDTA_fanal/logs/kiba/2022-11-02 17:43:15/refineAdj_epoch996_iter1.pkl', 'rb'))
#     affinity_matrix = np.asarray(affinity)
#     print(affinity)