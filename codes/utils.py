import torch
from torch import nn

def train(gnn, predictor, optimizer, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, epoch, batch_size, affinity_graph, writer, iter_num):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    gnn.train()
    predictor.train()
    log_interval = 30
    loss_fn = nn.MSELoss()
    
    affinity_graph = affinity_graph.to(device)
    loss_all = []

    for batch_idx, data in enumerate(train_loader):
        drug_idx, target_idx, aff = data
        aff = aff.to(device)

        optimizer.zero_grad()
        drug_embedding, target_embedding = gnn(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader)
        output = predictor(drug_idx, target_idx, drug_embedding, target_embedding)
        loss = loss_fn(output, aff.view(-1, 1).float())
        loss.backward()
        loss_all.append(loss.item())

        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, 
                                                                        len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                                                                        loss.item()))

    writer.add_scalars('train/mse', {f'{iter_num}':sum(loss_all)/len(loss_all)}, epoch)
    return sum(loss_all)/len(loss_all)

# predict test data
def infer(gnn, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_sim, target_sim, split_mode):
    gnn.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))

    affinity_graph =  affinity_graph.to(device)

    if drug_sim is not None and target_sim is not None:
        drug_sim, target_sim = torch.Tensor(drug_sim).to(device), torch.Tensor(target_sim).to(device)

    with torch.no_grad():
        for data in loader:
            drug_idx, target_idx, aff = data
            aff = aff.to(device)

            if split_mode == 'S1':
                drug_embedding, target_embedding = gnn(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader)
            else:
                # weighted sum
                drug_embedding, target_embedding = gnn(affinity_graph, drug_graphs_DataLoader, target_graphs_DataLoader, split_mode, drug_sim, target_sim)

            output = predictor(drug_idx, target_idx, drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, aff.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()