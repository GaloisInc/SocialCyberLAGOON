import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# torch.autograd.set_detect_anomaly(True) #catch runtime errors such as exploding nan gradients and display where they happened in the forward pass

from lagoon.ml.common import utils
from lagoon.ml.config import *
from data import get_data_toxicity


class PoolingAggregator(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, self_addition=False):
        super().__init__()
        self.net = nn.ModuleList([])
        hidden_sizes = [] if not hidden_sizes else hidden_sizes
        layers = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i],layers[i+1]))
        self.self_addition = self_addition
        
    def forward(self, x_self, x_neighbors):
        """
        x_self has shape (1, input_size)
        After passing through the net, x_self shape becomes (1, output_size)
        
        x_neighbors has shape (num_neighbors, input_size)
        After passing through the net, x_neighbors shape becomes (num_neighbors, output_size)
        max is taken across the num_neighbors dimension to make x_neighbors shape (1, output_size)

        x_self and x_neighbors are then added and shape becomes (1, output_size)
        """
        for layer in self.net:
            x_self = layer(x_self)
            x_self = F.relu(x_self)
            x_neighbors = layer(x_neighbors)
            x_neighbors = F.relu(x_neighbors)
        
        if self.self_addition: # Add the node to its aggregated neighbors, as in original GraphSage code
            x_neighbors = torch.max(x_neighbors, axis=0, keepdim=True)[0]
            x = x_self + x_neighbors

        else: # The node itself may not have useful info, so aggregate it along with its neighbors
            x = torch.max(torch.cat((x_self,x_neighbors),0), axis=0, keepdim=True)[0]
        
        return x


class Embedding(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Linear(input_size,output_size, bias=False)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(x)
        return x


class Regressor(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.net = nn.ModuleList([])
        layers = [input_size] + hidden_sizes + [1]
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i],layers[i+1]))

    def forward(self, x):
        for i,layer in enumerate(self.net):
            x = layer(x)
            if i != len(self.net)-1:
                x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, aggr_1_params, emb_1_params, aggr_2_params, emb_2_params, reg_params):
        super().__init__()
        self.aggr_net_1 = PoolingAggregator(**aggr_1_params)
        self.emb_net_1 = Embedding(**emb_1_params)
        self.aggr_net_2 = PoolingAggregator(**aggr_2_params)
        self.emb_net_2 = Embedding(**emb_2_params)
        self.reg_net = Regressor(**reg_params)

    def forward(self, x_self_all_slice, x_neighbors_all_slice):
        embs_1 = torch.zeros(len(x_self_all_slice),self.emb_net_1.net.out_features, dtype=torch.float32, device=DEVICE) #(num_1st_neighbors+1, embedding_1_size)
        for j in range(len(x_self_all_slice)):
            
            ## Collect 1st embeddings
            aggr = self.aggr_net_1(
                x_self = torch.as_tensor(x_self_all_slice[j], dtype=torch.float32, device=DEVICE).reshape(1,-1), #(1, num_features)
                x_neighbors = torch.as_tensor(x_neighbors_all_slice[j], dtype=torch.float32, device=DEVICE) #(num_2nd_neighbors,num_features)
            ) #(1, num_features)
            emb = self.emb_net_1(aggr) #(1, embedding_1_size)
            norm = torch.linalg.norm(emb)
            if norm!=0:
                emb = emb/norm
            embs_1[j] = emb
        
        ## Get 2nd embeddings and regress
        aggr = self.aggr_net_2(x_self=embs_1[:1], x_neighbors=embs_1[1:]) #(1, embedding_1_size)
        emb = self.emb_net_2(aggr) #(1, embedding_2_size)
        norm = torch.linalg.norm(emb)
        if norm!=0:
            emb = emb/norm
        reg = self.reg_net(emb) #(1,1)
        return reg 


def run_network(data, hyps, verbose=True):
    
    ## Get data
    x_self_all_train = data['x_self_all_train']
    x_self_all_val = data['x_self_all_val']
    x_neighbors_all_train = data['x_neighbors_all_train']
    x_neighbors_all_val = data['x_neighbors_all_val']
    targets_train = data['targets_train']
    targets_val = data['targets_val']

    ## Get hyps
    embedding_sizes = hyps.get('embedding_sizes', [20,30])
    regressor_hidden_sizes = hyps.get('regressor_hidden_sizes', [20])
    aggr_self_addition = hyps.get('aggr_self_addition', False)
    numepochs = hyps.get('numepochs', 10)
    # batch_size = hyps.get('batch_size',100)
    lr = hyps.get('lr', 1e-3)
    weight_decay = hyps.get('weight_decay', 0.)
    gamma = hyps.get('gamma', 0.99)

    ## Create net
    net = Net(
        aggr_1_params = {'input_size':len(x_self_all_train[0][0]), 'output_size':len(x_self_all_train[0][0]), 'self_addition':aggr_self_addition},
        emb_1_params = {'input_size':len(x_self_all_train[0][0]), 'output_size':embedding_sizes[0]},
        aggr_2_params = {'input_size':embedding_sizes[0], 'output_size':embedding_sizes[0], 'self_addition':aggr_self_addition},
        emb_2_params = {'input_size':embedding_sizes[0], 'output_size':embedding_sizes[1]},
        reg_params = {'input_size':embedding_sizes[1], 'hidden_sizes':regressor_hidden_sizes}
    )
    net.to(DEVICE)

    ## Create optimizer, etc
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    ## Record stats
    stats = {'epoch':[], 'train_loss':[], 'val_loss':[]}

    ## Epochs
    for ep in range(numepochs):
        stats['epoch'].append(ep+1)
        message = f"Epoch {ep+1}"

        ## Shuffle
        shuff = np.random.permutation(len(x_self_all_train))
        x_self_all_train = [x_self_all_train[s] for s in shuff]
        x_neighbors_all_train = [x_neighbors_all_train[s] for s in shuff]
        targets_train = [targets_train[s] for s in shuff]
        
        ## Train
        train_loss = 0.
        net.train()
        targets_train_tensor = torch.as_tensor(targets_train, dtype=torch.float32, device=DEVICE)
        for i in tqdm(range(len(x_self_all_train))):
            opt.zero_grad()
            out = net(x_self_all_slice=x_self_all_train[i], x_neighbors_all_slice=x_neighbors_all_train[i])
            loss = nn.L1Loss()(out.flatten(), targets_train_tensor[i:i+1]) #targets_tensor[i:i+1] is better than targets_tensor[i] since tensors become exactly the same shape = (1,)
            train_loss += loss.item()
            loss.backward()
            opt.step()
        scheduler.step()

        train_loss /= len(x_self_all_train)
        stats['train_loss'].append(train_loss)
        message += f", average train loss = {train_loss}"
        
        ## Validate
        val_loss = 0.
        net.eval()
        with torch.no_grad():
            targets_val_tensor = torch.as_tensor(targets_val, dtype=torch.float32, device=DEVICE)
            for i in tqdm(range(len(x_self_all_val))):
                out = net(x_self_all_slice=x_self_all_val[i], x_neighbors_all_slice=x_neighbors_all_val[i])
                loss = nn.L1Loss()(out.flatten(), targets_val_tensor[i:i+1]) #targets_tensor[i:i+1] is better than targets_tensor[i] since tensors become exactly the same shape = (1,)
                val_loss += loss.item()
        val_loss /= len(x_self_all_val)
        stats['val_loss'].append(val_loss)
        message += f", val loss = {val_loss}"
            
        ## Verbose
        if verbose:
            print(message)

    return net, stats


def run_network_wrapper():
    _, stats = run_network(
        data = get_data_toxicity(
            target_type='activity',
            start_year=2001,
            split=0.7,
            scaling='log',
            remove_all_zero_samples=True
        ),
        hyps = {
            'embedding_sizes': [20,20],
            'regressor_hidden_sizes': [20],
            'aggr_self_addition': False,
            'numepochs': 20,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'gamma': 0.99
        },
        verbose = True
    )

    ## Plot (comment out if not plotting)
    utils.plot_stats(stats=stats, foldername = os.path.join(RESULTS_FOLDER, 'graphsage'))


if __name__ == "__main__":
    run_network_wrapper()
