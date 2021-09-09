import os
import csv
import itertools
from tqdm import tqdm
import shortuuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# torch.autograd.set_detect_anomaly(True) #catch runtime errors such as exploding nan gradients and display where they happened in the forward pass

from lagoon.ml.common import utils
from lagoon.ml.config import *
from data import get_persons_toxicity


class Net(nn.Module):
    def __init__(self, input_sizes, gcn_embedding_sizes, mlp_hidden_sizes, output_size=1):
        super().__init__()
        
        # GCN
        self.gcn = nn.ModuleList([])
        for input_size,gcn_embedding_size in zip(input_sizes,gcn_embedding_sizes):
            self.gcn.append(nn.Linear(input_size,gcn_embedding_size))

        # MLP
        layers = [sum(gcn_embedding_sizes)] + mlp_hidden_sizes + [output_size]
        self.mlp = nn.ModuleList([])
        for i in range(len(layers)-1):
            self.mlp.append(nn.Linear(layers[i],layers[i+1]))
        
    def forward(self,x1,x2):
        
        # GCN
        out1 = self.gcn[0](x1)
        out1 = F.relu(out1)
        out2 = self.gcn[1](x2)
        out2 = F.relu(out2)
        out = torch.cat((out1,out2),1)

        # MLP
        for i,layer in enumerate(self.mlp):
            out = layer(out)
            if i != len(self.mlp)-1:
                out = F.relu(out)
        
        return out


def run_network(data, hyps, verbose=True):
    
    ## Get data and convert to torch tensors
    x1tr = torch.as_tensor(data['x1tr'], dtype=torch.float32, device=DEVICE)
    x2tr = torch.as_tensor(data['x2tr'], dtype=torch.float32, device=DEVICE)
    ytr = torch.as_tensor(data['ytr'], dtype=torch.float32, device=DEVICE)
    x1va = torch.as_tensor(data['x1va'], dtype=torch.float32, device=DEVICE)
    x2va = torch.as_tensor(data['x2va'], dtype=torch.float32, device=DEVICE)
    yva = torch.as_tensor(data['yva'], dtype=torch.float32, device=DEVICE)

    ## Get hyps
    gcn_embedding_sizes = hyps.get('gcn_embedding_sizes',[10,10])
    mlp_hidden_sizes = hyps.get('mlp_hidden_sizes',[20])
    numepochs = hyps.get('numepochs',100)
    batch_size = hyps.get('batch_size',100) #0 corresponds to batch_size = num_inputs
    lr = hyps.get('lr',1e-3)
    weight_decay = hyps.get('weight_decay',0.)
    gamma = hyps.get('gamma',1.)

    ## Get batch info
    if batch_size <= 0 or batch_size > x1tr.shape[0]:
        batch_size = x1tr.shape[0]
        numbatches = 1
    else:
        numbatches = int(np.ceil(x1tr.shape[0]/batch_size))

    ## Create net
    net = Net(
        input_sizes = [x1tr.shape[1],x2tr.shape[1]],
        gcn_embedding_sizes = gcn_embedding_sizes,
        mlp_hidden_sizes=mlp_hidden_sizes
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
        shuff = torch.randperm(x1tr.shape[0])		
        x1tr, x2tr, ytr = x1tr[shuff], x2tr[shuff], ytr[shuff]
        
        ## Train
        train_loss = 0.
        net.train()
        for batch in range(numbatches):
            opt.zero_grad()
            out = net(x1tr[batch*batch_size : (batch+1)*batch_size], x2tr[batch*batch_size : (batch+1)*batch_size])
            loss = nn.L1Loss()(out.flatten(), ytr[batch*batch_size : (batch+1)*batch_size])
            train_loss += loss.item()
            loss.backward()
            opt.step()
        scheduler.step()
        
        train_loss /= numbatches
        stats['train_loss'].append(train_loss)
        message += f", train loss = {train_loss}"
        
        ## Validate
        if x1va is not None and x2va is not None and yva is not None:
            net.eval()
            with torch.no_grad():
                out = net(x1va, x2va)
                val_loss = nn.L1Loss()(out.flatten(), yva)
            val_loss = val_loss.item()
            stats['val_loss'].append(val_loss)
            message += f", val loss = {val_loss}"
            
        ## Verbose
        if verbose:
            print(message)

    return net, stats


def run_network_wrapper():
    _, stats = run_network(
        data = get_persons_toxicity(
            target_type='activity',
            start_year=2001,
            split=0.7,
            scaling='log',
            remove_all_zero_samples=True
        ),
        hyps = {
            'gcn_embedding_sizes': [50,50],
            'mlp_hidden_sizes': [50],
            'numepochs': 300,
            'batch_size': 100,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'gamma': 0.99
        },
        verbose=True
    )

    ## Plot (comment out if not plotting)
    utils.plot_stats(stats=stats, foldername = os.path.join(RESULTS_FOLDER, 'gmlp'))


def run_network_hyp_search():
    gcn_embedding_size_1_all = [10,20,50,100]
    gcn_embedding_size_2_all = [10,20,50,100]
    mlp_hidden_size_all = [10,20,50,100]
    numepochs_all = [25,50,100,200]
    batch_size_all = [10,20,50,100]
    weight_decay_all = [0.,3e-5,1e-5,3e-5,1e-4]
    gamma_all = [0.9,0.99,0.999]

    options = list(itertools.product(gcn_embedding_size_1_all, gcn_embedding_size_2_all, mlp_hidden_size_all, numepochs_all, batch_size_all, weight_decay_all, gamma_all))
    shuff = np.random.permutation(len(options))
    options = [options[s] for s in shuff]

    data = get_persons_toxicity(
        target_type='activity',
        start_year=2001,
        split=0.7,
        scaling='log',
        remove_all_zero_samples=True
    )

    uuid = shortuuid.uuid()
    print(f'Saving results in {uuid}.csv ...')
    foldername = os.path.join(RESULTS_FOLDER, 'gmlp')
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, f'{uuid}.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['gcn_embedding_size_1', 'gcn_embedding_size_2', 'mlp_hidden_size', 'numepochs', 'batch_size', 'weight_decay', 'gamma', 'final_train_loss', 'best_val_loss', 'best_val_ep'])
        
        for option in tqdm(options[:2000]):
            gcn_embedding_size_1, gcn_embedding_size_2, mlp_hidden_size, numepochs, batch_size, weight_decay, gamma = option
            _, stats = run_network(
                data = data,
                hyps = {
                    'gcn_embedding_sizes': [gcn_embedding_size_1,gcn_embedding_size_2],
                    'mlp_hidden_sizes': [mlp_hidden_size],
                    'numepochs': numepochs,
                    'batch_size': batch_size,
                    'lr': 1e-3,
                    'weight_decay': weight_decay,
                    'gamma': gamma
                },
                verbose=False
            )
            csvwriter.writerow([*option, stats['train_loss'][-1], np.min(stats['val_loss']), np.argmin(stats['val_loss'])+1])


if __name__ == "__main__":
    utils.set_seed(15)
    run_network_wrapper()
