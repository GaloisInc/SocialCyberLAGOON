import os
import csv
import itertools
from tqdm import tqdm
import shortuuid
from ast import literal_eval
from typing import Tuple, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.nn import functional as F
# torch.autograd.set_detect_anomaly(True) #catch runtime errors such as exploding nan gradients and display where they happened in the forward pass

from lagoon.ml.common import utils
from lagoon.ml.config import *
from data import get_persons_toxicity


class Net(nn.Module):
    def __init__(self, input_sizes: Tuple[int,int], gcn_embedding_sizes: Tuple[int,int], mlp_hidden_sizes: List[int], output_size: int = 1):
        """
        Inputs:
            input_sizes is number of features in 1st hop and 2nd hop, respectively.
            Each gcn_embedding is a single-layer MLP. There must be 2 embeddings – the first is for the 1st hop features, and the 2nd for the 2nd hop features.
            mlp_hidden_sizes can be any number of layers.
            output_size is the final number of output neurons at the end of the MLP.
        Example:
            input_sizes = [8,8], gcn_embedding_sizes = [100,200], mlp_hidden_sizes = [50,20], output_size = 1
            The network looks as follows:
            8 -- 100 --
                        \
                        (concat) -- 50 -- 20 -- 1
                        /
            8 -- 200 --
        """
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


def run_network(data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor, Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]], hyps: Dict[str,Any], verbose: bool = True) -> Tuple[Net, Dict[str,Any]]:
    """
    Inputs:
        data: Tuple of (x1tr,x2tr,ytr, x1va,x2va,yva)
            If validation is not desired, pass (x1tr,x2tr,ytr, None,None,None)
        hyps: Most hyperparameters are self-explanatory
            lossfunc:
                Either 'L1' or 'L2'
            early_stopping:
                If False, no early stopping. The net will run for the complete `numepochs` epochs
                If an integer, early stop if val perf doesn't improve for that many epochs
                If a float, early stop if val perf doesn't improve for that fraction of epochs, rounded up
    
    Returns:
        Trained Pytorch net
        Dictionary of stats like train and val losses
    """
    ## Get data
    x1tr,x2tr,ytr, x1va,x2va,yva = data

    ## Get hyps
    gcn_embedding_size_1 = hyps.get('gcn_embedding_size_1',50)
    gcn_embedding_size_2 = hyps.get('gcn_embedding_size_2',100)
    mlp_hidden_sizes = hyps.get('mlp_hidden_sizes',[100])
    numepochs = hyps.get('numepochs',100)
    batch_size = hyps.get('batch_size',100) #0 corresponds to batch_size = num_inputs
    lr = hyps.get('lr',1e-3)
    weight_decay = hyps.get('weight_decay',0.)
    gamma = hyps.get('gamma',1.)
    lossfunc = hyps.get('lossfunc','L2')
    early_stopping = hyps.get('early_stopping',False)

    ## Get batch info
    if batch_size <= 0 or batch_size > x1tr.shape[0]:
        batch_size = x1tr.shape[0]
        numbatches = 1
    else:
        numbatches = int(np.ceil(x1tr.shape[0]/batch_size))

    ## Create net
    net = Net(
        input_sizes = (x1tr.shape[1],x2tr.shape[1]),
        gcn_embedding_sizes = (gcn_embedding_size_1,gcn_embedding_size_2),
        mlp_hidden_sizes=mlp_hidden_sizes
    )
    net.to(DEVICE)

    ## Create optimizer, etc
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    ## Record stats
    stats = {'train_loss':[], 'val_loss':[]}
    
    ## Early stopping logic
    if type(early_stopping)==float:
        early_stopping = int(np.ceil(early_stopping*numepochs))
    no_improvement_epochs_count = 0
    best_val_loss = np.inf
    early_stopping_flag = False

    ## Epochs
    for ep in range(numepochs):
        message = f"Epoch {ep+1}"

        ## Shuffle
        shuff = torch.randperm(x1tr.shape[0], device=DEVICE) #do not set dtype=torch.float32 since a permutation tensor needs to be of type int, which is the default
        x1tr, x2tr, ytr = x1tr[shuff], x2tr[shuff], ytr[shuff]
        
        ## Train
        train_loss = 0.
        net.train()
        for batch in range(numbatches):
            opt.zero_grad()
            out = net(x1tr[batch*batch_size : (batch+1)*batch_size], x2tr[batch*batch_size : (batch+1)*batch_size])
            loss = LOSSFUNC_MAPPING[lossfunc](out.flatten(), ytr[batch*batch_size : (batch+1)*batch_size])
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
                loss = LOSSFUNC_MAPPING[lossfunc](out.flatten(), yva)
            val_loss = loss.item()
            stats['val_loss'].append(val_loss)
            message += f", val loss = {val_loss}"
            
            ## Early stopping logic
            if early_stopping:
                if val_loss < best_val_loss:
                    no_improvement_epochs_count = 0
                    best_val_loss = val_loss
                else:
                    no_improvement_epochs_count += 1
                if no_improvement_epochs_count == early_stopping:
                    early_stopping_flag = True
                    message += f"\nEarly stopped due to no improvement for {early_stopping} epochs."
            
        ## Verbose
        if verbose:
            print(message)

        ## Early stopping
        if early_stopping_flag:
            break

    ## Final stats
    if verbose and x1va is not None and x2va is not None and yva is not None:
        print(f"Best validation loss = {np.min(stats['val_loss'])} obtained on epoch {np.argmin(stats['val_loss'])+1}.")
    
    return net, stats


def run_network_wrapper() -> None:
    """
    Wrapper for run_network()
    Can be used to plot stats
    """
    data = get_persons_toxicity(
        target_type='activity',
        start_year=2001,
        splits=(0.7,1.0),
        scaling='log',
        remove_all_zero_samples=True
    )
    lossfunc = 'L2'
    _, stats = run_network(
        data = (
            torch.as_tensor(data['x1tr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x2tr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['ytr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x1va'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x2va'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['yva'], dtype=torch.float32, device=DEVICE)
        ),
        hyps = {
            'gcn_embedding_sizes': (100,200),
            'mlp_hidden_sizes': [20],
            'numepochs': 100,
            'batch_size': 20,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'gamma': 0.99,
            'lossfunc': lossfunc,
            'early_stopping': 0.3
        },
        verbose=True
    )

    ## Compare to naive
    naive_val_loss = utils.get_naive_performance(data['yva'], lossfunc).item()
    print(f"Best validation loss is {(naive_val_loss-np.min(stats['val_loss']))/naive_val_loss*100}% better than naive validation loss = {naive_val_loss}.")

    ## Plot (comment out if not plotting)
    utils.plot_stats(stats=stats, foldername=os.path.join(RESULTS_FOLDER,'gmlp'), naive_val_loss=naive_val_loss)


def hyp_search(hyp_ranges: Dict[str,List[Any]], data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor, Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]], output_csv_path: str, numruns: int = 100, sort: bool = True) -> None:
    """
    Perform hyperparameter search.
    
    hyp_ranges : Dictionary with keys = hyperparameter names, and values = lists of their values to be explored.
    data : As required by run_network().
    numruns : If total number of hyperparameter configs (= product of len of lists in hyp_ranges) is greater than `numruns`, sample `numruns` configs. Otherwise, run for all configs.
    output_csv_path : Save the results here.
    sort : If True, the final csv file in output_csv_path is sorted according to best_val_loss
    """
    if 'lossfunc' in hyp_ranges:
        assert len(hyp_ranges['lossfunc']) == 1, "'lossfunc' cannot have multiple options, since that would change the loss value scale"
    #NOTE: Then why have 'lossfunc' as a key at all? That's because it is one of the hyps passed to run_network(), so it's just convenient to have it as a key with a single option

    hyps_list_all = list(itertools.product(*[hyp_ranges[key] for key in hyp_ranges.keys()]))
    try:
        hyps_list_all = random.sample(hyps_list_all, numruns)
    except ValueError:
        pass
    
    with open(output_csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(list(hyp_ranges.keys()) + ['final_train_loss', 'best_val_loss', 'best_val_ep'])
        
        for hyps_list in tqdm(hyps_list_all):
            hyps = dict(zip(hyp_ranges.keys(),hyps_list))
            _, stats = run_network(data=data, hyps=hyps, verbose=False)
            csvwriter.writerow([hyps[key] for key in hyps.keys()] + [stats['train_loss'][-1], np.min(stats['val_loss']), np.argmin(stats['val_loss'])+1])

    if sort:
        df = pd.read_csv(output_csv_path)
        df.sort_values('best_val_loss', ascending=True, inplace=True)
        df.to_csv(output_csv_path, index=False)


def hyp_search_wrapper() -> None:
    """
    Wrapper to run hyp_search over various input combinations
    """
    hyp_ranges = {
        'gcn_embedding_size_1': [20,50,100,200], #4
        'gcn_embedding_size_2': [20,50,100,200], #4
        'mlp_hidden_sizes': [[20],[50],[100],[200]], #4
        'numepochs': [100],
        'batch_size': [20,50,100], #3
        'lr': [1e-3],
        'weight_decay': [0.,1e-5,1e-4], #3
        'gamma': [0.99,0.999], #2
        'lossfunc': ['L2'],
        'early_stopping': [0.3]
    } #total = 1152
    
    data = get_persons_toxicity(
        target_type='activity',
        start_year=2001,
        splits=(0.7,1.0),
        scaling='log',
        remove_all_zero_samples=True
    )

    uuid = shortuuid.uuid()
    print(f'Saving results in {uuid}.csv ...')
    foldername = os.path.join(RESULTS_FOLDER, 'gmlp')
    os.makedirs(foldername, exist_ok=True)
    output_csv_path = os.path.join(foldername, f'{uuid}.csv')

    hyp_search(
        hyp_ranges = hyp_ranges,
        data = (
            torch.as_tensor(data['x1tr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x2tr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['ytr'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x1va'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['x2va'], dtype=torch.float32, device=DEVICE),
            torch.as_tensor(data['yva'], dtype=torch.float32, device=DEVICE)
        ),
        output_csv_path = output_csv_path,
        numruns = 100
    )


def run_network_test(net: Net, data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor], lossfunc: str = 'L2') -> float:
    """
    Run a pretrained `net` on test data, and return test loss computed using `lossfunc`
    data: Tuple (x1te,x2te,yte)
    """
    x1te,x2te,yte = data
    net.eval()
    with torch.no_grad():
        out = net(x1te, x2te)
        test_loss = LOSSFUNC_MAPPING[lossfunc](out.flatten(), yte).item()
    print(f"Test loss = {test_loss}")

    ## Compare to naive
    naive_test_loss = utils.get_naive_performance(yte, lossfunc).item()
    print(f"Test loss is {(naive_test_loss-test_loss)/naive_test_loss*100}% better than naive test loss = {naive_test_loss}.")

    return test_loss


def run_expt(repeats: int = 100, numruns: int = 200) -> None:
    """
    Repeat `repeats` times:
        Get a train-val-test split of data
        Perform hyperparameter search for `numruns` runs using train and val
        Pick the best net and find test performance
    
    --> The idea is to get a good estimate of the performance achievable on the data, independent of the way it is split. This is because for datasets in Social Cyber, performance is quite sensitive to the way the data is split. <--
    """
    hyp_ranges = {
        'gcn_embedding_size_1': [20,50,100], #3
        'gcn_embedding_size_2': [100,200,300], #3
        'mlp_hidden_sizes': [[50],[100],[200]], #3
        'numepochs': [100],
        'batch_size': [20,50,100], #3
        'lr': [1e-3],
        'weight_decay': [0.,1e-5,1e-4], #3
        'gamma': [0.99],
        'lossfunc': ['L2'],
        'early_stopping': [0.5]
    } #total = 243

    foldername = os.path.join(RESULTS_FOLDER, 'gmlp')
    os.makedirs(foldername, exist_ok=True)
    
    prefix = "20211022_hyp_search"
    print(f'Saving hyp search results with prefix {prefix}')
    hyp_search_csv_filename = os.path.join(foldername, prefix)

    results_csv_path = os.path.join(foldername, "20211022_results.csv")
    print(f'Saving final results as {results_csv_path}')
    with open(results_csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(list(hyp_ranges.keys()) + ['final_train_loss', 'best_val_loss', 'best_val_ep', 'test_loss', 'train_pct_improv_naive', 'val_pct_improv_naive', 'test_pct_improv_naive'])
    
        for repeat in range(repeats): 
            ## Get data
            data = get_persons_toxicity(
                target_type='activity',
                start_year=2001,
                splits=(0.6,0.8),
                scaling='log',
                remove_all_zero_samples=True
            )
            
            ## Do hyp search
            output_csv_path = f'{hyp_search_csv_filename}_{repeat+1}.csv'
            hyp_search(
                hyp_ranges = hyp_ranges,
                data = (
                    torch.as_tensor(data['x1tr'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['x2tr'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['ytr'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['x1va'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['x2va'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['yva'], dtype=torch.float32, device=DEVICE)
                ),
                output_csv_path = output_csv_path,
                numruns = numruns,
                sort = True
            )

            ## Train best net
            hyp_search_df = pd.read_csv(output_csv_path)
            best_hyps = dict(hyp_search_df.iloc[0]) #since the df is sorted, this is the best config
            net, _ = run_network(
                data = (
                    torch.as_tensor(np.concatenate((data['x1tr'],data['x1va']), axis=0), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.concatenate((data['x2tr'],data['x2va']), axis=0), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.concatenate((data['ytr'],data['yva']), axis=0), dtype=torch.float32, device=DEVICE),
                    None,
                    None,
                    None
                ), # combine train and val for the final best net
                hyps = {
                    'gcn_embedding_size_1': best_hyps['gcn_embedding_size_1'],
                    'gcn_embedding_size_2': best_hyps['gcn_embedding_size_2'],
                    'mlp_hidden_sizes': literal_eval(best_hyps['mlp_hidden_sizes']),
                    'numepochs': best_hyps['best_val_ep'], # Since batch size is kept same, training on train+val means more batches, hence more updates. So there is no point in increasing numepochs beyond best_val_ep.
                    'batch_size': best_hyps['batch_size'],
                    'lr': best_hyps['lr'],
                    'weight_decay': best_hyps['weight_decay'],
                    'gamma': best_hyps['gamma'],
                    'lossfunc': best_hyps['lossfunc']
                    # obviously no early_stopping here
                },
                verbose=False
            )

            ## Test best net
            test_loss = run_network_test(
                net = net,
                data = (
                    torch.as_tensor(data['x1te'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['x2te'], dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(data['yte'], dtype=torch.float32, device=DEVICE)
                ),
                lossfunc = best_hyps['lossfunc']
            )

            ## Get naive performances
            naive_train_loss = utils.get_naive_performance(data['ytr'], best_hyps['lossfunc']).item()
            naive_val_loss = utils.get_naive_performance(data['yva'], best_hyps['lossfunc']).item()
            naive_test_loss = utils.get_naive_performance(data['yte'], best_hyps['lossfunc']).item()

            ## Write final results to csv
            csvwriter.writerow([best_hyps[key] for key in best_hyps.keys()] + [
                test_loss,
                (naive_train_loss-best_hyps['final_train_loss'])/naive_train_loss*100,
                (naive_val_loss-best_hyps['best_val_loss'])/naive_val_loss*100,
                (naive_test_loss-test_loss)/naive_test_loss*100
            ])


if __name__ == "__main__":
    run_expt(repeats=100, numruns=300)
