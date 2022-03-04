import os
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.common import utils, targets
from lagoon.ml.config import *


def save_data_toxicity(start, end):
    """
    For a window, save the following:
        x_self_all:
            Contains features of all persons and 1st hop neighbors
            Shape = (num_persons, num_1st_neighbors+1 (variable), num_features)
            x_self_all[i][j] are toxicity features of jth neighbor of ith person
        x_neighbors_all:
            Contains features of all 2nd hop neighbors
            Shape = (num_persons, num_1st_neighbors+1 (variable), num_2nd_neighbors (variable), num_features)
            x_neighbors_all[i][j][k] are toxicity features of kth neighbor of jth neighbor of ith person
        ids:
            Contains ids of all persons
            Shape = (num_persons)
    """
    x_self_all = []
    x_neighbors_all = []
    ids = []
    
    with get_session() as sess:
        print('Getting persons in window...')
        entity_ids = utils.get_entities_in_window(sess, start=start, end=end)

        # Restrict persons
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(entity_ids)).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
        
        print('Creating data...')
        
        # NOTE: For this data algorithm, neighbors of a node refers to strictly the neighbors, i.e. do not count the node itself. This is because the node itself is counted separately in x_self and the strict neighbors in x_neighbors
        
        ## For each person
        for person in tqdm(persons.all()):
            x_self_interm = []
            x_neighbors_interm = []

            ## Get neighbors
            person_neighbors = utils.get_neighboring_entities(sess, person, hop=1, start=start, end=end, return_self=False)
            
            ## For each node in current person + its neighbors
            for node in [person]+person_neighbors.all():
                x_self = [node.attrs.get(key,0) for key in TOXICITY_CATEGORIES]
                
                ## Get neighbors and their features
                neighbors = utils.get_neighboring_entities(sess, node, hop=1, start=start, end=end, return_self=False)
                x_neighbors = []
                for neighbor in neighbors:
                    x_neighbors.append([neighbor.attrs.get(key,0) for key in TOXICITY_CATEGORIES])
                
                ## Append
                x_self_interm.append(x_self)
                x_neighbors_interm.append(x_neighbors)
        
            ## Append
            x_self_all.append(x_self_interm)
            x_neighbors_all.append(x_neighbors_interm)
            ids.append(person.id)
        
    ## Save
    foldername = os.path.join(DATA_FOLDER, 'data_toxicity')
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, f'data_toxicity_{start}_{end}.pkl'), 'wb') as f:
        pickle.dump(
            {
                'x_self_all': x_self_all, #shape = (num_persons, num_1st_neighbors+1 (variable), num_features)
                'x_neighbors_all': x_neighbors_all, #shape = (num_persons, num_1st_neighbors+1 (variable), num_2nd_neighbors (variable), num_features)
                'ids': ids #shape = (num_persons)
            },
            f
        )


def save_data_toxicity_wrapper(start_years=range(1998,2020)):
    for start_year in start_years:
        print(start_year)
        start=f'{start_year}-01-01'
        end=f'{start_year+1}-12-31'
        save_data_toxicity(start=start, end=end)


def get_data_toxicity(target_type, start_year, split=0.7, scaling='log', remove_all_zero_samples=True):
    """
    target_type: 'gaps' or 'activity'
    """
    # features
    with open(os.path.join(os.path.join(DATA_FOLDER, 'data_toxicity'), f'data_toxicity_{start_year}-01-01_{start_year+1}-12-31.pkl'), 'rb') as f:
        data = pickle.load(f)
    x_self_all, x_neighbors_all, ids = data['x_self_all'], data['x_neighbors_all'], data['ids']

    # remove samples which are all 0
    if remove_all_zero_samples:
        num_persons = len(x_self_all)
        num_features = len(x_self_all[0][0])
        
        self_keep_flags = num_persons*[False]
        for i in range(len(x_self_all)):
            for j in range(len(x_self_all[i])):
                if not np.array_equal(x_self_all[i][j], np.zeros(num_features)):
                    self_keep_flags[i] = True
                    break
        
        neighbors_keep_flags = num_persons*[False]
        for i in range(len(x_neighbors_all)):
            if self_keep_flags[i]:
                continue
            for j in range(len(x_neighbors_all[i])):
                for k in range(len(x_neighbors_all[i][j])):
                    if not np.array_equal(x_neighbors_all[i][j][k], np.zeros(num_features)):
                        neighbors_keep_flags[i] = True
                        break
                if neighbors_keep_flags[i]:
                    break

        keep_indices = [i for i in range(num_persons) if self_keep_flags[i] or neighbors_keep_flags[i]]
        x_self_all = [x_self_all[i] for i in keep_indices]
        x_neighbors_all = [x_neighbors_all[i] for i in keep_indices]
        ids = [ids[i] for i in keep_indices]

    # targets
    targs = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'targets'), 'persons_activity_yearly.csv' if target_type=='activity' else 'persons_183daygaps_yearly.csv'), index_col='id')
    if target_type=='activity':
        targs['target'] = targs.apply(lambda row: targets.get_persons_activity_target(row, start_year), axis=1)
    elif target_type=='gaps':
        targs['target'] = targs.apply(lambda row: targets.get_persons_gaps_target(row, start_year), axis=1)

    # join
    targs_final = []
    for id_ in ids:
        targs_final.append(targs['target'].loc[id_])

    ## Shuffle
    shuff = np.random.permutation(len(x_self_all))
    x_self_all = [x_self_all[s] for s in shuff]
    x_neighbors_all = [x_neighbors_all[s] for s in shuff]
    targs_final = [targs_final[s] for s in shuff]

    ## Apply log scaling
    if scaling=='log':
        for i in range(len(x_self_all)):
            for j in range(len(x_self_all[i])):
                x_self_all[i][j] = list(np.log10(1+np.asarray(x_self_all[i][j])))
        for i in range(len(x_neighbors_all)):
            for j in range(len(x_neighbors_all[i])):
                for k in range(len(x_neighbors_all[i][j])):
                    x_neighbors_all[i][j][k] = list(np.log10(1+np.asarray(x_neighbors_all[i][j][k])))
    # NOTE: minmax scaling is too cumbersome since the feature tensors are not of regular shape
    # TODO: gaussian scaling, where we ignore features and instead find (mu,sigma) from the completely flattened data and then scale each value. This would also require splitting into train and val since (mu,sigma) should be calculated on train.
    # If applying minmax or gaussian, scaling should go below train-val split)
    
    ## Split into train and val
    split_point = int(split*len(x_self_all))
    x_self_all_train = x_self_all[:split_point]
    x_self_all_val = x_self_all[split_point:]
    x_neighbors_all_train = x_neighbors_all[:split_point]
    x_neighbors_all_val = x_neighbors_all[split_point:]
    targets_train = targs_final[:split_point]
    targets_val = targs_final[split_point:]

    ## Return
    return {
        'x_self_all_train': x_self_all_train,
        'x_self_all_val': x_self_all_val,
        'x_neighbors_all_train': x_neighbors_all_train,
        'x_neighbors_all_val': x_neighbors_all_val,
        'targets_train': targets_train,
        'targets_val': targets_val
    }


if __name__ == "__main__":
    pass