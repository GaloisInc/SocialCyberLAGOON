import arrow
import os
from tqdm import tqdm

import pandas as pd
import numpy as np

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.common import utils, targets
from lagoon.ml.config import *


def aggregate_attrs_of_neighbors(entity, attr_keys, start, end, hops):
    """
    NOTE 1:
        This method actually aggregates attributes from self AND neighbors
        E.g.: If a node has id=10 and its 5 neighbors have ids 11-15, this method will aggregate attributes from nodes 10-15
        This is beneficial since we want self information to be also considered in the final features
    NOTE 2:
        When aggregating 2nd hop nodes, all nodes in 1st hop and self are also considered
        E.g.: If a node has id=10, its 2 neighbors have ids 11-12, and their neighbors are respectively [10,21,22] and [10,31,32,33], this method will aggregate attributes from nodes [10,11,12,21,22,31,32,33]
        If we do not want this behavior, we need to manintain separate `neighbor_ids` variables for each hop as a set and perform set difference
    """
    attrs_aggr_all = {}
    
    with get_session() as sess:
        for hop in hops:
            obs = entity.obs_hops(hop, time_min=arrow.get(start).datetime, time_max=arrow.get(end).datetime)
            neighbor_ids = utils.get_entities_from_obs(obs)
            # neighbor_ids.remove(entity.id) #Uncomment this if not following NOTE 1
            neighbors = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(list(neighbor_ids)))
            
            attrs_all = {key:[] for key in attr_keys}
            for neighbor in neighbors:
                for key in attr_keys:
                    attrs_all[key].append(neighbor.attrs.get(key,0))
            
            attrs_all = {f'hop{hop}_{key}': attrs_all[key] for key in attrs_all.keys()} #add prefix hop number to the keys
            attrs_aggr = {key: np.sum(attrs_all[key]) for key in attrs_all.keys()}
            attrs_aggr_all = {**attrs_aggr_all, **attrs_aggr}
    
    return attrs_aggr_all


def save_persons_toxicity(start, end, hops=(1,2)):
    """
    For all persons who have observations in a window, save the sum of the toxicity counts of all nodes 1 and 2 hops out
    """
    df = pd.DataFrame()
    
    print('Getting persons in window...')
    entity_ids = utils.get_entities_in_window(start=start, end=end)
    
    with get_session() as sess:
        # Restrict persons
        #TODO: delete batch id filtering later when data is more fair. Right now, only batch id 25 has messages with toxicity
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(entity_ids)).where(sch.FusedEntity.type==sch.EntityTypeEnum.person).where(sch.FusedEntity.batch_id==25)
        
        print('Creating dataframe...')
        for person in tqdm(persons.all()):
            df = pd.concat((df,
                pd.DataFrame([{
                    **{'id':person.id},
                    **aggregate_attrs_of_neighbors(entity=person, attr_keys=TOXICITY_CATEGORIES, start=start, end=end, hops=hops)
                }])
            ))
    
    foldername = os.path.join(DATA_FOLDER, 'persons_toxicity')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, f'persons_toxicity_sum_{start}_{end}.csv'), index=False)


def save_persons_toxicity_wrapper(start_years=range(1998,2020)):
    for start_year in start_years:
        print(start_year)
        start=f'{start_year}-01-01'
        end=f'{start_year+1}-12-31'
        save_persons_toxicity(start=start, end=end)


def get_persons_toxicity(target_type, start_year, split=0.7, scaling='minmax', remove_all_zero_samples=True):
    """
    target_type: 'gaps' or 'activity'
    scaling: 'minmax' or 'log'
    """
    
    # features
    features = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'persons_toxicity'), f'persons_toxicity_sum_{start_year}-01-01_{start_year+1}-12-31.csv'), index_col='id')
    
    # remove samples which are all 0
    if remove_all_zero_samples:
        features['sum'] = features.sum(axis=1)
        features = features[features['sum']!=0]
        features.drop('sum', axis=1, inplace=True)
    
    # targets
    targs = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'targets'), 'persons_activity_yearly.csv' if target_type=='activity' else 'persons_183daygaps_yearly.csv'), index_col='id')
    if target_type=='activity':
        targs['target'] = targs.apply(lambda row: targets.get_persons_activity_target(row, start_year), axis=1)
    elif target_type=='gaps':
        targs['target'] = targs.apply(lambda row: targets.get_persons_gaps_target(row, start_year), axis=1)
    
    # join
    data = features.join(targs['target'])

    # shuffle
    data = data.sample(frac=1).reset_index()

    # split into train and test
    data_tr = data.iloc[:int(split*len(data))]
    data_va = data.iloc[int(split*len(data)):]

    # split into NN I/O
    x1tr = np.asarray(data_tr[[col for col in data.columns if col.startswith('hop1')]])
    x2tr = np.asarray(data_tr[[col for col in data.columns if col.startswith('hop2')]])
    ytr = np.asarray(data_tr['target'])
    x1va = np.asarray(data_va[[col for col in data.columns if col.startswith('hop1')]])
    x2va = np.asarray(data_va[[col for col in data.columns if col.startswith('hop2')]])
    yva = np.asarray(data_va['target'])

    # apply scaling
    if scaling=='minmax':
        x1tr,x1va,_ = utils.minmax_scaling(x1tr,x1va)
        x2tr,x2va,_ = utils.minmax_scaling(x2tr,x2va)
    elif scaling=='log':
        x1tr,x1va,_ = utils.log_scaling(x1tr,x1va)
        x2tr,x2va,_ = utils.log_scaling(x2tr,x2va)

    # return
    return {'x1tr':x1tr, 'x2tr':x2tr, 'ytr':ytr, 'x1va':x1va, 'x2va':x2va, 'yva':yva}


if __name__ == "__main__":
    pass