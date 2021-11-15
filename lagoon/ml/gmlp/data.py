import os
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional

import pandas as pd
import numpy as np

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.common import utils, targets
from lagoon.ml.config import *


def aggregate_attrs_of_neighbors(entity: sch.FusedEntity, attr_keys: List[str], start: str, end: str, hops: List[int]) -> Dict[str,int]:
    """
    For an `entity` in the graph, get all its neighbors of observations in the window defined by `start` and `end` which are x `hops` out, for all x in `hops`
    From the neighbors, aggregate the values of the attributes given in `attr_keys`
    Return a dictionary for each of these aggregated attributes for each x in `hops`
    
    `start` and `end` must be in string form, like '2020-02-15'
    
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
    
    for hop in hops:
        neighbors = utils.get_neighboring_entities(entity=entity, hop=hop, start=start, end=end, return_self=True) # Set return_self=True if not following NOTE 1
        
        attrs_all = {key:[] for key in attr_keys}
        for neighbor in neighbors:
            for key in attr_keys:
                attrs_all[key].append(neighbor.attrs.get(key,0))
        
        attrs_all = {f'hop{hop}_{key}': attrs_all[key] for key in attrs_all.keys()} #add prefix hop number to the keys
        attrs_aggr = {key: np.sum(attrs_all[key]) for key in attrs_all.keys()} #NOTE: Aggregation is hardcoded to sum right now
        attrs_aggr_all = {**attrs_aggr_all, **attrs_aggr}
    
    return attrs_aggr_all


def save_persons_toxicity(start: str, end: str, hops: List[int] = [1,2]) -> None:
    """
    For all persons who have observations in a window defined by `start` and `end`, save a dataframe containing the aggregation of the toxicity counts of all nodes x hops out, for all x in `hops`
    
    `start` and `end` must be in string form, like '2020-02-15'
    """
    df = pd.DataFrame()
    
    print('Getting persons in window...')
    entity_ids = utils.get_entities_in_window(start=start, end=end)
    
    with get_session() as sess:
        # Restrict persons
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(entity_ids)).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
        
        print('Creating dataframe...')
        for person in tqdm(persons.all()):
            df = pd.concat((
                df,
                pd.DataFrame([{
                    **{'id':person.id},
                    **aggregate_attrs_of_neighbors(entity=person, attr_keys=TOXICITY_CATEGORIES, start=start, end=end, hops=hops)
                }])
            ))
    
    foldername = os.path.join(DATA_FOLDER, 'persons_toxicity')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, f'persons_toxicity_sum_{start}_{end}.csv'), index=False)


def save_persons_toxicity_wrapper(start_years: List[int] = range(1998,2020)) -> None:
    """
    Run save_persons_toxicity starting from all years in start_years
    """
    for start_year in start_years:
        print(start_year)
        start=f'{start_year}-01-01'
        end=f'{start_year+1}-12-31'
        save_persons_toxicity(start=start, end=end)


def get_persons_toxicity(target_type: str, start_year: int, splits: Tuple[float,float] = (0.6,0.8), scaling: Optional[str] = 'log', remove_all_zero_samples: bool = True) -> Dict[str, np.array]:
    """
    Inputs:
        target_type: 'gaps' or 'activity'
        start_year: There must be a persons_toxicity data file with matching start year
        splits: (x,y), where both are between 0 and 1, and y>=x. 0 to x fraction data is used for training, x to y frac for validation, and y to 1 frac for testing.
            If validation is not desired, pass something like (0.7,0.7)
            If test is not desired, pass something like (0.7,1.0)
        scaling: None for no scaling, or 'minmax' or 'log'
        remove_all_zero_samples: If True, samples whose features are all 0 are removed.
    Returns:
        Dict with keys for x1<data>, x2<data> and y<data>, where <data> can be tr, va, te for train, validation, test, respectively. The values are numpy arrays. x1 and x2 are 2D, y is 1D. For the same <data>, x1, x2 and y must have same 0th dimension length.
    """
    
    # features
    features = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'persons_toxicity'), f'persons_toxicity_sum_{start_year}-01-01_{start_year+1}-12-31.csv'), index_col='id')
    
    # remove samples which are all 0
    if remove_all_zero_samples:
        features['sum'] = features.sum(axis=1)
        features = features[features['sum']!=0]
        features.drop('sum', axis=1, inplace=True)
    
    # targets
    _choices = ['activity','gaps']
    assert target_type in _choices, f'<target_type> must be one out of {_choices}'
    targs = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'targets'), 'persons_activity_yearly.csv' if target_type=='activity' else 'persons_183daygaps_yearly.csv'), index_col='id')
    if target_type=='activity':
        targs['target'] = targs.apply(lambda row: targets.get_persons_activity_target(row, start_year), axis=1)
    else:
        targs['target'] = targs.apply(lambda row: targets.get_persons_gaps_target(row, start_year), axis=1)
    
    # join features and targets
    data = features.join(targs['target'])

    # shuffle
    data = data.sample(frac=1).reset_index()

    # split
    assert len(splits) == 2, '<splits> must be of length 2'
    for i,split in enumerate(splits):
        assert 0. <= split <= 1., f'All elements of <splits> must be between 0 and 1, but found {split}'
        if i>0:
            assert splits[i] >= splits[i-1], f'Elements of <splits> must be monotonically increasing, but {splits[i]} is not >= {splits[i-1]}'
    splits = (int(splits[0]*len(data)), int(splits[1]*len(data)))
    data_tr = data.iloc[:splits[0]]
    data_va = data.iloc[splits[0]:splits[1]]
    data_te = data.iloc[splits[1]:]

    # split into NN I/O
    x1tr = np.asarray(data_tr[[col for col in data.columns if col.startswith('hop1')]])
    x2tr = np.asarray(data_tr[[col for col in data.columns if col.startswith('hop2')]])
    ytr = np.asarray(data_tr['target'])

    x1va = np.asarray(data_va[[col for col in data.columns if col.startswith('hop1')]])
    x2va = np.asarray(data_va[[col for col in data.columns if col.startswith('hop2')]])
    yva = np.asarray(data_va['target'])

    x1te = np.asarray(data_te[[col for col in data.columns if col.startswith('hop1')]])
    x2te = np.asarray(data_te[[col for col in data.columns if col.startswith('hop2')]])
    yte = np.asarray(data_te['target'])

    # apply scaling
    _choices = ['minmax','log', None]
    assert scaling in _choices, f'<scaling> must be one out of {_choices}'
    if scaling=='minmax':
        x1tr,x1va,x1te = utils.minmax_scaling(x1tr,x1va,x1te)
        x2tr,x2va,x2te = utils.minmax_scaling(x2tr,x2va,x2te)
    elif scaling=='log':
        x1tr,x1va,x1te = utils.log_scaling(x1tr,x1va,x1te)
        x2tr,x2va,x2te = utils.log_scaling(x2tr,x2va,x2te)

    # return
    return {
        'x1tr':x1tr, 'x2tr':x2tr, 'ytr':ytr,
        'x1va':x1va, 'x2va':x2va, 'yva':yva,
        'x1te':x1te, 'x2te':x2te, 'yte':yte
    }


if __name__ == "__main__":
    pass