from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *
from lagoon.ml.common import targets, utils

import os
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from ast import literal_eval
from typing import List, Optional


def save_persons_toxicity_nlp(committers_only=True, starts : Optional[List[str]] = None, ends : Optional[List[str]] = None):
    """
    Collect the toxicity NLP classification scores for all entities linked to each person in the DB
    committers_only: If set, only consider persons with at least 1 commit
    starts, ends: Must be lists of strings in format 'yyyy-mm-dd'
        If given, loop over (start,end) pairs and only consider toxicity from neighbors connected by observations in this time frame
        Note that persons are not limited to time frame, i.e. all persons (or those with commits only if committers_only is set) are considered from BOT to EOT.
    """
    if (starts and not ends) or (ends and not starts):
        assert False, 'Both starts and ends must be provided, or neither'
    if starts and ends:
        assert len(starts) == len(ends), 'starts and ends, if provided, must be of same length'
    
    text_entity_types = [
        sch.EntityTypeEnum.git_commit,
        sch.EntityTypeEnum.message
    ]
    with get_session() as sess:
        if committers_only:
            person_ids = set()
            person_to_commit_obs = (
                sess.query(sch.FusedObservation)
                .join(sch.FusedObservation.src)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.person)
            ).intersect(
                sess.query(sch.FusedObservation)
                .join(sch.FusedObservation.dst)
                .where(sch.FusedEntity.type == sch.EntityTypeEnum.git_commit)
            ).distinct() # Get obs which are person --> commit
            for ob in person_to_commit_obs:
                person_ids.add(ob.src_id)
            persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(list(person_ids)))
        else:
            persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type == sch.EntityTypeEnum.person)
        
        count = persons.count()
        
        if not starts and not ends:
            starts = [None]
            ends = [None]
        
        for start,end in zip(starts,ends):
            with open(os.path.join(DATA_FOLDER, f"persons_toxicity_nlp/persons{'_committers_only' if committers_only else ''}_toxicity_nlp{'_'+start+'_'+end if start and end else ''}.csv"), 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([
                    'id',
                    'toxicity_nlp_classification_scores',
                ])
                if start and end:
                    print(f'{start} to {end} ...')
                
                for person in tqdm(persons, total=count):
                    if not person.attrs.get('name'): # this excludes persons with only emails added in the HIBP ingest 
                        continue
                    
                    text_neighbors = utils.get_neighboring_entities(
                        sess, person,
                        start=start, end=end
                    ).where(sch.FusedEntity.type.in_(text_entity_types))
                    toxicity_nlp_classification_scores = [round(text_neighbor.attrs.get('toxicity_nlp_classification',[1.,0.])[1], 4) for text_neighbor in text_neighbors]
                    
                    if toxicity_nlp_classification_scores:
                        csvwriter.writerow([
                            person.id,
                            toxicity_nlp_classification_scores
                        ])

def save_persons_toxicity_nlp_wrapper():
    save_persons_toxicity_nlp(
        committers_only = True,
        starts = [f'{year}-01-01' for year in range(1998,2020)],
        ends = [f'{year}-12-31' for year in range(1999,2021)]
    )


def get_persons_toxicity_nlp_hibp_breaches_data(features_aggr_func = utils.top_n_pct_mean, targets_cap = 100, targets_log_scale = False):
    """
    features_aggr_func: Use this function to aggregate all the toxicity scores into a single number
    targets_cap: Discard samples with more breaches than this number
    targets_log_scale: If True, replace breaches with log10(1+breaches)
    """
    features = pd.read_csv(os.path.join(DATA_FOLDER, 'persons_toxicity_nlp/persons_committers_only_toxicity_nlp.csv'), index_col='id')
    features['toxicity_aggr'] = features['toxicity_nlp_classification_scores'].apply(lambda val: features_aggr_func(literal_eval(val)))
    
    targs = pd.read_csv(os.path.join(DATA_FOLDER, 'targets/persons_hibp_breaches.csv'), index_col='id')
    if targets_cap:
        targs = targs[targs['breaches'] <= targets_cap]
    if targets_log_scale:
        targs['breaches'] = targs['breaches'].apply(lambda val: np.log(1+val))
    
    data = features.join(targs['breaches'], how='inner')
    data.rename(columns = {'breaches':'x', 'toxicity_aggr':'y'}, inplace=True)
    return data


def get_persons_toxicity_nlp_disengagement_data(features_aggr_func = utils.top_n_pct_mean, start_year = 2019, target_type = 'activity'):
    """
    features_aggr_func: As above
    start_year: Since disengagement targets are spoecific to year
    target_type: 'gaps' or 'activity'
    """
    features = pd.read_csv(os.path.join(DATA_FOLDER, f'persons_toxicity_nlp/persons_committers_only_toxicity_nlp_{start_year}-01-01_{start_year+1}-12-31.csv'), index_col='id')
    features['toxicity_aggr'] = features['toxicity_nlp_classification_scores'].apply(lambda val: features_aggr_func(literal_eval(val)))

    _choices = ['activity','gaps']
    assert target_type in _choices, f'<target_type> must be one out of {_choices}'
    targs = pd.read_csv(os.path.join(os.path.join(DATA_FOLDER, 'targets'), 'persons_activity_yearly.csv' if target_type=='activity' else 'persons_183daygaps_yearly.csv'), index_col='id')
    if target_type=='activity':
        targs['target'] = targs.apply(lambda row: targets.get_persons_activity_target(row, start_year), axis=1)
    else:
        targs['target'] = targs.apply(lambda row: targets.get_persons_gaps_target(row, start_year), axis=1)
    
    data = features.join(targs['target'])
    data.rename(columns = {'toxicity_aggr':'x', 'target':'y'}, inplace=True)
    return data


def split_data(data, split: float = 0.7):
    """
    data: From one of the get() methods
    split: Train test split
    """
    data = data.sample(frac=1).reset_index()
    
    split = int(split*len(data))
    data_tr = data.iloc[:split]
    data_te = data.iloc[split:]

    xtr = np.asarray(data_tr['x']).reshape(-1,1).astype(float)
    ytr = np.asarray(data_tr['y'])
    xte = np.asarray(data_te['x']).reshape(-1,1).astype(float)
    yte = np.asarray(data_te['y'])

    return {
        'xtr':xtr, 'ytr':ytr,
        'xte':xte, 'yte':yte
    }
