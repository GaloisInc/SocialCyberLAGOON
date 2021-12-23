import arrow
import os
import shortuuid
import random
import re
from typing import List, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

import sqlalchemy as sa
from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *


########################################################################
# Misc utils
########################################################################

def set_seed(seed):
    """
    Set a seed to make results reproducible
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_naive_performance(y, lossfunc):
    """
    Given targets `y`, find the performance of a naive predictor
    """
    y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    ones = torch.ones(y.shape[0], dtype=torch.float32, device=DEVICE)
    
    if lossfunc=='L1':
        naive_guess = torch.median(y)*ones
    elif lossfunc=='L2':
        naive_guess = torch.mean(y)*ones
    naive_loss = LOSSFUNC_MAPPING[lossfunc](naive_guess,y)

    return naive_loss


def plot_stats(stats, foldername, naive_val_loss=None):
    numepochs = len(stats['train_loss'])
    plt.figure()
    plt.plot(range(1,numepochs+1),stats['train_loss'], c='b', label='Train loss')
    if stats['val_loss']:
        plt.plot(range(1,numepochs+1),stats['val_loss'], c='r', label='Validation loss')
    if naive_val_loss:
        plt.axhline(y=naive_val_loss, c='k',linestyle='dashed', label='Naive validation loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    
    uuid = shortuuid.uuid()
    print(f'Saving figure {uuid}')
    os.makedirs(foldername, exist_ok=True)
    plt.savefig(os.path.join(foldername, f'{uuid}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


def count_words(inp: str) -> int:
    """
    Count the number of words in a string
        Example input: 'Guido van Rossum writes:\n > > What is the status of PEP 209?  I see'
        Method 1: len(re.findall(r'\w+', inp)) = 13 (Correct, takes into account '209')
        Method 2: sum([i.strip(string.punctuation).isalpha() for i in inp.split()]) = 12 (Incorrect, ignores '209')
        Method 3: len(inp.split()) = 15 (Incorrect, considers '>' and '>' as words)
    """
    return len(re.findall(r'\w+', inp))


########################################################################
# Monthly activity
########################################################################

def get_year_month_list(start: str, end: str):
    """
    Return a list of years and months in the time span from first to last
    first and last must be datetime objects
    Eg:
        start = '2020-09-11' and end = '2021-01-05', then return ['2020-9','2020-10','2020-11','2020-12','2021-1']
    """
    cols = []
    start, end = arrow.get(start).datetime, arrow.get(end).datetime
    for year in range(start.year,end.year+1):
        start_month = start.month if year==start.year else 1
        end_month = end.month if year==end.year else 12
        for month in range(start_month,end_month+1):
            cols.append(f'{year}-{month}')
    return cols


def get_entity_activity_monthly(entity: sch.FusedEntity, start: str = '1990-08-09', end: str = '2021-06-28'):
    """
    For a given `entity`, get the activity (i.e. observations) on a monthly basis starting from `start` to `end`
    Defaults are set to timeframe of cpython in DB
    """
    stats = {k:0 for k in get_year_month_list(start,end)}
    obs = entity.obs_hops(1)
    for ob in obs:
        t = ob.time
        stats[f'{t.year}-{t.month}'] = stats.get(f'{t.year}-{t.month}',0) + 1
    return stats


########################################################################
# Database interaction utils
########################################################################

def get_entity_ids_from_obs(obs: Union[List, sa.orm.query.Query]) -> List[int]:
    """
    Given a list or query of FusedObservations `obs`, get all entity IDs which are connected to those observations
    Entity_ids are deduplicated and returned as a list
    """
    entity_ids = set()
    for ob in obs:
        entity_ids.add(ob.src_id)
        entity_ids.add(ob.dst_id)
    return list(entity_ids)


def get_neighboring_entities(entity: sch.FusedEntity, hop: int = 1, start: str = None, end: str = None, return_self: bool = False) -> sa.orm.query.Query:
    """
    Given an entity `entity`, get all the entities at `hop`-hop distance from it for a time window from `start` to `end`. If None, use beginning of time to end of time.
    Return a query (instead of a list) so that additional SQL filtering can be applied if desired
    If return_self is False, remove `entity` from the returned entities since we generally don't want to return itself
    """
    entity_ids = get_entity_ids_from_obs(entity.obs_hops(k=hop, time_min = None if not start else arrow.get(start).datetime, time_max = None if not end else arrow.get(end).datetime))
    with get_session() as sess:
        if not return_self:
            entity_ids.remove(entity.id) #since get_entity_ids_from_obs also includes id of the entity itself
        neighbors = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(list(entity_ids)))
        return neighbors


def get_entities_in_window(start: str, end: str, method: int = 1) -> List[int]:
    """
    Get all entities which are connected to observations in a given time window

    Method 1 - Observation centric
    - Runtime to retrieve 4511 entities from 2015-01-01 to 2015-01-31 = 3.4 seconds
    - Runtime to retrieve 82885 entities from 2015-01-01 to 2016-12-31 = 21.44 seconds
    - Runtime to retrieve 338 persons from 2015-01-01 to 2015-01-31 = very large, as this would involve putting conditions like `if ob.src.type == sch.EntityTypeEnum.person` prior to the `entity_ids.add` statement

    Method 2 - Entity centric
    - Runtime to retrieve 4511 entities from 2015-01-01 to 2015-01-31 = very large
    - Runtime to retrieve 338 persons from 2015-01-01 to 2015-01-31 = 120 seconds
    - Runtime to retrieve 2625 persons from 2015-01-01 to 2015-01-31 = 144 seconds

    Method 3 - SQL joins
    - Runtime to retrieve 4511 entities from 2015-01-01 to 2015-01-31 = did not finish

    NOTE:
    - Method 1 is proportional to time frame since it retrieves observations in that time frame only. It requires type-filtering the returned entities later, such as `persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.id.in_(entity_ids)).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)`.
    - Method 2 is proportional to total entities in graph since it queries the whole graph. It is better to return already type-filtered entities.
    - Ignore Method 3.
    - Method 1 is usually superior since it is more generic (returns all entities which can be type-filtered as desired later) and is usually faster unless the time frame is very large, in which case we want to use Method 3 with some type-filtering.
    """
    if method==1:
        with get_session() as sess:
            obs = sess.query(sch.FusedObservation).where(sa.and_(sch.FusedObservation.time >= arrow.get(start).datetime, sch.FusedObservation.time <= arrow.get(end).datetime))
        return get_entity_ids_from_obs(obs)
    
    if method==2:
        entity_ids = []
        with get_session() as sess:
            for entity in sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.person):
                if entity.obs_hops(1, time_min=arrow.get(start).datetime, time_max=arrow.get(end).datetime):
                    entity_ids.append(entity.id)
        return entity_ids

    if method==3:
        with get_session() as sess:
            entity_ids = sess.query(sch.FusedEntity.id).where(sa.select(sch.FusedObservation).where(sch.FusedEntity.id.in_([sch.FusedObservation.src_id, sch.FusedObservation.dst_id])).where(sa.and_(sch.FusedObservation.time >= arrow.get(start).datetime, sch.FusedObservation.time <= arrow.get(end).datetime)).exists())
        return entity_ids.all()


def get_pep_authors(pep: sch.FusedEntity) -> List[str]:
    """
    Given a PEP entity from the DB, return the names of its authors as a list
    """
    authors = []
    for ob in pep.obs_hops(1):
        if ob.type == sch.ObservationTypeEnum.created and ob.src.type == sch.EntityTypeEnum.person:
            authors.append(ob.src.attrs['name'])
    return authors

def get_author_peps(author: sch.FusedEntity) -> List[int]:
    """
    Given an author entity from the DB, return the numbers of the PEPs authored by him/her as a list
    """
    peps = []
    for ob in author.obs_hops(1):
        if ob.type == sch.ObservationTypeEnum.created and ob.dst.type == sch.EntityTypeEnum.pep:
            peps.append(ob.dst.attrs['number'])
    return peps


########################################################################
# Scaling functions
########################################################################

def minmax_scaling(xtr, xva=None, xte=None):
    """
    Apply minmax scaling, i.e. data = (data-min)/(max-min)
    min and max are calculated on xtr (training data)
    xva (validation data) and xte (test data), if given, are normalized using training min and max
    """
    minvals = np.min(xtr, axis=0)
    maxvals = np.max(xtr, axis=0)
    xtr = (xtr-minvals)/(maxvals-minvals)
    if xva is not None:
        xva = (xva-minvals)/(maxvals-minvals)
    if xte is not None:
        xte = (xte-minvals)/(maxvals-minvals)
    return xtr,xva,xte

def log_scaling(xtr, xva=None, xte=None):
    """
    Apply log scaling, where data = log10(1+data)
    """
    xtr = np.log10(1+xtr)
    if xva is not None:
        xva = np.log10(1+xva)
    if xte is not None:
        xte = np.log10(1+xte)
    return xtr,xva,xte
