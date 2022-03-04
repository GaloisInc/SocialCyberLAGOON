########################################################################
# This file contains data processing functions to obtain targets
########################################################################

import arrow
import os
from ast import literal_eval
from tqdm import tqdm
import csv

import pandas as pd
import numpy as np

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *


def save_persons_gapranges(gap=183, dtformat='%Y-%m-%d'):
    """
    Retrieve all persons from the db and save a dataframe with their periods of inactivity if such period exceeds `gap` days
    Output dataframe cols:
        'id': id of person
        'gaps': List of tuples of datetimes in format `dtformat`. Example: [('2011-04-05','2012-01-01'), ('2019-03-06','2019-11-11')]
    """
    id_all = []
    gaps_all = []

    with get_session() as sess:
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
        count = persons.count()
        
        for person in tqdm(persons, total=count):
            id_all.append(person.id)
            datetimes = sorted([ob.time for ob in person.obs_hops(1)])
            
            gaps = []
            for j in range(1,len(datetimes)):
                if (datetimes[j] - datetimes[j-1]).days >= gap:
                    gaps.append((datetimes[j-1].strftime(dtformat),datetimes[j].strftime(dtformat)))
            gaps_all.append(gaps)

    df = pd.DataFrame({'id':id_all, 'gaps':gaps_all})
    foldername = os.path.join(DATA_FOLDER, 'targets')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, f'persons_{gap}daygapranges_all.csv'), index=False)


def save_persons_gaps_yearly(years = range(1990,2022), persons_gaps_filepath = os.path.join(os.path.join(DATA_FOLDER, 'targets'), 'persons_183daygapranges_all.csv')):
    """
    Should be run after running save_persons_gapranges
    Retrieve all persons from the db and save a dataframe with their annual gaps for all years in `years`
    Gaps are taken from `persons_gaps_filepath`
    Gaps for a year is the number of gaps beginning in that year
    """
    persons_gaps = pd.read_csv(persons_gaps_filepath, index_col='id')
    
    with get_session() as sess:
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
        count = persons.count()
        id_all = count*[0]
        gaps_all = {year:count*[0] for year in years}

        for i,person in enumerate(tqdm(persons, total=count)):
            id_all[i] = person.id
            gaps = literal_eval(persons_gaps['gaps'][person.id])
            
            for gap in gaps:
                gaps_all[arrow.get(gap[0]).year][i] += 1

    df = pd.DataFrame({**{'id':id_all}, **gaps_all})
    foldername = os.path.join(DATA_FOLDER, 'targets')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, 'persons_183daygaps_yearly.csv'), index=False)


def save_persons_activity_yearly(years = range(1990,2022)):
    """
    Retrieve all persons from the db and save a dataframe with their annual activity for all years in `years`
    Activity for a year is the number of activities in that year
    """
    with get_session() as sess:
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
        count = persons.count()
        id_all = count*[0]
        activity = {year:count*[0] for year in years}

        for i,person in enumerate(tqdm(persons, total=count)):
            id_all[i] = person.id
            for ob in person.obs_hops(1):
                activity[ob.time.year][i] += 1
        
    df = pd.DataFrame({**{'id':id_all}, **activity})
    foldername = os.path.join(DATA_FOLDER, 'targets')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, 'persons_activity_yearly.csv'), index=False)


def get_persons_gaps_target(row, start_year):
    """
    Get the gaps target for a single row in a df (i.e. for a single person) for a certain year
    Gaps target = Sum gaps over this year and next
    """
    target = row[str(start_year+1)] + row[str(start_year+2)]
    return target


def get_persons_activity_target(row, start_year):
    """
    Get the activity target for a single row in a df (i.e. for a single person) for a certain year
    Activity target = 1 - Avg activity for next 2 years / Avg activity for this year and next
    """
    prev_avg = np.mean((row[str(start_year)],row[str(start_year+1)]))
    next_avg = np.mean((row[str(start_year+1)],row[str(start_year+2)]))
    if prev_avg==0 or next_avg>prev_avg: #NOTE: may remove the greater than constraint to make increased activity show up as negative targets.
        target = 0
    else:
        target = 1 - next_avg/prev_avg
    return target


def save_persons_hibp_breaches():
    """
    For each person FusedEntity in the db, sum up attrs['breaches'] for its component entities
    Save this summed breaches number as the target
    
    Note:
        If none of the component entities for a FusedEntity have attrs['breaches'], then that person is excluded due to having no valid, non-spam emails
        This is different from the case where a person has valid, non-spam emails, none of which are found in HIBP. That leads to the target being 0, as expected.
    """
    with get_session() as sess:
        
        with open(os.path.join(DATA_FOLDER, 'targets/persons_hibp_breaches.csv'), 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'id',
                'breaches'
            ])

            persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.person)
            count = persons.count()
            
            for person in tqdm(persons, total=count):
                breaches = 0
                breaches_found = False # track whether any component entity has attrs['breaches']
                
                for ent in person.attrs_sources:
                    try:
                        breaches += ent.attrs['breaches']
                        breaches_found = True # at least one component entity must have the 'breaches' attr
                    except KeyError:
                        pass

                if breaches_found:
                    csvwriter.writerow([
                        person.id,
                        breaches
                    ])
