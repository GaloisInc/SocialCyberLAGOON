import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arrow

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *


def get_dates_active(person):
    """
    Input:
        person: FusedEntity of type person from graph
    Returns:
        Set of dates on which the person made an observation, i.e. something like a commit
    Eg:
        If a person made 2 commits on 2020-07-11 and 3 on 2020-07-23, this will return {datetime.date(2020-07-11), datetime.date(2020-07-23)}
    """
    obs = person.obs_hops(1)
    dates = set(ob.time.date() for ob in obs)
    return dates


def save_person_stats():
    """
    Save a csv with all person nodes and stats about them
    Columns may vary, best to see the Columns section of the code
    """
    ids = []
    num_obs = []
    
    with get_session() as sess:
        persons = sess.query(sch.FusedEntity).where(sch.FusedEntity.type == sch.EntityTypeEnum.person)
        for person in persons:
            ids.append(person.id)
            num_obs.append(len(person.obs_hops(1)))
            # num_dates_active.append(len(get_dates_active(person))) #skip this because it takes a lot of time and we are not using this right now

    person_stats = pd.DataFrame({'id':ids, 'num_obs':num_obs})
    foldername = os.path.join(RESULTS_FOLDER, 'person_monthly_stats')
    os.makedirs(foldername, exist_ok=True)
    person_stats.to_csv(os.path.join(foldername, 'person_stats.csv'), index=False)


def get_year_month_cols(start, end):
    """
    Return a list of years and months in the time span from first to last
    first and last must be datetime objects
    Eg:
        Earliest time is 2020-09-11 and latest is 2021-01-05, then return ['2020-9','2020-10','2020-11','2020-12','2021-1']
        Note that single-digit months are in single digits, i.e. 7 and not 07
    """
    cols = []
    start, end = arrow.get(start), arrow.get(end)
    for year in range(start.year,end.year+1):
        start_month = start.month if year==end.year else 1
        end_month = start.month if year==end.year else 12
        for month in range(start_month,end_month+1):
            cols.append(f'{year}-{month}')
    return cols


def save_top_persons_monthly_obs(num_persons=50, num_obs_threshold=None):
    """
    Save a csv with top person ids and their activity in terms of number of observations across all months. The csv looks something like:
    ```
    | id | 2020-07 | 2020-08 |
    | 12 |    23   |   103   |
    | 6  |   456   |    1    |
    ```
    Inputs:
        num_persons: If given, take this many most active persons
        num_obs_threshold: If num_persons is not given, use this and take persons who appear in this many observations
    """
    assert num_persons or num_obs_threshold, "Must specify either 'num_persons' or 'num_obs_threshold'"
    
    person_stats = pd.read_csv(os.path.join(os.path.join(RESULTS_FOLDER, 'person_monthly_stats'), 'person_stats.csv'))
    person_stats.sort_values('num_obs', ascending=False, inplace=True, ignore_index=True)
    if num_persons:
        top_persons = person_stats['id'][:num_persons]
    else:
        top_persons = person_stats['id'][person_stats['num_obs'] >= num_obs_threshold]

    year_month_cols = get_year_month_cols('1990-08-09','2021-06-28') #NOTE: these are for batch_id=3. Will need to be changed for other batch_ids.
    year_month_cols = pd.DataFrame({k:len(top_persons)*[0] for k in year_month_cols})
    top_persons = pd.concat((top_persons.to_frame(),year_month_cols), axis=1)
    
    with get_session() as sess:
        for _,row in top_persons.iterrows():
            person = sess.query(sch.FusedEntity).get(int(row['id']))
            obs = person.obs_hops(1)
            for ob in obs:
                row[f'{ob.time.year}-{ob.time.month}'] += 1

    foldername = os.path.join(RESULTS_FOLDER, 'person_monthly_stats')
    os.makedirs(foldername, exist_ok=True)
    top_persons.to_csv(os.path.join(foldername, 'top_persons_monthly_obs.csv'), index=False)


def plot_top_persons_monthly_obs():
    top_persons = pd.read_csv(os.path.join(os.path.join(RESULTS_FOLDER, 'person_monthly_stats'), 'top_persons_monthly_obs.csv'))
    cols = [col for col in top_persons.columns if col != 'id']
    xticks = [col for col in cols if col.endswith('-1')]
    xlabels = [tick[2:4] for tick in xticks]
    fontsize = 12
    
    foldername = os.path.join(os.path.join(RESULTS_FOLDER, 'person_monthly_stats'), 'figs_top_persons')
    os.makedirs(foldername, exist_ok=True)
    
    for i,row in top_persons.iterrows():
        vals = np.asarray(row[cols])
        plt.figure(figsize=(12,7))
        plt.title(f"Person with id = {row['id']}, total number of activities = {sum(vals)}", fontsize=fontsize)
        _,stemlines,_ = plt.stem(cols,vals)
        plt.setp(stemlines, 'linewidth',1)
        ax = plt.gca()
        ax.set_ylim(bottom=0.9)
        ax.set_yscale("log")
        plt.xticks(ticks=xticks,labels=xlabels)
        plt.xlabel('Year', fontsize=fontsize)
        plt.ylabel('# Activities', fontsize=fontsize)
        plt.grid(which='both',axis='y')
        plt.savefig(os.path.join(foldername, f"{i}_{row['id']}"), dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close() #avoid consuming excess memory by having too many figs open


if __name__ == "__main__":
    pass
