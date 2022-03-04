import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from collections import defaultdict
import arrow

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *
from lagoon.ml.common import utils


def save_authors_stats(start: str = None, end: str = None) -> None:
    """
    Create a Pandas dataframe to store authors and their corresponding PEP info inside RESULTS_FOLDER
    start, end: If given as dates, limit the PEPs considered to only those authored between these dates
    """
    ## First create a dict indexed by author keys
    authors_dict = defaultdict(lambda: defaultdict(list))
    with get_session() as sess:
        created_obs = (sess.query(sch.FusedObservation)
            .where(sch.FusedObservation.type==sch.ObservationTypeEnum.created)
            .join(sch.FusedObservation.dst)
            .where(sch.FusedEntity.type==sch.EntityTypeEnum.pep)
        )
        if start:
            created_obs = created_obs.where(sch.FusedObservation.time >= arrow.get(start).datetime)
        if end:
            created_obs = created_obs.where(sch.FusedObservation.time <= arrow.get(end).datetime)
        for ob in created_obs:
            author_id = ob.src_id
            author = sess.query(sch.FusedEntity).get(author_id)
            authors_dict[author_id]['name'] = author.attrs['name']
            authors_dict[author_id]['pep_numbers'].append(ob.dst.attrs['number'])
            authors_dict[author_id]['pep_types'].append(ob.dst.attrs['type'])
            authors_dict[author_id]['pep_statuses'].append(ob.dst.attrs['status'])
    
    ## Now create a Pandas df and write
    authors_stats = {k: [] for k in [
        'DB_id',
        'name',
        'peps',
        'num_peps',
        'S_peps',
        'num_S_peps',
        'P_peps',
        'num_P_peps',
        'I_peps',
        'num_I_peps',
        'good_peps',
        'num_good_peps',
        'bad_peps',
        'num_bad_peps',
        'other_peps',
        'num_other_peps'
    ]} # it would be better to use a defaultdict, but this ensures an easy way to see all the columns

    for k,v in authors_dict.items():
        sort_key = [v['pep_numbers'].index(elem) for elem in sorted(v['pep_numbers'])]
        v['pep_numbers'] = [v['pep_numbers'][i] for i in sort_key]
        v['pep_types'] = [v['pep_types'][i] for i in sort_key]
        v['pep_statuses'] = [v['pep_statuses'][i] for i in sort_key]

        authors_stats['DB_id'].append(k)
        authors_stats['name'].append(v['name'])

        authors_stats['peps'].append(' '.join([str(elem) for elem in v['pep_numbers']]))
        authors_stats['num_peps'].append(len(v['pep_numbers']))
        
        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_types'][i].lower().startswith('standards')]
        authors_stats['S_peps'].append(' '.join(peps))
        authors_stats['num_S_peps'].append(len(peps))

        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_types'][i].lower().startswith('process')]
        authors_stats['P_peps'].append(' '.join(peps))
        authors_stats['num_P_peps'].append(len(peps))

        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_types'][i].lower().startswith('information')]
        authors_stats['I_peps'].append(' '.join(peps))
        authors_stats['num_I_peps'].append(len(peps))
        
        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_statuses'][i].lower().startswith(tuple(PEP_STATUSES['good']))]
        authors_stats['good_peps'].append(' '.join(peps))
        authors_stats['num_good_peps'].append(len(peps))

        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_statuses'][i].lower().startswith(tuple(PEP_STATUSES['bad']))]
        authors_stats['bad_peps'].append(' '.join(peps))
        authors_stats['num_bad_peps'].append(len(peps))

        peps = [str(v['pep_numbers'][i]) for i in range(len(v['pep_numbers'])) if v['pep_statuses'][i].lower().startswith(tuple(PEP_STATUSES['other']))]
        authors_stats['other_peps'].append(' '.join(peps))
        authors_stats['num_other_peps'].append(len(peps))
    
    authors_stats = pd.DataFrame(data=authors_stats)
    authors_stats.sort_values('num_peps', ascending=False, inplace=True, ignore_index=True)
    authors_stats_filename = 'authors_stats'
    if start:
        authors_stats_filename += f'_start_{start[:4]}'
    if end:
        authors_stats_filename += f'_end_{end[:4]}'
    authors_stats.to_csv(os.path.join(RESULTS_FOLDER, f'python_peps/{authors_stats_filename}.csv'), index=False)


def save_authors_collab_matrix() -> None:
    """
    Save a square symmetric matrix (numpy 2d array) showing how many PEPs each PEP author has co-authored with others
    PEPs with solo authors get 1 added to the respective diagonal element
    Eg:
        Say Tom has authored PEPs 1 and 2 solo, Dick has authored PEP 3 solo, and Tom and Dick have co-authored PEPs 4, 5 and 6.
        Output: [2 3; 3 1], where the 1st row / col is Tom and the 2nd is Dick
    """
    authors_stats = pd.read_csv(os.path.join(RESULTS_FOLDER, 'python_peps/authors_stats.csv'))
    num_authors = len(authors_stats)
    authors_collab_matrix = np.zeros((num_authors, num_authors))
    
    with get_session() as sess:
        peps = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.pep)
        for pep in tqdm(peps, total=peps.count()):
            pep_authors = utils.get_pep_authors(pep)
            pep_authors_indexes = [authors_stats['name'][authors_stats['name']==author].index[0] for author in pep_authors]
            if len(pep_authors_indexes) == 1: # 1 author => increment diagonal
                authors_collab_matrix[pep_authors_indexes[0],pep_authors_indexes[0]] += 1
            else: # >1 author => increment non-diagonals
                author_combinations = list(itertools.combinations(pep_authors_indexes, 2))
                for author_combination in author_combinations:
                    authors_collab_matrix[author_combination] += 1
    
    authors_collab_matrix = authors_collab_matrix + authors_collab_matrix.T - np.diag(np.diag(authors_collab_matrix)) # copy upper triangular part to lower triangular
    np.save(os.path.join(RESULTS_FOLDER, 'python_peps/authors_collab_matrix.npy'), authors_collab_matrix)


def save_peps_stats(compute_badwords=False, compute_nlp=True) -> None:
    """
    Save stats for PEPs, ordered by number
        Some common stats are computed
        compute_badwords: If set, toxicity stats from bad words are computed
        compute_nlp: If set, toxicity stats from NLP model(s) are computed
    Toxicity stats can be aggregated in various ways
    """
    cols = [
        'number',
        'type',
        'status',
        'status_quality',
    ]
    if compute_badwords:
        cols += [
            'hop1_toxicity_badwords_messages_frac', # divide total badword count by number of neighboring messages
            'hop1_toxicity_badwords_words_frac', # divide total badword count by number of words in neighboring messages
            'hop2_toxicity_badwords_messages_frac',
            'hop2_toxicity_badwords_words_frac'
        ]
    if compute_nlp:
        cols += [
            'hop1_toxicity_nlp_top_n_pct_mean', # relevant functions are defined in utils, under aggregating functions
            'hop1_toxicity_nlp_top_n_mean',
            'hop2_toxicity_nlp_top_n_pct_mean',
            'hop2_toxicity_nlp_top_n_mean'
        ]
    peps_stats = {k: [] for k in cols} # it would be better to use a defaultdict, but this ensures an easy way to see all the columns

    def get_status_quality(status: str) -> str:
        for key in PEP_STATUSES.keys():
            if status.lower().startswith(tuple(PEP_STATUSES[key])):
                return key
        return 'unknown'
    
    with get_session() as sess:
        peps = sess.query(sch.FusedEntity).where(sch.FusedEntity.type==sch.EntityTypeEnum.pep)
        for pep in tqdm(peps, total=peps.count()):
            peps_stats['number'].append(pep.attrs['number'])
            peps_stats['type'].append(pep.attrs['type'])
            peps_stats['status'].append(pep.attrs['status'])
            peps_stats['status_quality'].append(get_status_quality(pep.attrs['status']))
            
            if compute_badwords:
                for hop in [1,2]:
                    computed_badwords_total = 0
                    words_total = 0
                    messages = utils.get_neighboring_entities(sess, pep, hop=hop).where(sch.FusedEntity.type==sch.EntityTypeEnum.message)
                    
                    for message in messages:
                        computed_badwords_total += sum([message.attrs.get(key,0) for key in TOXICITY_CATEGORIES])
                        words_total += utils.count_words(message.attrs.get('body_text',''))
                    
                    denom = messages.count()
                    peps_stats[f'hop{hop}_badwords_messages_frac'].append(computed_badwords_total / denom if denom!=0 else np.nan)
                    
                    denom = words_total
                    peps_stats[f'hop{hop}_badwords_words_frac'].append(computed_badwords_total / denom if denom!=0 else np.nan)

            if compute_nlp:
                text_entity_types = [
                    sch.EntityTypeEnum.git_commit,
                    sch.EntityTypeEnum.message
                ]
                for hop in [1,2]:
                    text_neighbors = utils.get_neighboring_entities(sess, pep, hop=hop).where(sch.FusedEntity.type.in_(text_entity_types))
                    toxicity_nlp_scores = [text_neighbor.attrs.get('toxicity_nlp_classification', [1.,0.])[1] for text_neighbor in text_neighbors]
                    peps_stats[f'hop{hop}_toxicity_nlp_top_n_pct_mean'].append(utils.top_n_pct_mean(toxicity_nlp_scores) if toxicity_nlp_scores else np.nan)
                    peps_stats[f'hop{hop}_toxicity_nlp_top_n_mean'].append(utils.top_n_mean(toxicity_nlp_scores) if toxicity_nlp_scores else np.nan)

    peps_stats = pd.DataFrame(data=peps_stats)
    peps_stats.sort_values('number', ascending=True, inplace=True, ignore_index=True)
    peps_stats.to_csv(os.path.join(RESULTS_FOLDER, 'python_peps/peps_stats.csv'), index=False)
