import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple, List

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *
from lagoon.ml.common import utils


LABELS = {k: f'{k} PEPs: ' + ' / '.join(PEP_STATUSES[k]) for k in PEP_STATUSES.keys()}


def plot_authors_collab_matrix(cutoff: int = 0, ignore_diagonal: bool = True, figsize: Tuple[int,int] = (16,12)) -> None:
    """
    cutoff: How many of the top PEP authors to consider.
        Set to 0 to consider all PEP authors.
        A good number is 40.
    ignore_diagonal: Zero out the diagonal elements of authors_collab_matrix
        This is because usually the diagonal elements correspond to the number of PEPs authored solo, and is way more than the other elements
    """
    authors_stats = pd.read_csv(os.path.join(RESULTS_FOLDER, 'python_peps/authors_stats.csv'))
    num_authors = len(authors_stats)
    if cutoff == 0 or cutoff > num_authors:
        cutoff = num_authors
    
    authors_collab_matrix = np.load(os.path.join(RESULTS_FOLDER, 'python_peps/authors_collab_matrix.npy'))
    if ignore_diagonal:
        np.fill_diagonal(authors_collab_matrix, 0)

    names = list(authors_stats['name'][:cutoff])
    plt.figure(figsize=figsize)
    plt.imshow(authors_collab_matrix[:cutoff,:cutoff], cmap='Greys')
    cbar = plt.colorbar(orientation='vertical')
    cbar.ax.tick_params(labelsize=int(1.5*figsize[1]))
    plt.xlabel(r'$\longleftarrow$ Authors with more PEPs', fontsize=figsize[0])
    plt.ylabel(r'Authors with more PEPs $\longrightarrow$', fontsize=figsize[0])
    
    title_1 = 'Number of times authors have collaborated on PEPs'
    title_diag = '\nDiagonal elements show number of PEPs authored solo' if not ignore_diagonal else ''
    filename_diag = '_soloPEPs_diagonal' if not ignore_diagonal else ''
    
    if cutoff != num_authors:
        plt.xticks(range(cutoff), names, rotation=90, fontsize=figsize[1])
        plt.yticks(range(cutoff), names, fontsize=figsize[1])
        title = title_1 + f'\nShown for top {cutoff} authors with the most PEPs' + title_diag
        plt.title(title, fontsize=figsize[0]+4)
        plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/authors_collab_matrix_top{cutoff}{filename_diag}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.xticks([])
        plt.yticks([])
        title = title_1 + '\nShown for all PEP authors' + title_diag
        plt.title(title, fontsize=figsize[0]+4)
        plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/authors_collab_matrix_all{filename_diag}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


def plot_authors_pep_statuses(cutoff : int = 0, start: str = None, end: str = None, figsize: Tuple[int,int] = (16,12)):
    """
    cutoff: How many of the top PEP authors to consider.
        Set to 0 to consider all PEP authors.
        A good number is 40.
    start, end: If given, consider the corresponding authors_stats file.
        For possible options, check out the CSV files named as 'authors_stats_start_<start_year>_end_<end_year>' in `<RESULTS_FOLDER>/python_peps/`
    """
    authors_stats_filename = 'authors_stats'
    title_1st_line = f'PEP statuses of top {cutoff} authors with the most PEPs'
    if start:
        authors_stats_filename += f'_start_{start[:4]}'
        title_1st_line += f' from {start[:4]}'
    if end:
        authors_stats_filename += f'_end_{end[:4]}'
        title_1st_line += f' to {end[:4]}'
    
    authors_stats = pd.read_csv(os.path.join(RESULTS_FOLDER, f'python_peps/{authors_stats_filename}.csv'))
    authors_stats.fillna('', inplace=True)

    num_authors = len(authors_stats)
    if cutoff == 0 or cutoff > num_authors:
        cutoff = num_authors

    names = list(authors_stats['name'][:cutoff])
    num_peps = np.asarray(authors_stats['num_peps'][:cutoff])
    num_good_peps = np.asarray(authors_stats['num_good_peps'][:cutoff])
    num_bad_peps = np.asarray(authors_stats['num_bad_peps'][:cutoff])
    num_other_peps = np.asarray(authors_stats['num_other_peps'][:cutoff])
    frac_bad_good = np.round(num_bad_peps/num_good_peps, 1)

    plt.figure(figsize=figsize)
    plt.bar(names, num_good_peps, color='b', label=LABELS['good'])
    plt.bar(names, num_bad_peps, color='r', label=LABELS['bad'], bottom=num_good_peps)
    plt.bar(names, num_other_peps, color='k', label=LABELS['other'], bottom=num_good_peps+num_bad_peps)
    
    plt.ylabel('Number of PEPs', fontsize=int(1.5*figsize[1]))
    plt.xticks(rotation=90, fontsize=figsize[1])
    plt.yticks(fontsize=figsize[0])
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.title(f'{title_1st_line}\nRatio of bad to good PEPs shown on top of each bar', fontsize=figsize[0]+4)
    plt.legend(fontsize=figsize[0]+4)
    plt.grid(axis='y')

    annot_offset_y = num_peps[0]/100
    for i in range(len(names)):
        plt.annotate(str(frac_bad_good[i]), xy=(names[i],num_peps[i]+annot_offset_y), ha='center', va='bottom', rotation=60, fontsize=figsize[0])

    plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/{authors_stats_filename.replace("_stats","")}_pep_statuses_top{cutoff}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


def plot_authors_pep_statuses_bottom(start : int = 40, figsize: Tuple[int,int] = (16,12)):
    """
    start: Consider all PEP authors after this.
    """
    authors_stats = pd.read_csv(os.path.join(RESULTS_FOLDER, 'python_peps/authors_stats.csv'))
    authors_stats.fillna('', inplace=True)

    names = list(authors_stats['name'][start:])
    num_good_peps = np.asarray(authors_stats['num_good_peps'][start:])
    num_bad_peps = np.asarray(authors_stats['num_bad_peps'][start:])
    num_other_peps = np.asarray(authors_stats['num_other_peps'][start:])

    plt.figure(figsize=figsize)
    plt.bar(names, num_good_peps, color='b', label=LABELS['good'])
    plt.bar(names, num_bad_peps, color='r', label=LABELS['bad'], bottom=num_good_peps)
    plt.bar(names, num_other_peps, color='k', label=LABELS['other'], bottom=num_good_peps+num_bad_peps)
    
    plt.ylabel('Number of PEPs', fontsize=int(1.5*figsize[1]))
    plt.xticks([])
    plt.yticks(fontsize=figsize[0])
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f'PEP statuses of all authors except top {start} with the most PEPs\nRatio of bad to good PEPs shown on top of each bar', fontsize=figsize[0]+4)
    plt.legend(fontsize=figsize[0]+4)
    plt.grid(axis='y')

    plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/authors_pep_statuses_except_top{start}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


def plot_peps_toxicity(good_bad_buffer: int = 10, figsize: Tuple[int,int] = (16,12)):
    """
    good_bad_buffer: How much extra horizontal space to put between the plotted bars for good and bad PEPs
    """
    peps_stats = pd.read_csv(os.path.join(RESULTS_FOLDER, 'python_peps/peps_stats.csv'))

    for hop in [1,2]:
        for suffix in ['nlp_top_n_pct_mean', 'nlp_top_n_mean', 'badwords_messages_frac', 'badwords_words_frac']:
            key = f'hop{hop}_toxicity_{suffix}'
            series = peps_stats[key].dropna()
            good = np.asarray(series[peps_stats['status_quality']=='good'])
            bad = np.asarray(series[peps_stats['status_quality']=='bad'])
            if suffix=='badwords_words_frac': #convert to percentage
                good *= 100.
                bad *= 100.
                key = key.replace('_frac','_percentage')
            
            plt.figure(figsize=figsize)
            plt.bar(range(len(good)), good, color='b', label=LABELS['good'])
            plt.bar(range(len(good)+good_bad_buffer, len(good)+good_bad_buffer+len(bad)), bad, color='r', label=LABELS['bad'])
            plt.xticks([])
            plt.yticks(fontsize=figsize[0])
            plt.ylabel('Chance (%)' if suffix=='badwords_words_frac' \
                else 'Count' if suffix=='badwords_messages_frac' \
                else 'Score',
                fontsize=int(1.5*figsize[1]))
            title = 'Average number of bad words in' if suffix=='badwords_messages_frac' \
                else 'Chance of a word being bad in' if suffix=='badwords_words_frac' \
                else 'Average of top 10% of toxicity scores of' if suffix=='nlp_top_n_pct_mean' \
                else 'Average of top 10 toxicity scores of'
            plt.title(f'{title} texts in the {hop}-hop neighborhood of PEPs\nMean for good PEPs = {np.round(np.mean(good),3)}, Mean for bad PEPs = {np.round(np.mean(bad),3)}', fontsize=figsize[0]+4)
            plt.grid()
            plt.legend(fontsize=figsize[0]+2)
            plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/peps_{key}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


def analyze_toxic_peps(peps_hops: List[Tuple[int,int]]):
    """
    Check `lagoon_artifacts/results/python_peps/peps_stats.csv` for highly toxic PEPs
    Get their IDs and hop for which the messages are toxic
    Feed these as `peps_hops`, e.g. [(1234567,1), (476437,2)]
    Analyze why they are toxic
    """
    def print_badwords_in_messages(pep_id, hop):
        with get_session() as sess:
            pep = sess.query(sch.FusedEntity).get(pep_id)
            for ob in pep.obs_hops(hop):
                src = ob.src
                if src.type == sch.EntityTypeEnum.message:
                    for key in [elem.replace('computed_badwords_','badwords_ex_') for elem in TOXICITY_CATEGORIES]:
                        if key in src.attrs:
                            print(f'{key}: {src.attrs[key]}')
                    print("___________")
    
    def print_messages(pep_id, hop):
        with get_session() as sess:
            pep = sess.query(sch.FusedEntity).get(pep_id)
            for ob in pep.obs_hops(hop):
                src = ob.src
                if src.type == sch.EntityTypeEnum.message:
                    print(src.attrs['body_text'])
                    print("_________")

    for pep_id, hop in peps_hops:
        print_badwords_in_messages(pep_id, hop)
        print_messages(pep_id, hop)
        print("============================================================")


def plot_author_peps_disengagement(author_id: int, name: str, blue_vlines: List[str] = [], red_vlines: List[str] = [], figsize: Tuple[int,int] = (16,12)):
    """
    Plot the monthly activity of person with `author_id` and name `name`
    `name` should be given in the form "moshe_zadka"
    Optionally, add xx-colored vertical lines at `xx_vlines` (use blue for good PEPs, red for bad)
    """
    with get_session() as sess:
        entity = sess.query(sch.FusedEntity).get(author_id)
        stats = utils.get_entity_activity_monthly(entity)

    plt.figure(figsize=figsize)
    labels, vals = list(stats.keys()), list(stats.values())
    _,stemlines,_ = plt.stem(labels,vals, linefmt='k-', markerfmt='ko', basefmt='k-')
    plt.setp(stemlines, linewidth=1)
    xticks = [label for label in labels if label.endswith('-1')]
    xlabels = [xtick[2:4] for xtick in xticks]
    plt.xticks(ticks=xticks,labels=xlabels, fontsize=figsize[0]-4)
    plt.yticks(fontsize=figsize[0]-4)
    plt.xlabel('Year', fontsize=figsize[0])
    plt.ylabel('# Activities', fontsize=figsize[0])
    plt.title(' '.join([elem.capitalize() for elem in name.split('_')]), fontsize=figsize[0])
    for vline in blue_vlines:
        plt.axvline(vline, linestyle=':', c='b', linewidth=3)
    for vline in red_vlines:
        plt.axvline(vline, linestyle=':', c='r', linewidth=3)
    plt.grid(which='both',axis='y')
    plt.savefig(os.path.join(RESULTS_FOLDER, f'python_peps/monthly_activity_and_peps_{name}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)

def plot_author_peps_disengagement_wrapper():
    #Important NOTE: The author_ids will need to be modified every time the DB is updated 
    plot_author_peps_disengagement(
        author_id = 2490280,
        name = 'moshe_zadka',
        blue_vlines = ['2001-3','2001-3','2000-7'],
        red_vlines = ['2001-3','2001-3','2000-11','2000-7']
    )
    plot_author_peps_disengagement(
        author_id = 2510279,
        name = 'jim_jewett',
        blue_vlines = ['2012-3'],
        red_vlines = ['2008-5','2007-4','2007-4','2007-4']
    )
    plot_author_peps_disengagement(
        author_id = 2510086,
        name = 'greg_ewing',
        blue_vlines = ['2009-2'],
        red_vlines = ['2009-2','2004-8','2002-3']
    )
    plot_author_peps_disengagement(
        author_id = 2511118,
        name = 'michael_pelletier',
        red_vlines = ['2001-1','2003-2','2004-6']
    )
    plot_author_peps_disengagement(
        author_id = 2511197,
        name = 'mark_e_haase',
        red_vlines = ['2015-9','2016-11']
    )


if __name__ == "__main__":
    # plot_authors_collab_matrix()
    # plot_authors_collab_matrix(ignore_diagonal=False)
    # plot_authors_collab_matrix(cutoff=40)
    # plot_authors_pep_statuses(cutoff=40)
    # plot_authors_pep_statuses(cutoff=40, start='2016-01-01', end='2020-12-31')
    # plot_authors_pep_statuses_bottom(start=40)
    plot_peps_toxicity()
    # plot_author_peps_disengagement_wrapper()
