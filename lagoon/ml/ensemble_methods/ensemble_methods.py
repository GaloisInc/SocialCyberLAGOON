import os
import itertools
from tqdm import tqdm
import shortuuid

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

from lagoon.ml.config import *
from lagoon.ml.gmlp.data import get_persons_toxicity


def process_data(data):
    xtr = np.concatenate((data['x1tr'],data['x2tr']), axis=1)
    ytr = data['ytr']
    xva = np.concatenate((data['x1va'],data['x2va']), axis=1)
    yva = data['yva']
    return xtr,ytr, xva,yva


def save_df(df, cols):
    df.sort_values('loss', ascending=True, inplace=True)
    feature_cols = [col for col in cols if col.startswith('hop')]
    top_10pc_means = [np.mean(df[col].iloc[:int(0.1*len(df))+1]) for col in feature_cols]
    means = [np.mean(df[col].iloc[:]) for col in feature_cols]
    df = pd.concat([df, pd.DataFrame( {**{'Label':['top_10pc_mean']}, **{key:[val] for key,val in zip(feature_cols,top_10pc_means)}} )])
    df = pd.concat([df, pd.DataFrame( {**{'Label':['mean']}, **{key:[val] for key,val in zip(feature_cols,means)}} )])
    
    uuid = shortuuid.uuid()
    print(f'Saving df {uuid}')
    foldername = os.path.join(RESULTS_FOLDER, 'ensemble_methods')
    os.makedirs(foldername, exist_ok=True)
    df.to_csv(os.path.join(foldername, f'{uuid}.csv'), index=False)


def gradient_boosting_feature_importances():
    
    ## Get data
    xtr,ytr, xva,yva = process_data(
        data = get_persons_toxicity(
            target_type='activity',
            start_year=2001,
            splits=(0.7,1.0),
            scaling='log',
            remove_all_zero_samples=True
        )
    )

    ## Define hyperparameter ranges
    n_estimators_all = [100,300,1000]
    learning_rate_all = [0.1,0.03,0.01]
    min_samples_split_all = [2,6,20,50]
    max_depth_all = [2,3,4,5]
    max_features_all = [4,8,16]

    options = list(itertools.product(n_estimators_all, learning_rate_all, min_samples_split_all, max_depth_all, max_features_all))
    cols = ['n_estimators','learning_rate','min_samples_split','max_depth','max_features', 'loss', *[f'hop1_{tc}' for tc in TOXICITY_CATEGORIES], *[f'hop2_{tc}' for tc in TOXICITY_CATEGORIES]]
    df = pd.DataFrame({})
    
    for option in tqdm(options):
        n_estimators, learning_rate, min_samples_split, max_depth, max_features = option

        net = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,

            loss='lad',
            subsample=0.9,
            min_samples_leaf=int(min_samples_split/2)
        )
        net.fit(xtr,ytr)
        preds = net.predict(xva)
        loss = np.mean(np.abs(yva-preds))
        fi = net.feature_importances_

        results = [n_estimators,learning_rate,min_samples_split,max_depth,max_features, loss, *fi]
        df = pd.concat([df, pd.DataFrame({key:[val] for key,val in zip(cols,results)})])

    save_df(df, cols)


def random_forest_feature_importances():

    ## Get data
    xtr,ytr, xva,yva = process_data(
        data = get_persons_toxicity(
            target_type='activity',
            start_year=2001,
            splits=(0.7,1.0),
            scaling='log',
            remove_all_zero_samples=True
        )
    )

    ## Define hyperparameter ranges
    n_estimators_all = [100,300,1000]
    min_samples_split_all = [2,6,20,50]
    max_depth_all = [2,3,4,5]
    max_features_all = [4,8,16]

    options = list(itertools.product(n_estimators_all, min_samples_split_all, max_depth_all, max_features_all))
    cols = ['n_estimators','min_samples_split','max_depth','max_features', 'loss', *[f'hop1_{tc}' for tc in TOXICITY_CATEGORIES], *[f'hop2_{tc}' for tc in TOXICITY_CATEGORIES]]
    df = pd.DataFrame({})
    
    for option in tqdm(options):
        n_estimators, min_samples_split, max_depth, max_features = option

        net = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,

            min_samples_leaf=int(min_samples_split/2)
        )
        net.fit(xtr,ytr)
        preds = net.predict(xva)
        loss = np.mean(np.abs(yva-preds))
        fi = net.feature_importances_

        results = [n_estimators,min_samples_split,max_depth,max_features, loss, *fi]
        df = pd.concat([df, pd.DataFrame({key:[val] for key,val in zip(cols,results)})])
    
    save_df(df, cols)


def plot_feature_importances():
    """
    Currently this contains the code used to obtain the feature importances figure for the M3 report
    It uses the CSVs for grdient boosting 2001 activity and gaps
    """
    mapping = {
        'GoogleInstant': [f'hop{i}_computed_badwords_googleInstantB_any' for i in [1,2]],
        'Swearing': [f'hop{i}_computed_badwords_swearing_any' for i in [1,2]],
        'Generic': [f'hop{i}_computed_badwords_mrezvan94Harassment_Generic' for i in [1,2]],
        'Appearance': [f'hop{i}_computed_badwords_mrezvan94Harassment_Appearance' for i in [1,2]],
        'Intelligence': [f'hop{i}_computed_badwords_mrezvan94Harassment_Intelligence' for i in [1,2]],
        'Politics': [f'hop{i}_computed_badwords_mrezvan94Harassment_Politics' for i in [1,2]],
        'Racial': [f'hop{i}_computed_badwords_mrezvan94Harassment_Racial' for i in [1,2]],
        'Sexual': [f'hop{i}_computed_badwords_mrezvan94Harassment_Sexual' for i in [1,2]]
    }
    values = {key:[] for key in mapping.keys()} #hop1 activity, hop2 activity, hop1 gaps, hop2 gaps

    ## Activity
    df = pd.read_csv(os.path.join(os.path.join(RESULTS_FOLDER, 'ensemble_methods'), 'hPD8n98JtNT3StTnieqvhJ.csv'))
    for key in values.keys():
        for col in mapping[key]:
            values[key].append(df.iloc[-2][col])

    ## Gaps
    df = pd.read_csv(os.path.join(os.path.join(RESULTS_FOLDER, 'ensemble_methods'), 'MKj852TTfYqs935wRHDyf9.csv'))
    for key in values.keys():
        for col in mapping[key]:
            values[key].append(df.iloc[-2][col])

    ## Organize data
    keys = ['GoogleInstant', 'Swearing', 'Generic', 'Appearance', 'Intelligence', 'Politics', 'Racial', 'Sexual']
    data = []
    for key in keys:
        data.extend(values[key]+[0,0])
    data = 100.*np.asarray(data[:-2])
    
    ## Plot
    ypos = np.arange(len(data))
    colors = 8*['b','r','g','k','k','k'] #last 2 'k's don't matter because those values are 0
    colors = colors[:-2]
    labels = []
    for key in keys:
        labels.extend(['','',key,'','',''])
    labels = labels[:-2]
    
    plt.figure(figsize=(15,7))
    ax = plt.gca()
    plt.bar(ypos, data, color=colors)
    plt.ylabel('Importance (%)', fontsize=16)
    plt.xticks(ypos,labels, fontsize=15)
    plt.yticks(fontsize=15)
    ax.tick_params(axis='x',length=0)
    plt.grid(axis='y')
    plt.annotate('Activity disengagement, 1-hop', xycoords='figure fraction', xy=(0.5,0.75), fontsize=16, color='b', fontweight='semibold')
    plt.annotate('Activity disengagement, 2-hop', xycoords='figure fraction', xy=(0.5,0.7), fontsize=16, color='r', fontweight='semibold')
    plt.annotate('Gaps disengagement, 1-hop', xycoords='figure fraction', xy=(0.5,0.65), fontsize=16, color='g', fontweight='semibold')
    plt.annotate('Gaps disengagement, 2-hop', xycoords='figure fraction', xy=(0.5,0.6), fontsize=16, color='k', fontweight='semibold')
    
    uuid = shortuuid.uuid()
    print(f'Saving figure {uuid}')
    foldername = os.path.join(RESULTS_FOLDER, 'ensemble_methods')
    os.makedirs(foldername, exist_ok=True)
    plt.savefig(os.path.join(foldername, f'{uuid}.png'), dpi=600, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    pass
