from lagoon.ml.config import *
from lagoon.ml.common import utils

from sklearn.linear_model import LinearRegression
import data as data_methods
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def run_regression(trials = 100, start_year = 2019):
    scores_tr = np.zeros(trials)
    scores_te = np.zeros(trials)
    slopes = np.zeros(trials)
    intercepts = np.zeros(trials)
    split = 0.7

    ## NOTE: Change this block depending on problem ##
    # For disengagement
    data = data_methods.get_persons_toxicity_nlp_disengagement_data(features_aggr_func = utils.top_n_pct_mean, start_year = start_year, target_type = 'gaps')
    result_name = f'persons_toxicity_nlp_disengagement_regression_top10pctmean_{start_year}-{start_year+2}'
    # For HIBP
    # data = data_methods.get_persons_toxicity_nlp_hibp_breaches_data(features_aggr_func = utils.top_n_mean, targets_cap = 100, targets_log_scale = False)
    # result_name = 'persons_toxicity_nlp_hibp_breaches_regression_top10mean'

    for i in tqdm(range(trials)):
        split_data = data_methods.split_data(data, split=split) # does a different random split for each trial
        xtr, ytr, xte, yte = split_data['xtr'], split_data['ytr'], split_data['xte'], split_data['yte']

        reg = LinearRegression()
        reg.fit(xtr,ytr)
        scores_tr[i] = reg.score(xtr,ytr)

        slopes[i] = reg.coef_
        intercepts[i] = reg.intercept_

        reg.predict(xte)
        scores_te[i] = reg.score(xte,yte)

    score_tr = np.mean(scores_tr)
    score_te = np.mean(scores_te)
    slope = np.round(np.mean(slopes),4)
    intercept = np.round(np.mean(intercepts),4)

    ## Plot
    plt.figure()
    plt.scatter(data['x'], data['y'])

    x_reg = np.asarray(sorted(data['x']))
    y_reg = slope*x_reg + intercept
    plt.plot(x_reg,y_reg, c='r', linewidth=2)
    
    plt.title(f"Correlation coefficient for all data = {np.corrcoef(data['x'], data['y'])[0,1]}\nBest fit line on {int(split*100)}% training data: Y = {slope}X + {intercept}")

    ## NOTE: Change the plot labels depending on problem ##
    # For disengagement
    plt.xlabel('Toxicity (average of top 10% of scores)', fontsize=12)
    plt.ylabel(f'Gaps Disengagement ({start_year}-{start_year+2})', fontsize=12)
    # For HIBP
    # plt.xlabel(f'Number of Breaches', fontsize=12)
    # plt.ylabel('Toxicity (average of top 10 scores)', fontsize=12)
    
    plt.savefig(os.path.join(RESULTS_FOLDER, f'regression/{result_name}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    with open(os.path.join(RESULTS_FOLDER, 'regression/log.txt'), 'a') as f:
        f.write(f'\n\n{result_name}:\nMean regression scores in {trials} trials: Training = {score_tr}, Test = {score_te}')


if __name__ == "__main__":
    for start_year in range(1998,2020):
        run_regression(start_year=start_year)
