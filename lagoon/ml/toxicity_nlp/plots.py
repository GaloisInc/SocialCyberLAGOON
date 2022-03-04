from lagoon.ml.config import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plot_scores_badwords(aggr = np.mean):
    """
    aggr: Some aggregation function like mean or max
    TODO (maybe): Violin plots: https://altair-viz.github.io/gallery/violin_plot.html
    """
    with open(os.path.join(DATA_FOLDER, 'toxicity_nlp/scores_badwords.pkl'), 'rb') as f:
        result = pickle.load(f)
    result_clf = result['clf']
    result_reg = result['reg']

    for key in result_clf.keys():
        result_clf[key] = aggr(result_clf[key])
        result_reg[key] = aggr(result_reg[key])

    # Sort keys prior to plotting
    result_clf = {k:result_clf[k] for k in sorted(list(result_clf.keys()), key = lambda elem: int(elem.replace('+','-').split('-')[0]))}
    result_reg = {k:result_reg[k] for k in sorted(list(result_reg.keys()), key = lambda elem: int(elem.replace('+','-').split('-')[0]))}

    fontsize = 16
    output_suffix = '_mean' if aggr == np.mean else '_max' if aggr == np.max else ''

    plt.figure(figsize=(12,9))
    plt.plot(result_clf.keys(),result_clf.values(), linewidth=2)
    plt.xlabel('Number of badwords', fontsize=fontsize)
    plt.ylabel('Toxicity NLP classification score', fontsize=fontsize)
    plt.xticks(rotation=60, fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.grid()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'toxicity_nlp/classification_vs_badwords{output_suffix}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.figure(figsize=(12,9))
    plt.plot(result_reg.keys(),result_reg.values(), linewidth=2)
    plt.xlabel('Number of badwords', fontsize=fontsize)
    plt.ylabel('Toxicity NLP regression score', fontsize=fontsize)
    plt.xticks(rotation=60, fontsize=fontsize)
    plt.yticks(fontsize=fontsize-2)
    plt.grid()
    plt.savefig(os.path.join(RESULTS_FOLDER, f'toxicity_nlp/regression_vs_badwords{output_suffix}.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    plot_scores_badwords()
