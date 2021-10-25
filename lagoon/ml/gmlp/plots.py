import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lagoon.ml.config import *


def plot_20211022_final_results() -> None:
    df = pd.read_csv(os.path.join(RESULTS_FOLDER, 'gmlp/20211022/final_results.csv'))

    plt.figure()
    x = df['val_pct_improv_naive']
    y = df['test_pct_improv_naive']
    plt.scatter(x,y)
    plt.xlabel('Best config best validation loss percentage improvement over naive')
    plt.ylabel('Best config test loss percentage improvement over naive')
    plt.title(f'Correlation coefficient = {np.round(np.corrcoef(x,y)[0,1],3)}')
    plt.grid()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'gmlp/20211022/final_results_val_vs_test.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    plot_20211022_final_results()