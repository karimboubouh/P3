import numpy as np
import matplotlib.pyplot as plt

from src import conf
from src.conf import EVAL_ROUND
from src.helpers import Map
from src.utils import log, verify_metrics


def plot_train_history(logs, metric='accuracy', measure="mean", info=None):
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    # plot data
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    plt.plot(x, data)  # , '-x'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def accuracy_epochs(history, measure="mean", xlabel="Epochs", ylabel="Accuracy", title=None):
    if measure == 'mean':
        print(history)

    exit()


# plt.plot(x, '-x')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title('Accuracy vs. No. of epochs')
# plt.show()


if __name__ == '__main__':
    args = Map()
    args.verbose = 10
    x = [1, 2, 3, 4, 5]
