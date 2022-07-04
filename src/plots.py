import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from src import conf
from src.conf import EVAL_ROUND
from src.helpers import Map
from src.utils import log, verify_metrics, load


def plot_train_history(logs, metric='accuracy', measure="mean", info=None, plot_peer=None):
    if isinstance(logs, str):
        logs = load(logs)
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in logs[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    # plot data
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
    x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    # Plot
    plt.plot(x, data)
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)

    plt.show()


def plot_many(logs, metric='accuracy', measure="mean", info=None):
    logs_0 = load("collab_log_100_0_234.pkl")
    logs_2 = load("collab_log_100_2_108.pkl")
    logs_10 = load("collab_log_100_10_776.pkl")
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    data_0 = np.mean([[v[metric] for v in lo] for lo in logs_0.values()], axis=0)
    data_2 = np.mean([[v[metric] for v in lo] for lo in logs_2.values()], axis=0)
    data_10 = np.mean([[v[metric] for v in lo] for lo in logs_10.values()], axis=0)

    # plot data
    xlabel = 'Number of rounds'
    ylabel = f'Test Accuracy'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    x = range(0, len(data_0) * EVAL_ROUND, EVAL_ROUND)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    plt.plot(x, data_0, label="Skip local step")  # , '-x'
    plt.plot(x, data_2, label="2 local epochs")  # , '-x'
    plt.plot(x, data_10, label="10 local epochs")  # , '-x'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend(loc="lower right", shadow=True)
    plt.show()


def plot_manymore(exps, metric='accuracy', measure="mean", info=None, save=False):
    # Configs
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info is not None:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    colors = ['green', 'blue', 'orange', 'black', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    # colors = ['black', 'green', 'orange', 'blue', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    line_styles = ['-', '--', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':']
    plt.grid(linestyle='dashed')
    plt.rc('legend', fontsize=12)
    plt.xticks(fontsize=13, )
    plt.yticks(fontsize=13, )
    std_data = None
    for i, exp in enumerate(exps):
        # Data
        logs = load(exp['file'])
        name = exp.get('name', "")
        if measure == "mean":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "mean-std":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
            std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "max":
            data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, data, color=colors[i], label=name, linestyle=line_styles[i])
        if std_data is not None:
            plt.fill_between(x, data - std_data, data + std_data, color=colors[i], alpha=.1)

    plt.legend(loc="lower right", shadow=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title(title)
    if save:
        unique = np.random.randint(100, 999)
        plt.savefig(f"../out/EXP_{unique}.pdf")
    plt.show()


if __name__ == '__main__':
    plot_manymore([
        {'file': "PT_P3_U_100_0_500.pkl", 'name': "P3, $e = 0$"},
        {'file': "PT_P3_U_100_2_500.pkl", 'name': "P3, $e = 2$"},
        {'file': "PT_P3_U_100_10_500.pkl", 'name': "P3, $e = 10$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy over all peers", 'title': ""}, save=True)


"""
-------------------------------------------------
EXP: 1
-------------------------------------------------
    plot_manymore([
        {'file': "PT_P3_IID_100_0_500.pkl", 'name': "P3, $e = 0$"},
        {'file': "PT_P3_IID_100_2_500.pkl", 'name': "P3, $e = 2$"},
        {'file': "PT_P3_IID_100_10_500.pkl", 'name': "P3, $e = 10$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy over all peers", 'title': ""}, save=True)
"""

"""
-------------------------------------------------
EXP: 2
-------------------------------------------------
    plot_manymore([
        {'file': "PT_P3_NIID_100_0_500.pkl", 'name': "P3, $e = 0$"},
        {'file': "PT_P3_NIID_100_2_500.pkl", 'name': "P3, $e = 2$"},
        {'file': "PT_P3_NIID_100_10_500.pkl", 'name': "P3, $e = 10$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy over all peers", 'title': ""}, save=True)
"""

"""
-------------------------------------------------
EXP: 3
-------------------------------------------------
    plot_manymore([
        {'file': "PT_P3_U_100_0_500.pkl", 'name': "P3, $e = 0$"},
        {'file': "PT_P3_U_100_2_500.pkl", 'name': "P3, $e = 2$"},
        {'file': "PT_P3_U_100_10_500.pkl", 'name': "P3, $e = 10$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy over all peers", 'title': ""}, save=True)
"""

"""
-------------------------------------------------
EXP: 4
-------------------------------------------------
    plot_manymore([
        {'file': "PT_P3_IID_100_2_500.pkl", 'name': "P3$_{\\bf{Server}}$ (all peers)"},
        {'file': "PT_P3_MOB_100_2_500.pkl", 'name': "P3$_{\\bf{Mobile}}$ (one device)"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy", 'title': ""}, save=True)
"""

"""
-------------------------------------------------
EXP: 5
-------------------------------------------------
    plot_manymore([
        {'file': "PT_P3_IID_100_2_500.pkl", 'name': r"P3 $_{(P2P)}$"},
        {'file': "PT_MP_IID_100_10_500.pkl", 'name': r"MP $_{(P2P)}$"},
        {'file': "PT_FL_IID_100_1_500.pkl", 'name': r"FedAvg $_{(FL)}$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy", 'title': ""}, save=True)
"""

"""
-------------------------------------------------
EXP: 6
-------------------------------------------------
    plot_manymore([
        {'file': "MP_100_2_W_0.pkl", 'name': r"$\rho = 1$"},
        {'file': "MP_100_2_W_0.33.pkl", 'name': r"$\rho = 0.6$"},
        {'file': "MP_100_2_W_0.66.pkl", 'name': r"$\rho = 0.3$"},
        {'file': "MP_100_2_W_0.95.pkl", 'name': r"$\rho = 0.05$"},
        {'file': "MP_100_2_W_0.99.pkl", 'name': r"$\rho = 0.01$"},
    ], metric="accuracy", measure="mean-std",
        info={'xlabel': "iterations", 'ylabel': "Test Accuracy", 'title': ""}, save=True)
"""
