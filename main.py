import os
import time

from src.datasets import get_dataset
from src.learners import p3
from src.models import initialize_models
from src.network import random_graph, network_graph
from src.p2p import Graph
from src.utils import exp_details, load_conf, fixed_seed, log

if __name__ == '__main__':
    # new update coming
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    args.num_users = 300
    args.epochs = 2
    args.batch_size = 16  # 128
    args.iid = 1
    args.unequal = 0
    args.rounds = 500
    # =================================
    fixed_seed(True)

    # Centralized Training ====================================================
    train_logs = Graph.centralized_training(args, inference=True)
    exit()
    # END Centralized Training ================================================

    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=False)

    # set up the network topology
    topology = random_graph(models, sigma=0.2)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args)
    graph.show_neighbors()
    # graph.show_similarity(ids=True)

    # Phase I: Local Training
    train_logs = graph.local_training(inference=True)
    # info = {'xlabel': "Epochs", 'title': "Accuracy. vs. No. of epochs"}
    # plot_train_history(train_logs, metric='accuracy', measure="mean", info=info)

    # Phase II: Collaborative training
    collab_logs = graph.collaborative_training(learner=p3, args=args)
    # info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    # plot_train_history(collab_logs, metric='accuracy', measure="mean")
    print("END.")
    # os._exit(1)
