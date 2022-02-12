import os
import time

from src.edge_device import edge_devices
from src.ml import get_dataset
from src.learners import p3
from src.ml import initialize_models
from src.network import random_graph, network_graph
from src.p2p import Graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, log, save, load

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    # 10 (sigma=0.4) // 100 (sigma=0.9) // 300 (sigma=0.95)
    args.num_users = 100
    args.epochs = 10
    args.batch_size = 128
    args.iid = 1
    args.unequal = 0
    args.rounds = 500
    # =================================
    fixed_seed(True)

    # Centralized Training ====================================================
    # train_logs = Graph.centralized_training(args, inference=True)
    # exit()
    # END Centralized Training ================================================

    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=True)
    # set up the network topology
    topology = random_graph(models, sigma=0.9)
    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=-1)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args, edge=edge)
    graph.show_neighbors()
    # graph.show_similarity(ids=True)

    # Phase I: Local Training
    train_logs = graph.local_training(inference=False)

    # Phase II: Collaborative training
    collab_logs = graph.collaborative_training(learner=p3, args=args)
    save(f"collab_log_{args.num_users}_{args.epochs}", collab_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(collab_logs, metric='accuracy', measure="mean")
    print("END.")
