from src.edge_device import edge_devices
from src.learners import mp
from src.ml import get_dataset
from src.ml import initialize_models
from src.network import random_graph, network_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    args.epochs = 10
    args.iid = 1
    args.unequal = 0
    args.num_users = 100
    args.rounds = 400
    # =================================
    fixed_seed(False)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=True)
    # set up the network topology
    topology = random_graph(models, rho=0.3)
    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=-1)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args, edge=edge)
    graph.show_neighbors()
    # graph.show_similarity(ids=False)

    # Phase I: Local Training
    train_logs = graph.local_training(inference=True)
    # Phase II: Collaborative training
    collab_logs = graph.collaborative_training(learner=mp, args=args)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    logs = {pid: train_logs[pid] + collab_logs[pid] for pid in train_logs.keys()}
    save(f"mp_log_{args.num_users}_{args.epochs}", logs)
    plot_train_history(logs, metric='accuracy', measure="mean-std")
    print("END.")
