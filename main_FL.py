from src.edge_device import edge_devices
from src.learners import fedavg
from src.ml import get_dataset
from src.ml import initialize_models
from src.network import network_graph, central_graph
from src.plots import plot_train_history
from src.utils import exp_details, load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.mp = 0
    args.epochs = 1
    args.iid = 0
    args.unequal = 0
    args.num_users = 100
    args.rounds = 500
    # =================================
    fixed_seed(False)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=True)
    topology = central_graph(models)
    # include physical edge devices  (count < 1 to only use simulated nodes)
    edge = edge_devices(args, count=-1)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args, edge=edge)

    train_logs = graph.collaborative_training(learner=fedavg, args=args)
    save(f"fl_logs_{args.num_users}", train_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(train_logs, metric='accuracy', measure="mean")
    print("END.")
