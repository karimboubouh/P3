from src.datasets import get_dataset
from src.models import initialize_models
from src.network import random_graph, network_graph
from src.learners import avgrad
from src.plots import plot_train_history
from src.utils import exp_details, load_conf

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    # =================================
    args.num_users = 20
    args.epochs = 1
    args.batch_size = 128
    # args.model = 'cnn'
    args.iid = 0
    args.unequal = 1
    args.rounds = 100
    # =================================
    exp_details(args)
    # load dataset and initialize user groups
    train_ds, test_ds, user_groups = get_dataset(args)
    # build users models
    models = initialize_models(args, same=True)
    # setup the network topology
    topology = random_graph(models, sigma=0.4)
    # build the network graph
    graph = network_graph(topology, models, train_ds, test_ds, user_groups, args)
    # graph.show_neighbors(verbose=True)
    graph.show_similarity(ids=True)

    # Phase I: Local Training
    train_logs = graph.local_training(inference=True)
    # info = {'xlabel': "Epochs", 'title': "Accuracy. vs. No. of epochs"}
    # plot_train_history(train_logs, metric='accuracy', measure="mean", info=info)

    # Phase II: Collaborative training
    collab_logs = graph.collaborative_training(learner=avgrad, args=args)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of rounds"}
    plot_train_history(collab_logs, metric='accuracy', measure="mean")
