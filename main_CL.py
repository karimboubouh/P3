from src.p2p import Graph
from src.plots import plot_train_history
from src.utils import load_conf, fixed_seed, save

if __name__ == '__main__':
    # load experiment configuration from CLI arguments
    args = load_conf()
    fixed_seed(True)

    train_logs = Graph.centralized_training(args, inference=True)
    save(f"train_logs_CL", train_logs)
    info = {'xlabel': "Rounds", 'title': "Accuracy. vs. No. of Epochs"}
    plot_train_history(train_logs, metric='accuracy', measure="mean")
    print("END.")
