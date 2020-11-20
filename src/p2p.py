import time
from typing import List

import torch
from termcolor import cprint

from src.helpers import Map
from src.utils import optimizer_func, log


class Node:

    def __init__(self, k, model, data, neighbors, clustered, similarity, args, params=None):
        self.id = k
        self.model = model
        self.local_model = model
        self.neighbors = neighbors
        self.clustered = clustered
        self.similarity = similarity
        self.train = data.get('train', None)
        self.val = data.get('val', None)
        self.test = data.get('test', None)
        self.inference = data.get('inference', None)
        # default params
        self.params = Map({
            'rounds': args.rounds,
            'epochs': args.epochs,
            'lr': args.lr,
            'opt_func': optimizer_func(args.optimizer),
            'confidence': 1,
            'alpha': 0.9
        })
        # override params if provided
        if isinstance(params, Map):
            self.params = Map(dict(self.params, **params))

    def fit(self):
        history = []
        optimizer = self.params.opt_func(self.model.parameters(), self.params.lr)
        for epoch in range(self.params.epochs):
            for batch in self.train:
                # Train Phase
                loss = self.model.train_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation Phase
            result = self.model.evaluate(self.val)
            self.model.epoch_end(epoch, result)
            history.append(result)
            # set local model variable
            self.local_model = self.model
        return history

    def evaluate(self, dataloader):
        return self.model.evaluate(dataloader)

    def save_model(self):
        pass

    def get_neighbors(self):
        return self.neighbors

    def set_neighbors(self, neighbors, similarity=None):
        self.neighbors = neighbors
        if similarity:
            self.similarity = similarity

    def reset_neighbors(self, nodes, similarity):
        pass

    #  Private methods --------------------------------------------------------

    def _eval_sample(self, sample):
        pass

    # Special methods
    def __repr__(self):
        return f"Node({self.id})"

    def __str__(self):
        return f"Node({self.id})"


class Graph:

    def __init__(self, peers, topology, test_ds, args):
        self.peers = peers  # type: List[Node]
        self.clusters = topology['clusters']
        self.similarity = topology['similarities']
        self.adjacency = topology['adjacency']
        self.test_ds = test_ds
        self.args = args

    def local_training(self):
        t = time.time()
        log('event', 'Starting local training ...')
        histories = dict()
        for peer in self.peers:
            log('info', f"{peer} is performing local training ...")
            histories[peer] = peer.fit()
        t = time.time() - t
        log("success", f"Local training finished in {t:.2f} seconds.")

        return histories

    def evaluation(self, dataset="test"):
        if dataset not in ["train", "val", "test", "inference"]:
            log("warning", f" unsupported dataset type, fallback to: test")
            dataset = "test"
        history = {}
        for peer in self.peers:
            if dataset == "train":
                history[peer.id] = peer.evaluate(peer.train)
            elif dataset == "val":
                history[peer.id] = peer.evaluate(peer.val)
            elif dataset == "test":
                history[peer.id] = peer.evaluate(peer.test)
            elif dataset == "inference":
                history[peer.id] = peer.evaluate(peer.inference)

        return history

    def collaborative_training(self, learner):
        t = time.time()
        log('event', f'Starting collaborative training using {learner.name} ...')
        collab_logs = learner.collaborate(self)
        t = time.time() - t
        log("success", f"Collaborative training finished in {t:.2f} seconds.")
        return collab_logs

    def get_peers(self):
        return self.peers

    def local_score(self, metric='accuracy', measure='mean'):
        if scope == 'local' and metric == 'accuracy':
            accuracy_epochs()

    def plot_local_score(self, metric='accuracy', measure='mean'):
        if scope == 'local' and metric == 'accuracy':
            accuracy_epochs()

    def show_similarity(self, matrix=False):
        log('info', "Similarity Matrix")
        if matrix:
            log('', self.similarity)
        else:
            for peer in self.peers:
                s = {k: round(v, 2) for k, v in peer.similarity.items()}
                log('', f"{peer}: {s}")

    def show_neighbors(self):
        log('info', "Neighbors list")
        for peer in self.peers:
            log('', f"{peer}: {peer.neighbors}")

    def __len__(self):
        return len(self.peers)
