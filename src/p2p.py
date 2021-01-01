import time
from copy import deepcopy
from typing import List

import torch

from src.helpers import Map
from src.utils import optimizer_func, log


class Node:

    def __init__(self, k, model, data, neighbors, clustered, similarity, args, params=None):
        self.id = k
        self.model = model
        self.local_model = model
        self.optimizer = None
        self.grads = None
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

    def fit(self, device='cpu'):
        history = []
        self.optimizer = self.params.opt_func(self.model.parameters(), self.params.lr)
        for epoch in range(self.params.epochs):
            for batch in self.train:
                # Train Phase
                loss = self.model.train_step(batch, device)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Validation Phase
            result = self.model.evaluate(self.val, device)
            self.model.epoch_end(epoch, result)
            history.append(result)
            # set local model variable
            self.local_model = self.model
        return history

    def train_one_epoch(self, device='cpu', random=False, evaluate=False):
        """
        Train the model on a random batch of the data
        :return: None
        """
        # train for single batch
        batch = next(iter(self.train))  # train for single batch
        # execute one training step
        self.optimizer.zero_grad()
        loss = self.model.train_step(batch, device)
        loss.backward()
        self.optimizer.step()
        # get gradients
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad.view(-1))
        self.grads = torch.cat(deepcopy(grads))

        return loss, grads

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

    def get_weights(self):
        return deepcopy(self.model.state_dict())

    def set_weights(self, w):
        self.model.load_state_dict(deepcopy(w))

    def get_gradients(self):
        return self.grads

    def set_gradients(self, grads):
        idx = 0
        grads_ = grads.clone()
        for param in self.model.parameters():
            size_layer = len(param.grad.view(-1))
            grads_layer = torch.Tensor(grads_[idx: idx + size_layer]).reshape_as(param.grad).detach()
            param.grad = grads_layer
            idx += size_layer
        self.grads = grads_

    def take_step(self):
        self.model.train()
        self.optimizer.step()
        self.optimizer.zero_grad()

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

    def local_training(self, device='cpu'):
        t = time.time()
        log('event', 'Starting local training ...')
        histories = dict()
        for peer in self.peers:
            log('info', f"{peer} is performing local training on {len(peer.train.dataset)} samples ...")
            histories[peer] = peer.fit(device)
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

    def collaborative_training(self, learner, device):
        t = time.time()
        log('event', f'Starting collaborative training using {learner.name} ...')
        collab_logs = learner.collaborate(self, device)
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

    def show_neighbors(self, verbose=False):
        log('info', "Neighbors list")
        for peer in self.peers:
            log('', f"{peer} has: {len(peer.neighbors)} neighbors.")
            if verbose:
                # log('', f"{peer} neighbors: {peer.neighbors}")
                log('',
                    f"{peer} has: {len(peer.train.dataset)} train samples / {len(peer.val.dataset)} validation samples "
                    f"/ {len(peer.test.dataset)} test samples / {len(peer.inference.dataset)} inference samples")

    def __len__(self):
        return len(self.peers)
