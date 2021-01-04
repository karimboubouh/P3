import time
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

from src.conf import RECORD_RATE
from src.p2p import Graph
from src.utils import log, inference_eval

name = "Gradient Averaging Collaborative Learner"


def collaborate(graph: Graph, device='cpu'):
    # initialize history holder
    history = dict.fromkeys(range(len(graph.peers)))
    for k in history.keys():
        history[k] = []

    # setup algorithm parameters
    for peer in graph.peers:
        peer.params.gradients = []

    # prepare tqdm
    log("info", f"Collaborative training for {graph.args.rounds} rounds")
    rounds = tqdm(range(graph.args.rounds))
    for epoch in rounds:
        # Randomly activate a peer
        peer = np.random.choice(graph.peers)
        if peer.clustered:
            # exchange with peer's cluster
            peer.params.gradients = []
            for neighbor in peer.neighbors:
                # Todo run in parallel
                # run one8 training epoch per neighbor
                ngrads = neighbor.train_one_epoch(device)
                peer.params.gradients.append(ngrads)
            # Update model
            average_weights(peer)
        else:
            # Randomly select a neighbor
            neighbor = np.random.choice(peer.neighbors)
            # run one training epoch
            # Todo run in parallel
            peer.train_one_epoch(device)
            neighbor.train_one_epoch(device)
            # Exchange gradients
            peer.params.gradients = [neighbor.get_gradients()]
            neighbor.params.gradients = [peer.get_gradients()]
            # average grads an take a step
            avg_step(peer)
            avg_step(neighbor)

        # Evaluate all models every RECORD_RATE
        if epoch != 0 and epoch % RECORD_RATE == 0:
            run_evaluation(graph, history, epoch)
            rounds.set_postfix({**{'peer': peer}, **history[peer.id][-1]})

    log("info", f"Evaluating the output of the collaborative training after {graph.args.rounds} rounds.")
    for peer in graph.peers:
        inference_eval(peer, device)

    return history


def avg_step(peer):
    average_grads = average_gradients(peer)
    peer.set_gradients(average_grads)
    peer.take_step()


def average_gradients(peer):
    grads = [peer.get_gradients()] + peer.params.gradients
    return torch.mean(torch.stack(grads), dim=0)


def average_weights(peer):
    """Returns the average of the weights."""
    wi = deepcopy(peer.model.state_dict())
    w = list()
    for m in peer.params.models:
        w.append(deepcopy(m.state_dict()))
    # average weights per channel
    for key in wi.keys():
        for wj in w:
            wi[key] += wj[key]
        wi[key] = torch.div(wi[key], len(w) + 1)
    # update calculated weights
    peer.model.load_state_dict(wi)

    return wi


def run_evaluation(graph, history, epoch, debug=True):
    t = time.time()
    current = []
    for peer in graph.peers:
        r = peer.model.evaluate(peer.inference, one_batch=True)
        history[peer.id].append(r)
        current.append(r)
    if debug:
        current_los = round(np.mean([e['val_loss'] for e in current]), 2)
        current_acc = round(np.mean([e['val_acc'] for e in current]), 2)
        t = round(time.time() - t, 2)
        log('', f"\nEvaluation after {epoch} rounds: mean accuracy: {current_acc} | mean loss {current_los}. ({t}s)\n")

    return history


def log_round(peer, epoch, history):
    peer_loss = history[peer.id][-1]['val_loss']
    peer_acc = history[peer.id][-1]['val_acc']
    log('', f"Round [{epoch}], {peer}, loss: {peer_loss:.4f}, val_acc: {peer_acc:.4f}")


def old_average_weights(peer, data):
    """
    Returns the average of the weights.
    """
    wi = peer.model.state_dict()
    w = []
    # list of models weights
    for d in data:
        w.append(d['model'].state_dict())
    # average weights per channel
    for key in wi.keys():
        for wj in w:
            wi[key] += wj[key]
        wi[key] = torch.div(wi[key], len(w) + 1)
    # update calculated weights
    peer.model.load_state_dict(wi)

    return peer, wi
