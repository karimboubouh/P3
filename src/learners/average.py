import torch
import numpy as np
from tqdm import tqdm
from threading import Thread

from src.conf import RECORD_RATE
from src.utils import log

name = "Average-based Collaborative Learner"


def collaborate(graph):
    # initialize history holder
    history = dict.fromkeys(range(len(graph.peers)))
    for k in history.keys():
        history[k] = []
    # prepare tqdm
    rounds = tqdm(range(graph.args.rounds))
    log("info", f"Collaborative training for {graph.args.rounds} rounds")
    for epoch in rounds:
        # Randomly activate a peer
        peer = np.random.choice(graph.peers)
        send = [{'model': peer.model}]
        if peer.clustered:
            # exchange with peer's cluster
            targets = peer.neighbors
            receive = list()
            for member in peer.neighbors:
                receive.append({'model': member.model})
        else:
            # exchange with a randomly selected neighbor
            neighbor = np.random.choice(peer.neighbors)
            targets = [neighbor]
            receive = [{'model': neighbor.model}]

        # Exchange data and perform computation
        threads = []
        t = Thread(target=average_weights, args=(peer, receive), daemon=True)
        threads.append(t)
        t.start()
        # Run weights updates for selected peers
        for target in targets:
            t = Thread(target=average_weights, args=(target, send), daemon=True)
            threads.append(t)
            t.start()

        # Wait for weights update
        for th in threads:
            th.join()

        # evaluate all models every RECORD_RATE
        if epoch % RECORD_RATE == 0:
            run_evaluation(graph, history)
            rounds.set_postfix({**{'peer': peer}, **history[peer.id][-1]})

    for peer in graph.peers:
        print(peer, peer.model.evaluate(peer.test))

    return history


def average_weights(peer, data):
    """
    Returns the average of the weights.
    """
    wi = peer.model.state_dict()
    w = list()
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


def run_evaluation(graph, history):
    for peer in graph.peers:
        r = peer.model.evaluate(peer.test)
        history[peer.id].append(r)

    return history


def log_round(peer, epoch, history):
    peer_loss = history[peer.id][-1]['val_loss']
    peer_acc = history[peer.id][-1]['val_acc']
    log('', f"Round [{epoch}], {peer}, loss: {peer_loss:.4f}, val_acc: {peer_acc:.4f}")
