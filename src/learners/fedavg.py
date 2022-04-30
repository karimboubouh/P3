import time

from tqdm import tqdm

from src import protocol
from src.conf import EVAL_ROUND, WAIT_TIMEOUT, WAIT_INTERVAL
from src.ml import model_inference, train_for_x_epoch, GAR
from src.p2p import Graph, Node
from src.utils import log, wait_until

name = "Federated averaging (FedAvg)"


def collaborate(graph: Graph, args):
    args.server_id = len(graph.peers) - 1
    log("info", f"Initializing Model Propagation...")
    # init peers parameters
    for peer in graph.peers:
        peer.execute(train_init, args)
    graph.join()

    log("info", f"Collaborative training for T = {graph.args.rounds} rounds")
    T = tqdm(range(graph.args.rounds), position=0)
    for t in T:
        for peer in graph.peers:
            peer.execute(train_step, t, args)
        graph.join(t)

    # stop train
    log("info", f"Evaluating the output of the collaborative training.")
    for peer in graph.peers:
        peer.execute(train_stop, args)
    graph.join()
    log('info', f"Graph G disconnected.")

    # get collaboration logs
    server = [peer for peer in graph.peers if peer.id == args.server_id][0]
    collab_logs = {server.id: server.params.logs}

    return collab_logs


def train_init(peer: Node, args):
    peer.params.exchanges = 0
    if peer.id == args.server_id:
        # server:
        r = peer.evaluate(peer.inference, one_batch=True)
        peer.params.logs = [r]
        peer.params.models = {i: [] for i in range(args.rounds)}

    return


def train_step(peer: Node, t, args):
    T = t if isinstance(t, tqdm) or isinstance(t, range) else [t]
    for t in T:
        if peer.id == args.server_id:
            # Server
            wait_until(enough_received, WAIT_TIMEOUT * 100, WAIT_INTERVAL * 10, peer, t, len(peer.neighbors))
            w = GAR(peer, [v for i, v in peer.V[t]])
            msg = protocol.train_step(t, peer.get_model_params())  # not grads
            peer.broadcast(msg)
            peer.set_model_params(w)
            if t % EVAL_ROUND == 0:
                t_eval = peer.evaluate(peer.inference, one_batch=True)
                peer.params.logs.append(t_eval)
        else:
            if t > 0:
                wait_until(server_received, WAIT_TIMEOUT * 100, WAIT_INTERVAL * 10, peer, t)
                w_server = peer.V[t - 1][0][1]
                peer.set_model_params(w_server)
            # Worker
            train_for_x_epoch(peer, args.epochs)
            msg = protocol.train_step(t, peer.get_model_params())  # not grads
            server = peer.neighbors[0]
            peer.send(server, msg)
            # peer.params.server.params.models[t].append(peer.get_model_params())
    return


def train_stop(peer: Node, args):
    if peer.id == args.server_id:
        model_inference(peer, one_batch=True)
    peer.stop()


# ---------- Helper functions -------------------------------------------------

def enough_received(peer: Node, t, size):
    if t in peer.V and len(peer.V[t]) >= size:
        return True
    return False


def server_received(peer: Node, t):
    if t - 1 in peer.V and len(peer.V[t - 1]) == 1:
        return True
    return False
