from __future__ import annotations

import pickle
import socket
import struct
import time
import traceback
from copy import deepcopy
from threading import Thread
from typing import List

import numpy as np
import torch

from src import protocol
from src.conf import HOST, PORT, SOCK_TIMEOUT, TCP_SOCKET_SERVER_LISTEN
from src.datasets import get_dataset
from src.helpers import Map
from src.measure_energy import measure_energy
from src.models import initialize_models
from src.profiler import profiler
from src.utils import optimizer_func, log, inference_eval, inference_ds, create_tcp_socket, train_val_test


class Node(Thread):

    def __init__(self, k, model, data, neighbors_ids, clustered, similarity, args: Map, params=None):
        super(Node, self).__init__()
        self.id = k
        self.mp = bool(args.mp)
        self.host = HOST
        self.port = PORT + k
        self.device = args.device
        self.model = model
        self.local_model = model
        self.optimizer = None
        self.grads = None
        self.V = {}
        self.current_round = 0
        self.current_exec = None
        self.neighbors_ids = neighbors_ids
        self.neighbors = []
        self.in_neighbors = []
        self.clustered = clustered
        self.similarity = similarity
        self.train = data.get('train', None)
        self.val = data.get('val', None)
        self.test = data.get('test', None)
        self.inference = data.get('inference', None)
        self.terminate = False
        # default params
        self.params = Map({
            'frac': args.frac,
            'epochs': args.epochs,
            'lr': args.lr,
            'momentum': args.momentum,
            'opt_func': optimizer_func(args.optimizer),
            'gar': args.gar,
            'D': sum(self.similarity.values()),
            'confidence': 1,
            'alpha': 0.9,
        })
        # override params if provided
        if isinstance(params, Map):
            self.params = Map(dict(self.params, **params))
        # initialize networks
        self._init_server()

    def run(self):
        while not self.terminate:
            try:
                conn, address = self.sock.accept()
                if not self.terminate:
                    neighbor_conn = NodeConnection(self, address[1], conn)
                    neighbor_conn.start()
                    self.neighbors.append(neighbor_conn)
                    # self.in_neighbors.append(in_neighbor_conn)
            except socket.timeout:
                pass
            except Exception as e:
                log('error', f"{self}: Node Exception\n{e}")

        for neighbor in self.neighbors:
            neighbor.stop()
        self.sock.close()
        log('log', f"{self}: Stopped")

    def connect(self, neighbor: Node):
        try:
            if neighbor.id in [n.neighbor_id for n in self.neighbors]:
                log('log', f"{self}, neighbor {neighbor} already connected.")
                return True
            if self.mp:
                sock = create_tcp_socket()
                sock.settimeout(SOCK_TIMEOUT)
                sock.connect((neighbor.host, neighbor.port))
                neighbor_conn = NodeConnection(self, neighbor.id, sock)
                neighbor_conn.start()
                neighbor_conn.send(protocol.connect(sock.getsockname(), self.id))
                self.neighbors.append(neighbor_conn)
            else:
                slink = NodeLink(self, neighbor, None)
                dlink = NodeLink(neighbor, self, slink)
                slink.link = dlink
                self.neighbors.append(slink)
                neighbor.neighbors.append(dlink)

            return True
        except Exception as e:
            log('error', f"{self}: Can't connect to {neighbor} -- {e}")
            return False

    def disconnect(self, neighbor_conn: NodeConnection):
        if not neighbor_conn.terminate:
            neighbor_conn.send(protocol.disconnect(self.id))
            neighbor_conn.terminate = True
            if neighbor_conn in self.neighbors:
                self.neighbors.remove(neighbor_conn)
            log('log', f"{self} disconnected from {neighbor_conn.neighbor_name}")

    def stop(self):
        for neighbor in self.neighbors:
            self.disconnect(neighbor)
        self.terminate = True

    def send(self, neighbor, msg):
        neighbor.send(msg)

    def broadcast(self, msg, active=None):
        active = self.neighbors if active is None else active
        for neighbor in active:
            self.send(neighbor, msg)

    def execute(self, func, *args):
        try:
            self.current_exec = Thread(target=func, args=(self, *args), name=func.__name__)
            # self.current_exec = Process(target=func, args=(self, *args), name=func.__name__)
            # self.current_exec.daemon = True
            self.current_exec.start()
        except Exception as e:
            log('error', f"{self} Execute exception: {e}")
            traceback.print_exc()
            return None

    def fit(self, inference=True):
        history = []
        self.optimizer = self.params.opt_func(self.model.parameters(), self.params.lr)  # , 0.99
        for epoch in range(self.params.epochs):
            for batch in self.train:
                # Train Phase
                loss = self.model.train_step(batch, self.device)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Validation Phase
            result = self.model.evaluate(self.val, self.device)
            self.model.epoch_end(epoch, result)
            history.append(result)
        if inference:
            # evaluate against a batch of the inference dataset
            inference_eval(self)
        # set local model variable
        self.local_model = self.model
        return history

    def train_one_epoch(self, batches=1, evaluate=False):
        for i in range(batches):
            # train for single batch randomly chosen when Dataloader is set with shuffle=True
            batch = next(iter(self.train))
            # execute one training step
            if self.optimizer:
                self.optimizer.zero_grad()
            else:
                self.optimizer = self.params.opt_func(self.model.parameters(), self.params.lr)
            loss = self.model.train_step(batch, self.device)
            loss.backward()
            # TODO new verify
            self.optimizer.step()
            self.optimizer.zero_grad()
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
        grads_ = grads.clone().cpu()
        for param in self.model.parameters():
            size_layer = len(param.grad.view(-1))
            grads_layer = torch.Tensor(grads_[idx: idx + size_layer]).reshape_as(param.grad).detach().to(self.device)
            param.grad = grads_layer
            idx += size_layer
        self.grads = grads_.to(self.device)

    def take_step(self):
        self.model.train()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def local_data_scope(self):
        batch = next(iter(self.train))
        scope = set(batch[1].numpy())
        return list(scope)

    def neighborhood_data_scope(self):
        scope = set(self.local_data_scope())
        for neighbor in self.neighbors:
            scope = scope.union(neighbor.local_data_scope())
        return list(scope)

    #  Private methods --------------------------------------------------------

    def _eval_sample(self, sample):
        pass

    def _init_server(self):
        self.sock = create_tcp_socket()
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(SOCK_TIMEOUT)
        self.sock.listen(TCP_SOCKET_SERVER_LISTEN)

    # Special methods
    def __repr__(self):
        return f"Node({self.id})"

    def __str__(self):
        return f"Node({self.id})"


class NodeConnection(Thread):
    def __init__(self, node, neighbor_id, sock):
        super(NodeConnection, self).__init__()
        self.node = node
        self.sock = sock
        self.address = None
        self.neighbor_id = neighbor_id
        self.neighbor_name = f"Node({neighbor_id})"
        self.terminate = False

    def run(self):
        # Wait for messages from device
        while not self.terminate:
            try:
                (length,) = struct.unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(4096 if to_read > 4096 else to_read)
                if buffer:
                    data = pickle.loads(buffer)
                    if data and data['mtype'] == protocol.TRAIN_STEP:
                        self.handle_step(data['data'])
                    elif data and data['mtype'] == protocol.CONNECT:
                        self.handle_connect(data['data'])
                    elif data and data['mtype'] == protocol.DISCONNECT:
                        self.handle_disconnect(data['data'])
                    else:
                        log('error', f"{self.node.name}: Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                log('error', f"{self.node}: Corrupted message : {e}")
            except socket.timeout:
                pass
            except struct.error as e:
                pass
            except Exception as e:
                self.terminate = True
                # todo remove node from list of connected neighbors
                traceback.print_exc()
                log('error', f"{self.node} NodeConnection <{self.neighbor_name}> Exception\n{e}")
        self.sock.close()
        log('log', f"{self.node}: neighbor {self.neighbor_name} disconnected")

    def send(self, msg):
        try:
            if self.terminate:
                log('log', f"{self} tries to send on terminated")
            length = struct.pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            log('error', f"{self}: Socket error: {e}: ")
        except Exception as e:
            log('error', f"{self}: Exception\n{e}")

    def stop(self):
        self.terminate = True

    def handle_step(self, data):
        self.node.params.exchanges += 1
        if self.node.current_round <= data['t']:
            if data['t'] in self.node.V:
                self.node.V[data['t']].append((self.neighbor_id, data['update']))
            else:
                self.node.V[data['t']] = [(self.neighbor_id, data['update'])]

    def handle_connect(self, data):
        self.neighbor_id = data['id']
        self.address = data['address']

    def handle_disconnect(self, data):
        self.terminate = True
        if self in self.node.neighbors:
            self.node.neighbors.remove(self)

    #  Private methods --------------------------------------------------------

    def __repr__(self):
        return f"NodeConn({self.node.id}, {self.neighbor_id})"

    def __str__(self):
        return f"NodeConn({self.node.id}, {self.neighbor_id})"


class NodeLink:
    def __init__(self, node: Node, neighbor: Node, link: NodeLink = None):
        self.node = node
        self.neighbor = neighbor
        self.link = link
        # kept for compatibility with NodeConnection
        self.terminate = False
        self.neighbor_id = neighbor.id
        self.neighbor_name = str(neighbor)

    def send(self, msg):
        if msg:
            data = pickle.loads(msg)
            if data and data['mtype'] == protocol.TRAIN_STEP:
                self.link.handle_step(data['data'])
            elif data and data['mtype'] == protocol.CONNECT:
                self.link.handle_connect(data['data'])
            elif data and data['mtype'] == protocol.DISCONNECT:
                self.link.handle_disconnect(data['data'])
            else:
                log('error', f"{self.node.name}: Unknown type of message: {data['mtype']}.")
        else:
            log('error', f"{self.node.name}: Corrupted message.")

    def handle_step(self, data):
        self.node.params.exchanges += 1
        if self.node.current_round <= data['t']:
            if data['t'] in self.node.V:
                self.node.V[data['t']].append((self.neighbor_id, data['update']))
            else:
                self.node.V[data['t']] = [(self.neighbor_id, data['update'])]

    def handle_connect(self, data):
        self.neighbor_id = data['id']

    def handle_disconnect(self, data):
        self.terminate = True
        if self in self.node.neighbors:
            self.node.neighbors.remove(self)

    def stop(self):
        self.terminate = True

    #  Private methods --------------------------------------------------------

    def __repr__(self):
        return f"NodeLink({self.node.id}, {self.neighbor_id})"

    def __str__(self):
        return f"NodeLink({self.node.id}, {self.neighbor_id})"


class Graph:

    def __init__(self, peers, topology, test_ds, args):
        self.device = args.device
        self.peers = peers  # type: List[Node]
        self.clusters = topology['clusters']
        self.similarity = topology['similarities']
        self.adjacency = topology['adjacency']
        self.test_ds = test_ds
        self.args = args

    @staticmethod
    def centralized_training(args, inference=True):
        t = time.time()
        log('event', 'Centralized training ...')
        args.num_users = 1
        args.iid = 1
        args.unequal = 0
        args.iid = 1
        args.rounds = 0
        log('info', f"Loading {args.dataset} dataset")
        train_ds, test_ds, user_groups = get_dataset(args)
        train, val, test = train_val_test(train_ds, user_groups[0], args)
        data = {'train': train, 'val': val, 'test': test, 'inference': test_ds}
        log('info', f"Initializing {args.model} model.")
        models = initialize_models(args, same=True)
        server = Node(0, models[0], data, [], False, {}, args)
        server.inference = inference_ds(server, args)
        log('info', f"Start server training on {len(server.train.dataset)} samples ...")
        history = server.fit(inference)

        for name, param in server.model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n{param.data}")

        server.stop()
        t = time.time() - t
        log("success", f"Centralized training finished in {t:.2f} seconds.")

        return [history]

    # @measure_energy
    # @profiler
    def local_training(self, device='cpu', inference=True):
        t = time.time()
        log('event', 'Starting local training ...')
        histories = dict()
        for peer in self.peers:
            classes = []
            for b in peer.train:
                classes.extend(b[1].numpy())
            log('info',
                f"{peer} is performing local training on {len(peer.train.dataset)} samples of classes {set(classes)}.")
            histories[peer] = peer.fit(inference)
            # peer.stop()
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

    # @measure_energy
    # @profiler
    def collaborative_training(self, learner, args):
        t = time.time()
        log('event', f'Starting collaborative training using {learner.name} ...')
        collab_logs = learner.collaborate(self, args)
        t = time.time() - t
        log("success", f"Collaborative training finished in {t:.2f} seconds.")
        return collab_logs

    def join(self, r=None):
        t = time.time()
        name = self.peers[0].current_exec.name
        for peer in self.peers:
            if peer.current_exec is not None:
                peer.current_exec.join()
                del peer.current_exec
                peer.current_exec = None
        t = time.time() - t
        if r is not None:
            log("log", f"Round {r}: {name} joined in {t:.2f} seconds.")
        else:
            log("success", f"{name} joined in {t:.2f} seconds.")

    def get_peers(self):
        return self.peers

    def local_score(self, metric='accuracy', measure='mean'):
        if scope == 'local' and metric == 'accuracy':
            accuracy_epochs()

    def plot_local_score(self, metric='accuracy', measure='mean'):
        if scope == 'local' and metric == 'accuracy':
            accuracy_epochs()

    def show_similarity(self, ids=False, matrix=False):
        log('info', "Similarity Matrix")
        if matrix:
            print(self.similarity)
        else:
            for peer in self.peers:
                if ids:
                    s = list(peer.similarity.keys())
                else:
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

                iterator = iter(peer.train)
                x_batch, y_batch = iterator.next()
                log('', f"{peer} has: [{len(peer.train.dataset)}] {set(y_batch.numpy())}")
                print()

    def set_inference(self, args):
        for peer in self.peers:
            peer.inference = inference_ds(peer, args)

    def PSS(self, peer: Node, k):
        nid = [n.neighbor_id for n in peer.neighbors]
        nid.append(peer.id)
        candidates = [p for p in self.peers if p.id not in nid]
        k = min(k, len(candidates))
        return np.random.choice(candidates, k, replace=False)

    def __len__(self):
        return len(self.peers)


class IONode:

    def __init__(self, node: Node):
        # TODO reduce deepcopy instructions
        self.node = node
        self.id = node.id
        self.host = node.host
        self.port = node.host
        self.device = node.device
        self.local_model = deepcopy(node.local_model)
        self.optimizer = deepcopy(node.optimizer)
        self.grads = deepcopy(node.grads)
        self.model = deepcopy(node.model)
        self.V = node.V
        self.current_round = node.current_round
        self.current_exec = node.current_exec
        self.neighbors_ids = node.neighbors_ids
        self.neighbors = node.neighbors
        self.in_neighbors = node.neighbors
        self.clustered = node.clustered
        self.similarity = node.similarity
        self.train = node.train
        self.val = node.val
        self.test = node.test
        self.inference = node.inference
        self.terminate = node.terminate
        self.params = node.params

    def train_one_epoch(self, batches=1, evaluate=False):
        return self.node.train_one_epoch(batches=batches, evaluate=evaluate)

    def connect(self, neighbor: Node):
        return self.node.connect(neighbor=neighbor)

    def disconnect(self, neighbor_conn: NodeConnection):
        self.node.disconnect(neighbor_conn=neighbor_conn)

    def broadcast(self, msg, active=None):
        return self.node.broadcast(msg=msg, active=active)

    def stop(self):
        self.node.stop()

    # Special methods
    def __repr__(self):
        return f"IONode({self.id})"

    def __str__(self):
        return f"IONode({self.id})"
