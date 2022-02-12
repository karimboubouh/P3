import pickle
import socket
import struct
import traceback
from copy import deepcopy
from importlib import import_module
from threading import Thread

from kivymd.toast import toast

from src import protocol
from src.conf import SOCK_TIMEOUT, TCP_SOCKET_SERVER_LISTEN, ALGORITHM_MODULE
from src.ml import model_fit, model_inference, train_for_x_epoch, evaluate_model, get_params, set_params
from src.ml.numpy.datasets import get_local_data
from src.utils import Map, create_tcp_socket, labels_set, get_ip_address


class Node(Thread):

    def __init__(self, manager):
        super(Node, self).__init__()
        self.id = None
        self.manager = manager
        self.host = get_ip_address()
        self.port = 0
        self.bridge = None
        self.model = None
        self.dataset_path = ""
        self.local_model = None
        self.V = {}
        self.current_round = 0
        self.current_exec = None
        self.start_train = False
        self.neighbors_ids = []
        self.neighbors = []
        self.clustered = None
        self.similarity = []
        self.train = None
        self.val = None
        self.test = None
        self.inference = None
        self.terminate = False
        self.params = Map()
        self.conn_attempts = 0
        # initialize network
        self._init_server()

    def run(self):
        while not self.terminate:
            try:
                conn, address = self.sock.accept()
                self.manager.get_screen("conf").connect_logs += f"{conn}/n{address}"
                if not self.terminate:
                    if self.bridge is None:
                        print("BRIDGE Joined")
                        bridge = NodeConnection(self, address[1], conn)
                        bridge.start()
                        self.bridge = bridge
                    else:
                        print(f"NEW NodeConnection: {address} // N:{len(self.neighbors)}")
                        neighbor_conn = NodeConnection(self, address[1], conn)
                        neighbor_conn.start()
                        self.neighbors.append(neighbor_conn)
            except socket.timeout:
                pass
            except Exception as e:
                toast(f"Node Exception: {e}")

        for neighbor in self.neighbors:
            neighbor.stop()
        self.sock.close()
        toast(f"Node Stopped")

    def log(self, typ, txt, remote=True):
        if remote:
            self.bridge.send(protocol.log(typ, txt))
        self.manager.get_screen("train").log(typ, txt)
        print(f"{typ.upper()} >> {txt}")

    def connect_bridge(self, bridge_host, bridge_port):
        try:
            sock = create_tcp_socket()
            sock.settimeout(SOCK_TIMEOUT)
            sock.connect((bridge_host, bridge_port))
            self.bridge = Bridge(self, sock)
            self.bridge.start()
            return True
        except Exception as e:
            toast(f"Cannot connect to bridge\n{e}")
            return False

    def connect(self, nid, host, port):
        try:
            if nid in [n.neighbor_id for n in self.neighbors]:
                return True
            sock = create_tcp_socket()
            sock.settimeout(SOCK_TIMEOUT)
            sock.connect((host, port))
            neighbor_conn = NodeConnection(self, nid, sock)
            neighbor_conn.start()
            neighbor_conn.send(protocol.connect(sock.getsockname(), self.id))
            self.neighbors.append(neighbor_conn)
            return {'s': True, 'm': f"Connected to Node({nid})"}
        except Exception as e:
            toast(f"Cannot connect to Node({nid}, <{host}, {port}>)\n{e}")
            print(f"Cannot connect to Node({nid}, <{host}, {port}>)\n{e}")
            return {'s': False, 'm': f"{self} could not connect to Node({nid}): {e}"}

    def disconnect(self, neighbor_conn):
        if not neighbor_conn.terminate:
            neighbor_conn.send(protocol.disconnect(self.id))
            neighbor_conn.terminate = True
            if neighbor_conn in self.neighbors:
                self.neighbors.remove(neighbor_conn)
            toast(f"Disconnected from {neighbor_conn}")

    def stop(self):
        for neighbor in self.neighbors:
            self.disconnect(neighbor)
        self.terminate = True
        self.sock.close()

    def send(self, neighbor, msg):
        neighbor.send(msg)

    def broadcast(self, msg, active=None):
        active = self.neighbors if active is None else active
        for neighbor in active:
            self.send(neighbor, msg)

    def fit(self, inference=True):
        # train the model
        history = model_fit(self)
        # set local model variable
        self.local_model = self.model
        # evaluate against a one batch or the whole inference dataset
        # history = None
        if inference:
            model_inference(self, one_batch=False)

        return history

    def train_one_epoch(self, batches=1, evaluate=False):
        return train_for_x_epoch(self, batches, evaluate)

    def evaluate(self, dataholder, one_batch=True):
        return evaluate_model(self.model, dataholder, one_batch)

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

    def get_model_params(self, named=False, numpy=False):
        return get_params(self.model, named=named)

    def set_model_params(self, params, named=False, numpy=False):
        return set_params(self.model, params, named=named)

    def set_weights(self, w):
        self.model.load_state_dict(deepcopy(w))

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
        self.port = self.sock.getsockname()[1]

    # Special methods
    def __repr__(self):
        return f"Device({self.id})"

    def __str__(self):
        return f"Device({self.id})"


class NodeConnection(Thread):
    def __init__(self, node, neighbor_id, sock):
        super(NodeConnection, self).__init__()
        self.node = node
        self.log = node.log
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
                        self.log('error', f"{self.node.name}: Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                self.log('error', f"{self.node}: Corrupted message : {e}")
            except socket.timeout:
                pass
            except struct.error as e:
                pass
            except Exception as e:
                self.terminate = True
                # todo remove node from list of connected neighbors
                self.log('error', f"{self.node} NodeConnection <{self.neighbor_name}> Exception\n{e}")
        self.sock.close()
        self.log('log', f"{self.node}: neighbor {self.neighbor_name} disconnected", remote=False)

    def send(self, msg):
        try:
            if self.terminate:
                self.log('log', f"{self} tries to send on terminated", remote=False)
            length = struct.pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            self.log('error', f"{self}: Socket error: {e}: ")
        except Exception as e:
            self.log('error', f"{self}: Exception\n{e}")

    def stop(self):
        self.terminate = True

    def handle_step(self, data):
        try:
            self.node.params.exchanges += 1
        except Exception as e:
            print("self.node.params.exchanges")
            print("handle_step()")
            exit()
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


class Bridge(Thread):
    def __init__(self, device: Node, sock):
        super(Bridge, self).__init__()
        self.device = device
        self.sock = sock
        self.log = device.log
        self.address = None
        self.learner = None
        self.terminate = False

    def run(self):
        # Wait for messages from device
        while not self.terminate:
            try:
                (length,) = struct.unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(4096000 if to_read > 4096000 else to_read)
                    if len(buffer) > 409600:
                        toast(f"Buffer={len(buffer)}")
                if buffer:
                    data = pickle.loads(buffer)
                    if data and data['mtype'] == protocol.CALL_METHOD:
                        self.call_method(data['data'])
                    elif data and data['mtype'] == protocol.DISCONNECT:
                        print("self.handle_disconnect()")
                        self.handle_disconnect(data['data'])
                    else:
                        print(f"Unknown type of message from bridge: {data['mtype']}")
            except pickle.UnpicklingError as e:
                print(f"Corrupted message from bridge : {e}")
            except socket.timeout as e:
                traceback.print_exc()
                print(f"ERROR: TIMIII {e}")
            except struct.error as e:
                pass
            except Exception as e:
                self.terminate = True
                traceback.print_exc()
                print(f"Bridge run Exception\n{e}")
        self.sock.close()
        print(f"Bridge disconnected")

    def send(self, msg):
        try:
            if self.terminate:
                self.log('log', f"Tries to send to bridge on terminated", remote=False)
            length = struct.pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            toast('error', f"Bridge Socket error: {e}: ")
        except Exception as e:
            toast('error', f"Bridge send Exception\n{e}")

    def send_pref(self, request_data, share_logs):
        msg = protocol.preferences({'host': self.device.host, 'port': self.device.port, 'request_data': request_data,
                                    'share_logs': share_logs})
        self.send(msg)

    def call_method(self, d):
        if d['method'] == "populate":
            self.method_populate(*d['args'], **d['kwargs'])
        elif d['method'] == "connect":
            self.method_connect(*d['args'], **d['kwargs'])
        if d['method'] == "fit":
            self.method_fit(*d['args'], **d['kwargs'])
        elif "execute" in d['method']:
            self.method_execute(d)

    def method_populate(self, info: dict):
        self.device.id = info['id']
        self.device.model = info['model']
        self.device.clustered = info['clustered']
        self.device.similarity = info['similarity']
        self.device.neighbors_ids = info['ids']
        if info.get('dataset', None):
            self.device.train = info['dataset']['train']
            self.device.val = info['dataset']['val']
            self.device.test = info['dataset']['test']
            self.device.inference = info['dataset']['inference']
        else:
            ds_duplicate = info.get('ds_duplicate', 1)
            num_users = info.get('num_users', 1)
            try:
                train, val, test, inference = get_local_data(self.device.dataset_path, num_users, ds_duplicate)
                self.device.train = train
                self.device.val = val
                self.device.test = test
                self.device.inference = inference
            except MemoryError as e:
                toast(f"MemoryError:  {e}")
            except Exception as e:
                toast(f"Exception:  {e}")
        if info.get('args', None):
            self.device.params = Map({
                'epochs': info['args']['epochs'],
                'frac': info['args']['frac'],
                'batch_size': info['args']['batch_size'],
                'lr': info['args']['lr'],
                'momentum': info['args']['momentum'],
                'gar': info['args']['gar'],
            })
        self.device.manager.get_screen("conf").log_pref()
        # self.device.manager.get_screen("conf").ids.join_btn.disabled = True

    def method_connect(self, nid, host, port):
        connected = self.device.connect(nid, host, port)
        self.send(protocol.return_method("connect", connected))

    def method_fit(self, inference):
        labels = labels_set(self.device.train)
        train_size = len(self.device.train.dataset)
        info = f"{self.device} is performing local training on {train_size} samples of labels {labels}."
        self.device.log('info', info)
        history = self.device.fit(inference)
        self.send(protocol.return_method("fit", {'s': True, 'm': history}))
        self.device.log('warning', "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -", remote=False)

    def method_execute(self, d):
        try:
            if self.learner is None:
                self.learner = import_module(f'src.learners.{ALGORITHM_MODULE}', __name__)
            func_name = getattr(self.learner, d['method'].replace('execute.', ''))
            func_name(self.device, *d['args'], **d['kwargs'])
        except ModuleNotFoundError as e:
            toast(f"call_method: {d['method']} >> {e}")
            self.send(protocol.return_method(d['method'], {'s': False, 'm': e}))
        self.send(protocol.return_method(d['method'], {'s': True}))

    def handle_disconnect(self, data):
        self.terminate = True
        self.device.bridge = None
        del self

    def stop(self):
        self.terminate = True

    def __repr__(self):
        return f"Bridge"

    def __str__(self):
        return f"Bridge"
