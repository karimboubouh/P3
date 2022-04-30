import argparse
import pickle
import random
import socket
import time
from inspect import getframeinfo, currentframe
from itertools import combinations

import numpy as np
import torch
from scipy.spatial import distance
from termcolor import cprint
from torch import linalg as LA

from src import conf
from src.conf import ML_ENGINE, TCP_SOCKET_BUFFER_SIZE
from src.helpers import Map

args: argparse.Namespace = None


def set_device(gpu):
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available() and gpu:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def exp_details(args):
    print('Experimental details:')
    print(f'    Model      : {args.model.upper()}')
    print(f'    Optimizer  : {args.optimizer}')
    print(f'    Learning   : {args.lr}')
    print(f'    Epochs     : {args.epochs}')
    print(f'    Batch size : {args.batch_size}')
    print('Collaborative learning parameters:')
    print(f'    Data distribution     : {"IID" if args.iid else "Non-IID"}')
    print(f'    Data size             : {"Unequal" if args.unequal else "Equal"} data size')
    print(f'    Test scope            : {args.test_scope}')
    print(f'    Number of peers       : {args.num_users}')
    print(f'    Rounds                : {args.rounds}')
    print(f'    Communication channel : {"TCP" if args.mp else "Shared memory"}')
    print(f'    Device                : {args.device}')
    print(f'    Seed                  : {args.seed}')
    log('info', f'Used ML engine: {ML_ENGINE}')

    return


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rounds', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of active neighbors')
    parser.add_argument('--gar', type=str, default='average',
                        help='Gradient Aggregation rule to use: \
                         average, median, krum, aksel')
    parser.add_argument('--epochs', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--mp', type=int, default=1,
                        help='Use message passing (MP) via sockets or shared \
                        memory (SM). Default set to MP. Set to 0 for SM.')
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--test_scope', type=str, default='global', help="test \
                        data scope (local, neighborhood, global)")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--verbose', type=int, default=2, help='verbose')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    global args
    args = parser.parse_args()
    return args


def load_conf():
    # sys.argv = ['']
    global args
    args = args_parser()
    args.device = set_device(args.gpu)
    return Map(vars(args))


def fixed_seed(fixed=True):
    global args
    if fixed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


def log(mtype, message):
    global args
    title = True
    if not mtype:
        title = False
        mtype = log.old_type
    log.old_type = mtype
    if args.verbose > -2:
        if mtype == "result":
            if title:
                cprint("\r Result:  ", 'blue', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'blue', end=' ')
            cprint(str(message), 'blue')
            log.old_type = 'result'
            return
    if args.verbose > -1:
        if mtype == "error":
            if title:
                cprint("\r Error:   ", 'red', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'red', end=' ')
            cprint(str(message), 'red')
            log.old_type = 'error'
            return
        elif mtype == "success":
            if title:
                cprint("\r Success: ", 'green', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'green', end=' ')
            cprint(str(message), 'green')
            log.old_type = 'success'
            return
    if args.verbose > 0:
        if mtype == "event":
            if title:
                cprint("\r Event:   ", 'cyan', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'cyan', end=' ')
            cprint(str(message), 'cyan')
            log.old_type = 'event'
            return
        elif mtype == "warning":
            if title:
                cprint("\r Warning: ", 'yellow', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", 'yellow', end=' ')
            cprint(str(message), 'yellow')
            log.old_type = 'warning'
            return
    if args.verbose > 1:
        if mtype == "info":
            if title:
                cprint("\r Info:    ", attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", end=' ')
            cprint(str(message))
            log.old_type = 'info'
            return
    if args.verbose > 2:
        if mtype not in ["info", "warning", "event", "success", "error", "result"]:
            if title:
                cprint("\r Log:     ", 'magenta', attrs=['reverse'], end=' ', flush=True)
            else:
                cprint("          ", end=' ')
            cprint(str(message))
            log.old_type = 'log'


def elog(message, mtype="log"):
    frameinfo = getframeinfo(currentframe().f_back)
    filename = frameinfo.filename.split('/')[-1]
    linenumber = frameinfo.lineno
    info = 'File: %s, line: %d' % (filename, linenumber)
    log(mtype, message)
    exit()


# def partition(x, nb_nodes, cluster_data=True, method="random", random_state=None):
#     M = x.shape[0]
#     print(M)
#     exit(0)
#     if cluster_data:
#         # method: kmeans or random.
#         gm = GaussianMixture(nb_nodes, init_params=method, random_state=random_state)
#         gm.fit(x)
#         labels = gm.predict(x)
#         print(type(labels))
#         groups = [[x[labels == i], y[labels == i]] for i in range(nb_nodes)]
#     else:
#         shuffled_ids = np.random.permutation(M)
#         s = M // nb_nodes
#         groups = [[x[shuffled_ids][i * s:(i + 1) * s], y[shuffled_ids][i * s:(i + 1) * s]] for i in range(nb_nodes)]
#
#     return groups


def cluster_peers(nb_nodes, k):
    shuffled_ids = np.random.permutation(nb_nodes)
    multi = np.random.multinomial(nb_nodes, np.ones(k) / k, size=10)
    s = None
    for m in multi:
        if (m < 2).sum() == 0:
            s = m
            break
    if s is None:
        exit('Error: cannot cluster peers.')
    clusters = {}
    step = 0
    for i in range(k):
        clusters[i] = shuffled_ids[step: step + s[i]]
        step += s[i]
    return clusters


def similarity_matrix(mask, clusters, rho=0.2, data=None):
    """Compute the similarity matrix randomly or according to users data"""
    if isinstance(mask, int):
        randm = True
        nb_nodes = mask
    else:
        randm = False
        nb_nodes = len(mask)
    combi = combinations(range(nb_nodes), 2)
    pairs = []
    for c in combi:
        for cluster in clusters.values():
            if np.in1d(c, cluster).sum() == 2:
                pairs.append(c)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # calculate similarity matrix
    for p in pairs:
        if randm:
            similarities[p] = similarities.T[p] = np.random.uniform(-1, 1)
        else:
            mask1 = np.sort(mask[p[0]])
            mask2 = np.sort(mask[p[1]])
            vec1 = data.targets[mask1].numpy()
            vec2 = data.targets[mask2].numpy()
            similarities[p] = similarities.T[p] = 1 - distance.cosine(vec1, vec2)
    # apply similarity threshold
    if rho is not None:
        sim_vals = similarities.copy()
        similarities[similarities < rho] = 0
        for i, sim in enumerate(similarities):
            if np.all(np.logical_not(sim)):
                if len(sim) > 0:
                    neighbor_id = random.choice(range(len(sim)))
                    sim[neighbor_id] = round(abs(sim_vals[i][neighbor_id]), 2)
                    log('warning', f"Node({i}) was attributed one peer: Node({neighbor_id}) with W={sim[neighbor_id]}")
                else:
                    log('error', f"The generated Random Graph is disconnected [s={rho}], Node({i}) has 0 neighbors.")
                    exit(0)

    # get adjacency matrix
    adjacency = similarities != 0

    return adjacency, similarities


def node_topology(i, topology):
    similarity = topology['similarities'][i]
    similarity = {key: value for key, value in enumerate(similarity) if value != 0}
    neighbors_ids = [j for j, adj in enumerate(topology['adjacency'][i]) if bool(adj) is True]
    return neighbors_ids, similarity


def optimizer_func(optim):
    optim = optim.lower()
    if optim == 'sgd':
        return torch.optim.SGD
    elif optim == 'adam':
        return torch.optim.Adam
    else:
        log('error', f"unsupported optimization algorithm: {optim}")
        return torch.optim.SGD


def verify_metrics(_metric, _measure):
    if _metric not in ['accuracy', 'loss']:
        log("error", f"Unknown metric: {_metric}")
        log("", f"Set metric to default: accuracy")
        metric = f"{conf.DEFAULT_VAL_DS}_acc"
    elif _metric == "accuracy":
        metric = f"{conf.DEFAULT_VAL_DS}_acc"
    else:
        metric = f"{conf.DEFAULT_VAL_DS}_loss"

    if _measure not in ['mean', 'mean-std', 'max', 'std']:
        log("error", f"Unknown {_metric} measure: {_measure}")
        log("", f"Set measure to default: mean")
        measure = "mean"
    else:
        measure = _measure
    return metric, measure


def fill_history(a):
    lens = np.array([len(item) for item in a.values()])
    log('info', f"Number of rounds performed by each peer:")
    log('', f"{lens}")
    ncols = lens.max()
    last_ele = np.array([a_i[-1] for a_i in a.values()])
    out = np.repeat(last_ele[:, None], ncols, axis=1)
    mask = lens[:, None] > np.arange(lens.max())
    out[mask] = np.concatenate(list(a.values()))
    out = {k: v for k, v in enumerate(out)}
    return out


def active_peers(peers, frac):
    m = max(int(frac * len(peers)), 1)
    return np.random.choice(peers, m, replace=False)


def wait_until(predicate, timeout=2, period=0.2, *args_, **kwargs):
    start_time = time.time()
    mustend = start_time + timeout
    while time.time() < mustend:
        if predicate(*args_, **kwargs):
            return True
        time.sleep(period)
    log("error", f"{predicate} finished after {time.time() - start_time} seconds.")
    return False


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def get_node_conn_by_id(node, node_id):
    for conn in node.neighbors:
        if conn.neighbor_id == node_id:
            return conn
    return None


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(10)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def labels_set(dataset):
    try:
        labels = set(dataset.train_labels_set)
    except AttributeError:
        classes = []
        for b in dataset:
            classes.extend(b[1].numpy())
        labels = set(classes)

    return labels


# def angular_metric(u, v):
#     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#     sim = cos(u, v)
#     angle = torch.rad2deg(torch.acos(sim)).item()
#     similarity = sim.item()
#     distance = 1 - similarity
#
#     return angle, similarity, distance


def save(filename, data):
    unique = np.random.randint(100, 999)
    filename = f"./out/{filename}_{unique}.pkl"
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("Writing to file", filename)
    return


def load(filename):
    with open(f"../out/{filename}", 'rb') as fp:
        return pickle.load(fp)


def norm_squared(vi, vj):
    if ML_ENGINE == "PyTorch":
        return LA.norm(vi - vj).item() ** 2
    else:
        fvi = np.concatenate([x.ravel() for x in vi])
        fvj = np.concatenate([x.ravel() for x in vj])
        return np.linalg.norm(fvi - fvj) ** 2
