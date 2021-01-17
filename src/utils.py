import sys
import time
from functools import partial

import numpy as np
import argparse
import torch

from random import shuffle
from termcolor import cprint
from torch.utils.data.dataloader import DataLoader, default_collate
from itertools import combinations
from scipy.spatial import distance

from src import conf
from src.conf import TRAIN_VAL_TEST_RATIO, INFERENCE_BATCH_SIZE, TEST_SCOPE
from src.helpers import DatasetSplit, Map

args: argparse.Namespace = None


def set_device(args):
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available() and args.gpu:
        print('    Device             : Cuda')
        return torch.device('cuda')
    else:
        print('    Device             : CPU')
        return torch.device('cpu')


def exp_details(args):
    print('Experimental details:')
    print(f'    Model      : {args.model.upper()}')
    print(f'    Optimizer  : {args.optimizer}')
    print(f'    Learning   : {args.lr}')
    print(f'    Epochs     : {args.epochs}')
    print(f'    Batch size : {args.batch_size}')
    print('Collaborative learning parameters:')
    iid = 'IID' if args.iid else 'Non-IID'
    size = 'Unequal' if args.unequal else 'Equal'
    print(f'    Data distribution  : {iid}')
    print(f'    Data size          : {size} data size')
    print(f'    Test scope         : {args.test_scope}')
    print(f'    Number of peers    : {args.num_users}')
    print(f'    Rounds             : {args.rounds}')

    return


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--epochs', type=int, default=4,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for imagenet.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--test_scope', type=str, default='local', help="test \
                        data scope (local, neighborhood, global)")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=10, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    global args
    args = parser.parse_args()
    return args


def load_conf():
    # sys.argv = ['']
    global args
    args = args_parser()
    return Map(vars(args))


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
                cprint(" Result:  ", 'blue', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'blue', end=' ')
            cprint(message, 'blue')
            log.old_type = 'result'
            return
    if args.verbose > -1:
        if mtype == "error":
            if title:
                cprint(" Error:   ", 'red', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'red', end=' ')
            cprint(message, 'red')
            log.old_type = 'error'
            return
        elif mtype == "success":
            if title:
                cprint(" Success: ", 'green', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'green', end=' ')
            cprint(message, 'green')
            log.old_type = 'success'
            return
    if args.verbose > 0:
        if mtype == "event":
            if title:
                cprint(" Event:   ", 'cyan', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'cyan', end=' ')
            cprint(message, 'cyan')
            log.old_type = 'event'
            return
        elif mtype == "warning":
            if title:
                cprint(" Warning: ", 'yellow', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'yellow', end=' ')
            cprint(message, 'yellow')
            log.old_type = 'warning'
            return
    if args.verbose > 1:
        if mtype == "info":
            if title:
                cprint(" Info:    ", attrs=['reverse'], end=' ')
            else:
                cprint("          ", end=' ')
            cprint(message)
            log.old_type = 'info'
            return
        else:
            if title:
                cprint(" Log:     ", 'magenta', attrs=['reverse'], end=' ')
            else:
                cprint("          ", 'magenta', end=' ')
            log.old_type = 'log'


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


def similarity_matrix(data, clusters, sigma=0.2):
    """Compute the similarity matrix randomly or according to users data"""
    if isinstance(data, int):
        random = True
        nb_nodes = data
    else:
        random = False
        nb_nodes = len(data)
    combi = combinations(range(nb_nodes), 2)
    pairs = []
    for c in combi:
        for cluster in clusters.values():
            if np.in1d(c, cluster).sum() == 2:
                pairs.append(c)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # calculate similarity matrix
    for p in pairs:
        if random:
            similarities[p] = similarities.T[p] = np.random.uniform(-1, 1)
        else:
            vec1 = np.hstack(data[p[0]])
            vec2 = np.hstack(data[p[1]])
            similarities[p] = similarities.T[p] = 1 - distance.cosine(vec1, vec2)
    # apply similarity threshold
    if sigma is not None:
        similarities[similarities < sigma] = 0
    # get adjacency matrix
    adjacency = similarities != 0

    return adjacency, similarities


def node_info(i, topology, train_ds, data_mask, args):
    similarity = topology['similarities'][i]
    similarity = {key: value for key, value in enumerate(similarity) if value != 0}
    neighbors_ids = [j for j, adj in enumerate(topology['adjacency'][i]) if bool(adj) is True]
    train, val, test = train_val_test(train_ds, data_mask, args)

    return neighbors_ids, similarity, train, val, test


def train_val_test(train_ds, mask, args, ratio=None):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    ratio = TRAIN_VAL_TEST_RATIO if ratio is None else ratio
    mask = list(mask)
    shuffle(mask)
    assert np.sum(ratio) == 1, "Ratio between train, dev and test must sum to 1."
    v1 = int(ratio[0] * len(mask))
    v2 = int((ratio[0] + ratio[1]) * len(mask))
    # split indexes for train, validation, and test (80, 10, 10)
    train_mask = mask[:v1]
    val_mask = mask[v1:v2]
    test_mask = mask[v2:]
    # create data loaders
    train_loader = DataLoader(DatasetSplit(train_ds, train_mask), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DatasetSplit(train_ds, val_mask), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(DatasetSplit(train_ds, test_mask), batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def inference_ds(peer, args):
    # global_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test = None
    if args.test_scope == 'global':
        test = DataLoader(peer.inference, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)
    elif args.test_scope == 'neighborhood':
        collate = partial(collate_fn, scope=peer.neighborhood_data_scope())
        test = DataLoader(peer.inference, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, collate_fn=collate)
    elif args.test_scope == 'local':
        collate = partial(collate_fn, scope=peer.local_data_scope())
        test = DataLoader(peer.inference, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, collate_fn=collate)
    else:
        exit('Error: unrecognized TEST_SCOPE value')

    return test


def collate_fn(batch, scope):
    modified_batch = []
    for item in batch:
        image, label = item
        if label in scope:
            modified_batch.append(item)
    return default_collate(modified_batch)


def optimizer_func(optim):
    optim = optim.lower()
    if optim == 'sgd':
        return torch.optim.SGD
    elif optim == 'adam':
        return torch.optim.Adam
    else:
        # todo log unsupported optimized
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

    if _measure not in ['mean', 'max', 'std']:
        log("error", f"Unknown {_metric} measure: {_measure}")
        log("", f"Set measure to default: mean")
        measure = "mean"
    else:
        measure = _measure
    return metric, measure


def get_neighbors_by_ids(peers, ids):
    return
    return None


def inference_eval(peer, device, one_batch=False):
    t = time.time()
    r = peer.model.evaluate(peer.inference, device, one_batch)
    o = "I" if one_batch else "*"
    acc = round(r['val_acc'] * 100, 2)
    loss = round(r['val_loss'], 2)
    t = round(time.time() - t, 1)
    log('result', f"Node {peer.id} [{t}s]{o} Inference acc: {acc}%,  loss: {loss}")


def estimate_shards(data_size, num_users):
    shards = num_users * 2 if num_users > 10 else 20
    imgs = int(data_size / shards)

    return shards, imgs


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
