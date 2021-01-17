import numpy as np
from itertools import combinations
from scipy.spatial import distance

from src.p2p import Node, Graph
from src.utils import cluster_peers, similarity_matrix, node_info, inference_ds, log


def random_graph(models, sigma=0.2, cluster_enabled=False, k=2):
    if sigma and sigma < 0:
        log('warning', f"Generating a negative similarity matrix.")
    # prob_edge = 1, rnd_state = None
    nb_nodes = len(models)
    if cluster_enabled:
        clusters = cluster_peers(nb_nodes, k)
    else:
        clusters = {0: np.arange(nb_nodes)}
    adjacency, similarities = similarity_matrix(nb_nodes, clusters, sigma)
    return {
        'clusters': clusters,
        'similarities': similarities,
        'adjacency': adjacency
    }


def random2_graph(models, train_ds):
    # test_ds, user_groups, prob_edge=1, cluster_data=True, rnd_state=None
    # similarities based on data /or/ random similarities based on clfs
    # generate adj matrix based on similarities
    # partition nodes if cluster_data is True

    nb_nodes = len(models)
    x = train_ds.data
    # y = train_ds.targets

    print(x[0].shape)
    print(nb_nodes)
    exit(0)

    # clustering
    # groups = partition(x, y, nb_nodes, cluster_data, random_state=rnd_state)
    # print(groups)
    # exit()
    # nodes = list()
    # for i in range(nb_nodes):
    #     n = Node(i, *groups[i])
    #     nodes.append(n)
    #
    # for i, n in enumerate(nodes):
    #     n = [n] + [nodes[j] for j in range(nb_nodes) if i != j and random() < prob_edge]
    #     n.set_neighbors(n, [1 / len(n)] * len(n))

    # return nodes


def datasim_network(data, sigma=0.2):
    """Compute graph matrices according to users data"""
    nb_nodes = len(data)

    pairs = combinations(range(nb_nodes), 2)
    similarities = np.zeros((nb_nodes, nb_nodes))
    # calculate similarity matrix
    for p in pairs:
        vec1 = np.hstack(data[p[0]])
        vec2 = np.hstack(data[p[1]])
        similarities[p] = similarities.T[p] = 1 - distance.cosine(vec1, vec2)
    # apply similarity threshold
    similarities[similarities < sigma] = 0
    # get adjacency matrix
    adjacency = similarities > 0

    return adjacency, similarities


def network_graph(topology, models, train_ds, test_ds, user_groups, args):
    nbr_nodes = len(user_groups)
    clustered = True if len(topology['clusters']) > 1 else False
    peers = list()
    for i in range(nbr_nodes):
        neighbors_ids, similarity, train, val, test = node_info(i, topology, train_ds, user_groups[i], args)
        data = {'train': train, 'val': val, 'test': test, 'inference': test_ds}
        peer = Node(i, models[i], data, neighbors_ids, clustered, similarity, args)
        peers.append(peer)
    for peer in peers:
        neighbors = [p for p in peers if p.id in peer.neighbors]
        peer.set_neighbors(neighbors)
    graph = Graph(peers, topology, test_ds, args)
    # Transformations
    graph.set_inference(args)

    return graph
