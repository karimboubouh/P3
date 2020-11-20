import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture

from src.helpers import Map


def partition(x, y, nb_nodes, cluster_data=True, method="random", random_state=None):
    M, _ = x.shape

    if cluster_data:
        gm = GaussianMixture(nb_nodes, init_params=method, random_state=random_state)
        gm.fit(x)
        labels = gm.predict(x)
        groups = [[x[labels == i], y[labels == i]] for i in range(nb_nodes)]

    else:
        shuffled_ids = np.random.permutation(M)
        print(shuffled_ids)
        s = M // nb_nodes
        print(M, s)
        groups = [[x[shuffled_ids][i * s:(i + 1) * s], y[shuffled_ids][i * s:(i + 1) * s]] for i in range(nb_nodes)]

    return groups


def cosine(vec1, vec2):
    print(vec1)
    vec1 = np.array([vec1.flatten()])
    print(vec1)
    vec2 = np.array([vec2.flatten()])
    print(vec2)
    cs = cosine_similarity(vec1, vec2)
    return cs  # , (1 - cs)


def cosined(vec1, vec2):
    vec1 = np.array([[0, 3, 0, 4, 1], [5, 5, 5, 5, 5]])
    vec2 = np.array([[0, 3, 0, 4]])
    return cosine_similarity(vec1, vec2)


def sim_map(arr, sigma):
    """Function used to map [-1,1] into [0,2]
    (length of arc between two points on the unit circle)"""
    return np.exp(-((1 - arr) ** 2 + (1 - arr ** 2)) / (2 * sigma))


def get_adj_matrix(similarities, eps=1e-3):
    thresholds = similarities.max() * np.sqrt(10) ** (- np.arange(1, 100))
    for thresh in thresholds:
        adjacency = similarities > thresh
        if np.abs(np.linalg.eigvalsh(np.diag(adjacency.sum(axis=1)) - adjacency)[1]) > eps:
            break
    return adjacency


def compute_adjacencies(clfs, n, sigma=0.1):
    """Compute graph matrices according to true models of agents"""

    pairs = list(zip(*combinations(range(n), 2)))
    similarities = np.zeros((n, n))
    norms_ = np.linalg.norm(clfs, axis=1)
    similarities[pairs] = similarities.T[pairs] = (
            (clfs[pairs[0],] * clfs[pairs[1],]).sum(axis=1) / (norms_[pairs[0],] * norms_[pairs[1],]))

    similarities = sim_map(similarities, sigma)
    similarities[np.diag_indices(n)] = 0

    adjacency = get_adj_matrix(similarities)

    return adjacency, similarities


def sim_threshold(sim, thresh):
    sim[sim < thresh] = 0
    return sim


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


if __name__ == '__main__':
    x = Map()
    y = Map()
    z = Map()
    y.me = "Hii"
    print(y)
    x.me = 'hello'

    print(x)

    print(Map(dict(x, **z)))
    # x = [.8, .1, .14]

    # if condition returns False, AssertionError is raised:

    # assert np.sum(x) == 1, "x should be 'hello'"
    #
    # print("Hiii")
    exit(0)
    data = [np.array([[1, 2], [1, 3]]), np.array([[9, 5], [-1, -13]]), np.array([[1, 6], [3, 3]]),
            np.array([[1, 5], [3, 3]])]
    clfs = np.array([[1, 2], [1, 3], [9, 5], [-1, -13], [0, 0], [1, 2], [1, 5], [3, 3]])
    # clfs = np.array([[1, 2, 5], [1, 3, 5], [1, 4, 5.4], [5, 2, 5], [3, 3, 3], [0, -3, 3]])
    # compute_adjacencies(clfs, 4)
    a = adjacencies(data, sigma=0)
    print(a)
    # print(s)

    # x1 = np.array([[1, 2], [1, 2]])
    # x2 = np.array([[0.44, 2.5], [0.44, 2.5]])
    # x3 = np.array([[1, 2], [1, 2]])
    # x = np.array([x1, x2, x3])
    # clfs = np.array([[1, 2], [1, 3], [1, 4], [5, 2], [6, 2], [7, 2]])
    #
    # a = compute_adjacencies(x, 3)
    # print(a)

    # x = np.array([[1, 2], [1, 3], [5, 6], [5, 7], [9, 8], [1, 1]])
    # y = np.array([1, 1, 2, 2, 2, 1])
    # g = partition(x, y, 3, cluster_data=False)
    # print(g)
    # c = cosi`ned(x, y)
    # x = x.reshape(-1, 1)
    # print(x.shape)
    # print(c)
    # using sklearn to calculate cosine similarity
    # cos_sim = cosine_similarity(x, y)
    # print(f"Cosine Similarity between A and B:{cos_sim.flatten()}")
    # print(f"Cosine Distance between A and B:{1 - cos_sim}")
