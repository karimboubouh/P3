from math import pi
from pyJoules.energy_meter import measure_energy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def angular_metric_basic(u, v, metric="degree"):
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    gamma = np.dot(u, v) / norm
    if metric == "degree":
        alpha = np.degrees(np.arccos(np.clip(gamma, -1.0, 1.0)))
    elif metric == "rad":
        alpha = np.arccos(np.clip(gamma, -1.0, 1.0))
    else:
        raise Exception(f"Angle type {metric} unsupported")
    return alpha, gamma


def similarity_metric(u, v, backend="numpy"):
    u = u.reshape(1, u.shape[0])
    v = v.reshape(1, v.shape[0])
    if backend == "numpy":
        similarity = cosine_similarity(u, v).item()
        distance = 1 - similarity
        angle = np.degrees(np.arccos(np.clip(similarity, -1.0, 1.0)))
    elif backend == "pytorch":
        tu = torch.from_numpy(u).type(torch.FloatTensor)
        tv = torch.from_numpy(v).type(torch.FloatTensor)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(tu, tv)
        angle = torch.rad2deg(torch.acos(sim)).item()
        similarity = sim.item()
        distance = 1 - similarity
    else:
        raise Exception(f"Backend {backend} unsupported")
    return angle, similarity, distance


def angular_metric(u, v, metric="cosine"):
    """
    Angular metric: calculates the angle and distance between two
    vectors using different distance metrics: euclidean, cosine,
    Triangle's Area Similarity (TS), Sector's Area Similarity (SS),
    and TS-SS. More details in the paper at
    https://github.com/taki0112/Vector_Similarity
    :param u: 1D Tensor
    :param v: 1D Tensor
    :param metric: choices are: cosine, euclidean, TS, SS, TS-SS
    :return: angle, distance
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(u, v)
    rad = torch.acos(similarity)
    angle = torch.rad2deg(rad).item()
    if metric == "cosine":
        distance = 1 - similarity.item()
    elif metric == "euclidean":
        distance = torch.cdist(u, v).item()
    elif metric == "TS":
        # Triangle's Area Similarity (TS)
        rad_10 = torch.deg2rad(rad + torch.deg2rad(torch.tensor(10.0)))
        distance = (torch.norm(u) * torch.norm(v)) * torch.sin(rad_10) / 2
        distance = distance.item()
    elif metric == "SS":
        # Sector's Area Similarity (SS)
        ed_md = torch.cdist(u, v) + torch.abs(torch.norm(u) - torch.norm(v))
        rad_10 = rad + torch.deg2rad(torch.tensor(10.0))
        distance = pi * torch.pow(ed_md, 2) * rad_10 / 360
        distance = distance.item()
    elif metric == "TS-SS":
        _, triangle = angular_metric(u, v, metric="TS")
        _, sector = angular_metric(u, v, metric="SS")
        distance = triangle * sector
    else:
        raise Exception(f"Distance metric {metric} unsupported")

    return angle, distance


@measure_energy
def run():
    torch.device('cpu')
    x = torch.Tensor([[1, 1]])
    y = torch.Tensor([[-2, 5]])
    for i in range(100):
        a, d = angular_metric(x, y, metric="TS-SS")
    print(f"Distance = {d}, Angle = {a}°")


if __name__ == '__main__':
    run()
