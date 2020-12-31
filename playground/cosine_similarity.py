import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

u = np.array([[3, 1, 2]])
v = np.array([[1, 4, 3]])

output = cosine_similarity(u, v).item()
acos = math.degrees(math.acos(output))

print(f"sklearn: {output} / {acos}º")

tu = torch.from_numpy(u).type(torch.FloatTensor)
tv = torch.from_numpy(v).type(torch.FloatTensor)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(tu, tv)
acos = torch.rad2deg(torch.acos(output)).item()

print(f"torch: {output.item()} / {acos}º")

output = F.cosine_similarity(tu, tv).item()

print(f"F: {output}")

# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#
# print(cos(tu, tv))

# angle = math.degrees(math.acos(cos))
# print(angle)
