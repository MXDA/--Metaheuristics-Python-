import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

def insertion(route1, i, j):
    if i < j:
        route2 = copy.deepcopy(route1[0: i])
        route2 = np.append(route2, route1[i + 1: j + 1])
        route2 = np.append(route2, route1[i])
        route2 = np.append(route2, route1[j + 1:])
    else:
        route2 = copy.deepcopy(route1[0: j + 1])
        route2 = np.append(route2, route1[i])
        route2 = np.append(route2, route1[j + 1: i])
        route2 = np.append(route2, route1[i + 1:])
    return route2

a = np.array([1, 3, 2, 4, 5])
print(insertion(a, 3, 0))
print(np.abs(4 - 3))
i, j = np.random.permutation(5)[0: 2]
print(i, j)

a = np.array([[1, 3, 2, 4], [3, 0, -1, 1]])
b = np.min(a)
print(np.argwhere(a == b)[0])
c, d = np.argwhere(a == b)[0].ravel()
print(c, d)

a = np.array([3, 1, 2, 4, 6])
print(insertion(a, 3, 0))
print(a)
c = np.random.randint(4)
print(c)
print(np.random.choice(range(1, 5), 2))
date =range(10)

i, j = random.sample(range(5), 2)
print(i, j)
