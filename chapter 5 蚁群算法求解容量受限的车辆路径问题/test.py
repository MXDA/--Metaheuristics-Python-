import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import math
import copy
import os

def roulette(p_value):
    r = np.random.rand()
    c = copy.deepcopy(p_value)
    for i in range(np.size(c) - 1):
        c[i + 1] += c[i]
    print(c, r)
    for i in range(np.size(c)):
        if c[i] >= r:
            return i

p = np.array([0.1, 0.2, 0.4, 0.3])

c = roulette(p)
print(c)